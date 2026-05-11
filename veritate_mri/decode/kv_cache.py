# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - KVCachedDecoder: classic transformer KV cache for Veritate inference.
#   Per-step decode goes from O(T^2) (re-encode prompt+gen every step) to O(T)
#   (only encode the new token; attention runs against cached K/V).
#
# - Works with both model variants:
#     * veritate.model.Veritate            — canonical, learned abs pos_emb
#     * plugins.veritate_800m.Veritate800M — RoPE, MTP head
#   Detected by the presence of `pos_emb` vs `rope_cos`.
#
# - Cache layout: per layer, `[B, heads, max_T, head_dim]` for K and V,
#   pre-allocated. `length` tracks how many slots are valid. K stored
#   POST-RoPE-rotation for the RoPE variant — the rotation is position-
#   dependent and we want to bake it into the cached value so attention
#   against the cache uses the correct rotated key.
#
# - Monkey-patches `CausalSelfAttention.forward` and
#   `CausalSelfAttentionRoPE.forward` inside a context manager; restores on
#   exit. Same pattern as experiments/streaming/streaming_decoder.py. The
#   canonical model file and the 800M plugin are not touched on disk.
#
# - Position handling:
#     * canonical: we override the embedding step to pass explicit positions
#       (`prefill` uses 0..T-1, `decode_one` uses `pos = length-1` for the new
#       byte). Attention uses `is_causal=False` against the cache because the
#       cache only contains past tokens by construction.
#     * RoPE 800M: the patched attention slices `rope_cos[start:start+T]` and
#       `rope_sin[start:start+T]` for the new tokens so the rotation matches
#       their absolute positions. The cache holds the post-RoPE K.
# veritate_mri/decode/kv_cache.py
# ------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------------
# Per-layer cache

class _LayerKV:
    """Pre-allocated `[B, heads, max_T, head_dim]` K/V cache for one block."""

    def __init__(self, B: int, heads: int, head_dim: int, max_T: int,
                 device: torch.device, dtype: torch.dtype):
        self.B = B
        self.heads = heads
        self.head_dim = head_dim
        self.max_T = max_T
        self.device = device
        self.dtype = dtype
        self.k = torch.zeros(B, heads, max_T, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(B, heads, max_T, head_dim, device=device, dtype=dtype)
        self.length = 0  # number of valid slots in [0, length)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """k_new / v_new: `[B, heads, T_new, head_dim]`."""
        T_new = k_new.size(2)
        if self.length + T_new > self.max_T:
            raise RuntimeError(
                f"KV cache overflow: length={self.length} + T_new={T_new} > max_T={self.max_T}"
            )
        end = self.length + T_new
        self.k[:, :, self.length:end, :] = k_new.to(self.dtype)
        self.v[:, :, self.length:end, :] = v_new.to(self.dtype)
        self.length = end

    def view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.k[:, :, :self.length, :], self.v[:, :, :self.length, :])

    def bytes(self) -> int:
        per_el = self.k.element_size()
        return 2 * self.B * self.heads * self.length * self.head_dim * per_el


# ------------------------------------------------------------------------------------
# Decoder

class KVCachedDecoder:
    """Wrap a Veritate / Veritate800M model with a KV-cached decode path.

    Usage:
        dec = KVCachedDecoder(model, max_T=1024)
        with dec.cached():
            logits_last = dec.prefill(prompt_ids)              # [B, vocab]
            for _ in range(max_new):
                nxt = int(logits_last.argmax(-1).item())
                logits_last = dec.decode_one(nxt)
                ...
    """

    # ----- detection ---------------------------------------------------------
    @staticmethod
    def _detect_variant(model) -> str:
        has_rope = hasattr(model, "rope_cos") and hasattr(model, "rope_sin")
        has_mtp  = hasattr(model, "mtp")
        if has_rope and has_mtp:
            return "rope_mtp"
        if has_rope:
            return "rope_no_mtp"
        if hasattr(model, "pos_emb"):
            return "abs_pos"
        raise TypeError(f"unsupported model: no pos_emb and no rope_cos: {type(model).__name__}")

    # ----- ctor --------------------------------------------------------------
    def __init__(self, model, max_T: Optional[int] = None, B: int = 1):
        self.model = model
        self.variant = self._detect_variant(model)
        self.B = B
        self.heads = model.heads
        self.head_dim = model.hidden // model.heads
        self.n_layers = model.layers
        self.seq = model.seq
        if max_T is None:
            max_T = self.seq
        self.max_T = max_T

        device = next(model.parameters()).device
        # Cache dtype = whatever the attention activations naturally live in.
        # During inference without autocast that's the param dtype.
        dtype = next(model.parameters()).dtype
        self.device = device
        self.dtype = dtype

        self.caches: List[_LayerKV] = [
            _LayerKV(B, self.heads, self.head_dim, max_T, device, dtype)
            for _ in range(self.n_layers)
        ]

    # ----- public ------------------------------------------------------------
    @contextlib.contextmanager
    def cached(self):
        """Install patched attention forwards, restore on exit."""
        if self.variant == "abs_pos":
            with self._patched_abs_pos():
                yield self
        else:
            # rope_mtp and rope_no_mtp share the same attention patch path —
            # the only difference is which head produces byte-0 logits (mtp.lm_head
            # vs lm_head), handled in _forward_chunk below.
            with self._patched_rope():
                yield self

    def reset(self) -> None:
        for c in self.caches:
            c.length = 0

    def cache_lengths(self) -> List[int]:
        return [c.length for c in self.caches]

    def cache_total_bytes(self) -> int:
        return sum(c.bytes() for c in self.caches)

    # ----- prefill / decode --------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the prompt through the model, fill caches, return logits for
        the LAST position only. `ids`: `[B, T]` (long). Returns `[B, vocab]`."""
        assert ids.dim() == 2, f"expected [B, T], got {tuple(ids.shape)}"
        assert ids.size(0) == self.B, f"batch mismatch: cache B={self.B}, ids B={ids.size(0)}"
        self.reset()
        logits_all = self._forward_chunk(ids, start_pos=0)
        # logits_all: [B, T, vocab]
        return logits_all[:, -1, :]

    @torch.no_grad()
    def decode_one(self, byte_id: int) -> torch.Tensor:
        """Append one byte to the cache. Returns `[B, vocab]` for the next-byte
        prediction at the appended position."""
        start = self.caches[0].length
        if start >= self.max_T:
            raise RuntimeError(
                f"decode_one called when cache is full ({start} == max_T={self.max_T})"
            )
        ids = torch.full((self.B, 1), int(byte_id), dtype=torch.long, device=self.device)
        logits = self._forward_chunk(ids, start_pos=start)
        return logits[:, -1, :]

    # ----- shared forward ----------------------------------------------------
    def _forward_chunk(self, ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Forward `[B, T_new]` token IDs that occupy absolute positions
        `[start_pos, start_pos + T_new)`. Appends to caches in the patched
        attention. Returns `[B, T_new, vocab]`.
        """
        m = self.model
        B, T_new = ids.shape

        if self.variant == "abs_pos":
            # Manual embed with explicit positions so we don't run into
            # `T > seq` on `embed()` or the wrong starting offset.
            positions = torch.arange(start_pos, start_pos + T_new,
                                     device=ids.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(B, T_new)
            x = m.tok_emb(ids) + m.pos_emb(positions)
            # Run blocks. Each block.attn is patched to consult its cache.
            self._chunk_start_pos = start_pos
            for blk in m.blocks:
                x = blk(x)
            x = m.n_out(x)
            logits = m.lm_head(x)
            return logits

        # rope_mtp / rope_no_mtp variants. Same attention path; only the final
        # output projection differs.
        need = start_pos + T_new
        if need > m.rope_cos.size(0):
            m.extend_rope(need)
        x = m.embed(ids)
        # Stash start_pos for the patched attention. The block's attn.forward
        # is patched to slice rope_cos/rope_sin from start_pos onwards.
        self._chunk_start_pos = start_pos
        for blk in m.blocks:
            x = blk(x, m.rope_cos, m.rope_sin)
        x = m.n_out(x)
        if self.variant == "rope_mtp":
            # Veritate800M: byte-0 logits route through MTP head-0 to match
            # what model.forward() returns.
            head0  = m.mtp.norms[0](m.mtp.transforms[0](x))
            logits = m.mtp.lm_head(head0)
        else:
            # rope_no_mtp (VeritateRoPE): standard tied lm_head.
            logits = m.lm_head(x)
        return logits

    # ----- monkey-patches ----------------------------------------------------
    @contextlib.contextmanager
    def _patched_abs_pos(self):
        """Patch CausalSelfAttention.forward on every block."""
        originals = []
        try:
            for i, blk in enumerate(self.model.blocks):
                attn = blk.attn
                originals.append(attn.forward)
                cache = self.caches[i]

                def make_fwd(attn_mod=attn, cache=cache):
                    def fwd(x):
                        B, T, C = x.shape
                        h, d = attn_mod.h, attn_mod.d
                        qkv = attn_mod.qkv(x).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        # Append new K/V to the cache.
                        cache.append(k, v)
                        K, V = cache.view()
                        # Causal mask: q positions are [start, start+T). K
                        # contains positions [0, start+T). We need q_i to
                        # attend to k_j iff j <= start + i.
                        # When T == 1 (decode), q always attends to all K and
                        # no mask is needed.
                        # When T > 1 (prefill), we need a causal mask
                        # restricted to the "new" rows.
                        if T == 1:
                            out = F.scaled_dot_product_attention(
                                q, K, V, is_causal=False
                            )
                        else:
                            # Build a [T, S] attention mask where S = K.size(2).
                            # row i corresponds to absolute pos (start + i),
                            # col j corresponds to absolute pos j.
                            # Allow j <= start + i.
                            S = K.size(2)
                            start = K.size(2) - T  # == cache.length pre-append? no: K is post-append. Use cache.length - T.
                            # cache.length already equals start+T after append.
                            # So absolute start = cache.length - T.
                            abs_start = cache.length - T
                            i_idx = torch.arange(T, device=x.device).view(T, 1) + abs_start
                            j_idx = torch.arange(S, device=x.device).view(1, S)
                            mask = (j_idx <= i_idx)  # bool, [T, S]
                            attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
                            attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
                            out = F.scaled_dot_product_attention(
                                q, K, V, attn_mask=attn_mask, is_causal=False
                            )
                        out = out.transpose(1, 2).contiguous().view(B, T, C)
                        return attn_mod.proj(out)
                    return fwd

                attn.forward = make_fwd()
            yield
        finally:
            for blk, fwd in zip(self.model.blocks, originals):
                blk.attn.forward = fwd

    @contextlib.contextmanager
    def _patched_rope(self):
        """Patch CausalSelfAttentionRoPE.forward on every block.

        The patched forward needs to know the absolute starting position of
        the current chunk to slice the right RoPE rows. We read that from
        `self._chunk_start_pos`, which `_forward_chunk` sets before calling
        the blocks.
        """
        # Resolve `apply_rope` from whatever module the model class lives in.
        # Both `plugins/veritate_800m/plugin.py` and `veritate/model_rope.py`
        # define `apply_rope(x, cos, sin)` at module scope. This avoids the
        # old `from plugin import apply_rope` indirection.
        import sys as _sys
        _model_mod = _sys.modules.get(type(self.model).__module__)
        apply_rope = getattr(_model_mod, "apply_rope", None)
        if apply_rope is None:
            raise RuntimeError(
                f"KV cache: cannot find apply_rope in {type(self.model).__module__}; "
                "the RoPE model module must export apply_rope(x, cos, sin)."
            )

        decoder_self = self
        originals = []
        try:
            for i, blk in enumerate(self.model.blocks):
                attn = blk.attn
                originals.append(attn.forward)
                cache = self.caches[i]

                def make_fwd(attn_mod=attn, cache=cache):
                    def fwd(x, rope_cos, rope_sin):
                        B, T, C = x.shape
                        h, d = attn_mod.h, attn_mod.d
                        qkv = attn_mod.qkv(x).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        start = decoder_self._chunk_start_pos
                        # Slice the RoPE rows for absolute positions
                        # [start, start+T). apply_rope reads x.size(-2) rows
                        # from the (offset) tables.
                        cos_slice = rope_cos[start:start + T].to(q.dtype)
                        sin_slice = rope_sin[start:start + T].to(q.dtype)
                        q = apply_rope(q, cos_slice, sin_slice)
                        k = apply_rope(k, cos_slice, sin_slice)
                        # Append POST-rotated K and raw V to cache.
                        cache.append(k, v)
                        K, V = cache.view()
                        if T == 1:
                            out = F.scaled_dot_product_attention(
                                q, K, V, is_causal=False
                            )
                        else:
                            S = K.size(2)
                            abs_start = cache.length - T
                            i_idx = torch.arange(T, device=x.device).view(T, 1) + abs_start
                            j_idx = torch.arange(S, device=x.device).view(1, S)
                            mask = (j_idx <= i_idx)
                            attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
                            attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
                            out = F.scaled_dot_product_attention(
                                q, K, V, attn_mask=attn_mask, is_causal=False
                            )
                        out = out.transpose(1, 2).contiguous().view(B, T, C)
                        return attn_mod.proj(out)
                    return fwd

                attn.forward = make_fwd()
            yield
        finally:
            for blk, fwd in zip(self.model.blocks, originals):
                blk.attn.forward = fwd
