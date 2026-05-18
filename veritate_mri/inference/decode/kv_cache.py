# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - KVCachedDecoder: per-step decode via cached K/V. Model-agnostic: every block
#   patch is built by the model itself (model.kv_cache_patch_attn). No variant
#   sniffing in this module (preflight rule 11a).
# - Cache layout: per layer, [B, heads, max_T, head_dim] for K and V,
#   pre-allocated. length tracks how many slots are valid.
# - The model decides whether K is stored pre-rotation or post-rotation; the
#   cache is opaque storage. The model's patch is the source of truth.
# veritate_mri/decode/kv_cache.py
# ------------------------------------------------------------------------------------
# Imports:

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

class _LayerKV:
    """Pre-allocated [B, heads, max_T, head_dim] K/V cache for one block."""

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
        self.length = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
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


class KVCachedDecoder:
    """Wrap any Veritate model with a KV-cached decode path. The model exposes
    the cross-model contract (embed, run_blocks, ensure_context, project_byte0,
    kv_cache_patch_attn); this decoder calls it blindly."""

    def __init__(self, model, max_T: Optional[int] = None, B: int = 1):
        self.model = model
        self.B = B
        self.heads = model.heads
        self.head_dim = model.hidden // model.heads
        self.n_layers = model.layers
        self.seq = model.seq
        if max_T is None:
            max_T = self.seq
        self.max_T = max_T

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.device = device
        self.dtype = dtype
        self._chunk_start_pos = 0

        self.caches: List[_LayerKV] = [
            _LayerKV(B, self.heads, self.head_dim, max_T, device, dtype)
            for _ in range(self.n_layers)
        ]

    @contextlib.contextmanager
    def cached(self):
        originals = []
        get_start = lambda: self._chunk_start_pos
        try:
            for i, blk in enumerate(self.model.blocks):
                attn = blk.attn
                originals.append(attn.forward)
                cache = self.caches[i]
                attn.forward = self.model.kv_cache_patch_attn(attn, cache, get_start)
            yield self
        finally:
            for blk, fwd in zip(self.model.blocks, originals):
                blk.attn.forward = fwd

    def reset(self) -> None:
        for c in self.caches:
            c.length = 0

    def cache_lengths(self) -> List[int]:
        return [c.length for c in self.caches]

    def cache_total_bytes(self) -> int:
        return sum(c.bytes() for c in self.caches)

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        assert ids.dim() == 2, f"expected [B, T], got {tuple(ids.shape)}"
        assert ids.size(0) == self.B, f"batch mismatch: cache B={self.B}, ids B={ids.size(0)}"
        self.reset()
        logits_all = self._forward_chunk(ids, start_pos=0)
        return logits_all[:, -1, :]

    @torch.no_grad()
    def decode_one(self, byte_id: int) -> torch.Tensor:
        start = self.caches[0].length
        if start >= self.max_T:
            raise RuntimeError(
                f"decode_one called when cache is full ({start} == max_T={self.max_T})"
            )
        ids = torch.full((self.B, 1), int(byte_id), dtype=torch.long, device=self.device)
        logits = self._forward_chunk(ids, start_pos=start)
        return logits[:, -1, :]

    def _forward_chunk(self, ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        m = self.model
        B, T_new = ids.shape
        m.ensure_context(start_pos + T_new)
        self._chunk_start_pos = start_pos
        x = m.embed(ids, start_pos=start_pos)
        x = m.run_blocks(x, start_pos=start_pos)
        return m.project_byte0(x)
