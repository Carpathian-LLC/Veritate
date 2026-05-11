# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MTPDecoder: a pure wrapper around a Veritate800M (or any model whose .mtp
#   submodule emits [B, T, N, vocab] logits over the next N bytes). NEVER
#   modifies the model. Three decoding strategies share the same forward path,
#   so wall-clock comparisons are apples-to-apples.
# - The wrapper does not maintain a KV cache. Each forward pass runs over the
#   full (rolling) context window, exactly how Brain.stream() does it today, so
#   the speedup numbers are directly comparable to the dashboard's measured
#   ms/byte.
# - This module is the inference plumbing the 800M's MTP head needs at decode
#   time. Wired into Brain.stream() via the `?fast=mtp` request mode in app.py.
# veritate_mri/decode/mtp_decode.py
# ------------------------------------------------------------------------------------

import time

import torch


class MTPDecoder:
    """Wraps a Veritate800M-like model with three multi-byte decode modes.

    The model must expose:
      - .seq (int)             : trained context length
      - .n_predict (int)       : number of MTP heads (>= 1)
      - .embed(tokens)         : trunk embedding entry
      - .blocks (iterable)     : list of attention blocks accepting (x, cos, sin)
      - .n_out (Module)        : final RMSNorm
      - .mtp(h) -> logits      : [B, T, N, vocab]
      - .rope_cos / .rope_sin  : RoPE buffers
      - .extend_rope(new_max_seq) (optional, only triggered if T > rope cache)
      - .forward(tokens) -> (logits_b0, _)  -- not used by us but the canonical
        contract; we deliberately call the trunk pieces directly to get all N
        head logits in ONE forward pass instead of two.
    """

    def __init__(self, model, k=4):
        self.model = model
        # Cap k by the model's actual head count so we never index past
        # self.mtp's last head.
        self.k = int(min(int(k), int(getattr(model, "n_predict", 1))))
        # Some checkpoints will be loaded eval-mode already; this is idempotent.
        self.model.eval()

    # --------------------------------------------------------------------------
    # Internal: run the trunk and the MTP head once. Returns all-heads logits
    # at every position. Same path the model's own .forward() takes internally,
    # but we don't discard the heads-1..N-1 slice.

    @torch.no_grad()
    def _forward_all_heads(self, tokens):
        m = self.model
        T = tokens.size(1)
        # Defensive: if context grew past the cached RoPE table, extend it.
        # This mirrors what Veritate800M.forward_with_breakdown does.
        if hasattr(m, "rope_cos") and T > m.rope_cos.size(0):
            if hasattr(m, "extend_rope"):
                m.extend_rope(T)
        x = m.embed(tokens)
        for blk in m.blocks:
            x = blk(x, m.rope_cos, m.rope_sin)
        x = m.n_out(x)
        all_logits = m.mtp(x)  # [B, T, N, vocab]
        return all_logits

    # --------------------------------------------------------------------------
    # Sampling. Greedy for determinism; temperature is a multiplier on logits
    # in the rare case the caller wants softmax-sampled output (used only in
    # the head0_only path; the verify path is intrinsically deterministic and
    # the accept_all path mirrors the verify path's first byte).

    @staticmethod
    def _pick_greedy(logits_row):
        return int(logits_row.argmax(dim=-1).item())

    # --------------------------------------------------------------------------
    # mode="head0_only": baseline 1-byte-per-step. Reference for byte-exactness.

    @torch.no_grad()
    def _decode_head0_only(self, ctx, max_new, temperature):
        m = self.model
        seq_max = m.seq
        device = next(m.parameters()).device
        ctx = list(ctx)
        n_fwd = 0
        for _ in range(max_new):
            window = ctx[-seq_max:]
            toks = torch.tensor([window], dtype=torch.long, device=device)
            logits = self._forward_all_heads(toks)
            n_fwd += 1
            row = logits[0, -1, 0, :]  # head 0 at last position
            if temperature != 1.0:
                row = row / max(1e-6, float(temperature))
            ctx.append(self._pick_greedy(row))
        return ctx, n_fwd, 0  # no acceptance counter in this mode

    # --------------------------------------------------------------------------
    # mode="accept_all": each forward pass produces K bytes (argmax over each
    # head's last-position logits). FAST but heads 1..K-1 are lower-quality.

    @torch.no_grad()
    def _decode_accept_all(self, ctx, max_new, temperature):
        m = self.model
        seq_max = m.seq
        device = next(m.parameters()).device
        ctx = list(ctx)
        K = self.k
        n_fwd = 0
        produced = 0
        while produced < max_new:
            window = ctx[-seq_max:]
            toks = torch.tensor([window], dtype=torch.long, device=device)
            logits = self._forward_all_heads(toks)
            n_fwd += 1
            last = logits[0, -1]  # [N, vocab]
            if temperature != 1.0:
                last = last / max(1e-6, float(temperature))
            # Greedy argmax per head. Take only the first K (== self.k) heads.
            new_bytes = last[:K].argmax(dim=-1).tolist()
            remaining = max_new - produced
            new_bytes = [int(b) for b in new_bytes[:remaining]]
            ctx.extend(new_bytes)
            produced += len(new_bytes)
        return ctx, n_fwd, 0

    # --------------------------------------------------------------------------
    # mode="verify": Medusa-style speculative decoding.
    #
    # Per outer step:
    #   1. Forward at context C. Draft K bytes d_0..d_{K-1} from the K heads at
    #      the last position. d_0 is by definition the byte head-0 would have
    #      produced (so it's "verified" for free).
    #   2. Run a SECOND forward at C ++ d_0..d_{K-2} (length K-1 appended). For
    #      i in 1..K-1, the head-0 prediction at position (T + i - 1) in pass 2
    #      is the head-0 byte-at-that-context — i.e. what straight head0_only
    #      decode would produce after observing d_0..d_{i-1}. Accept d_i if it
    #      matches that head-0 prediction.
    #   3. Take the longest prefix that matches, ALWAYS extend by one
    #      head-0-canonical byte at the mismatch (so we make >= 1 byte/step
    #      progress and the output is byte-exact equivalent to head0_only).
    #
    # Cost: 2 forward passes per step. Gain: 1..K bytes per step. Speedup =
    # (bytes_accepted / 2) per forward, vs 1 byte/forward for head0_only.
    # Break-even at 2 bytes accepted; above that is net gain.

    @torch.no_grad()
    def _decode_verify(self, ctx, max_new, temperature):
        m = self.model
        seq_max = m.seq
        device = next(m.parameters()).device
        ctx = list(ctx)
        K = self.k
        n_fwd = 0
        proposed = 0   # total drafts proposed (>= 1 each step, == K most steps)
        accepted_extra = 0  # bytes accepted BEYOND the 1-byte baseline
        produced = 0
        while produced < max_new:
            # ---- pass 1: draft K bytes from K heads at last position ---------
            window = ctx[-seq_max:]
            toks = torch.tensor([window], dtype=torch.long, device=device)
            logits_p1 = self._forward_all_heads(toks)
            n_fwd += 1
            last = logits_p1[0, -1]  # [N, vocab]
            if temperature != 1.0:
                last = last / max(1e-6, float(temperature))
            drafts = [int(x) for x in last[:K].argmax(dim=-1).tolist()]

            # ---- pass 2: verify drafts 1..K-1 via head-0 in the extended ctx -
            # We only need pass 2 if K > 1. The verify positions are
            # (len(window) - 1 + i) for i in 1..K-1 in the EXTENDED window.
            if K > 1:
                verify_window = (window + drafts[:K - 1])[-seq_max:]
                v_toks = torch.tensor([verify_window], dtype=torch.long, device=device)
                logits_p2 = self._forward_all_heads(v_toks)
                n_fwd += 1
                # head-0 predictions at positions corresponding to "byte AFTER
                # observing d_0..d_{i-1}" for i = 1..K-1. Those are the last
                # K-1 positions of pass-2's output.
                T_v = logits_p2.size(1)
                verify_preds = []
                for i in range(1, K):
                    pos = T_v - K + i
                    if pos < 0 or pos >= T_v:
                        break
                    row = logits_p2[0, pos, 0, :]
                    if temperature != 1.0:
                        row = row / max(1e-6, float(temperature))
                    verify_preds.append(int(row.argmax().item()))
            else:
                verify_preds = []

            # ---- accept longest matching prefix ------------------------------
            # d_0 is always accepted (head-0 of pass 1 == head-0 by trivial id).
            accepted = [drafts[0]]
            mismatch_at = None
            for i, vp in enumerate(verify_preds, start=1):
                if i < len(drafts) and drafts[i] == vp:
                    accepted.append(drafts[i])
                else:
                    mismatch_at = i
                    # On mismatch, take the head-0 byte instead. This is the
                    # byte that head0_only would have produced from the same
                    # context — so the output is byte-exact lossless.
                    accepted.append(vp)
                    break

            proposed += K
            # accepted has length (matched_prefix_len + 1). The "+1" is either
            # the trivial d_0 (no extra) or the corrective head-0 byte after a
            # mismatch (no extra). True extra-bytes-vs-head0_only is the count
            # of MATCHED drafts in positions >= 1.
            n_extra = sum(
                1 for i in range(1, min(len(accepted), len(verify_preds) + 1))
                if (mismatch_at is None or i < mismatch_at)
            )
            accepted_extra += n_extra

            # Truncate at max_new boundary.
            remaining = max_new - produced
            accepted = accepted[:remaining]
            ctx.extend(int(b) for b in accepted)
            produced += len(accepted)

        # Acceptance rate: of the (K-1) extra drafts proposed per step, what
        # fraction were accepted. Defined only when K > 1.
        n_steps = max(1, n_fwd // 2) if K > 1 else max(1, n_fwd)
        possible_extras = (K - 1) * n_steps
        accept_rate = (accepted_extra / possible_extras) if possible_extras > 0 else 0.0
        return ctx, n_fwd, (accepted_extra, accept_rate)

    # --------------------------------------------------------------------------
    # Public API.

    def decode(self, prompt, max_new, mode="verify", temperature=1.0):
        """Decode `max_new` bytes following `prompt` using one of the three modes.

        Args:
          prompt:     bytes
          max_new:    int, number of new bytes to emit
          mode:       "head0_only" | "accept_all" | "verify"
          temperature: float, divides logits before argmax. Default 1.0 (== greedy).

        Returns:
          (text, stats)
            text  : str, prompt + generated bytes decoded utf-8 (replace).
            stats : dict with keys:
              mode, bytes_generated, n_forward, wall_s, ms_per_byte,
              accepted_extra (verify only), accept_rate (verify only)
        """
        if not isinstance(prompt, (bytes, bytearray)):
            raise TypeError(f"prompt must be bytes, got {type(prompt).__name__}")
        if mode not in ("head0_only", "accept_all", "verify"):
            raise ValueError(f"unknown mode: {mode!r}")
        ctx_in = list(prompt)
        prompt_len = len(ctx_in)

        t0 = time.perf_counter()
        if mode == "head0_only":
            ctx_out, n_fwd, _ = self._decode_head0_only(ctx_in, max_new, temperature)
            extra_stats = {}
        elif mode == "accept_all":
            ctx_out, n_fwd, _ = self._decode_accept_all(ctx_in, max_new, temperature)
            extra_stats = {}
        else:  # verify
            ctx_out, n_fwd, (accepted_extra, accept_rate) = self._decode_verify(
                ctx_in, max_new, temperature
            )
            extra_stats = {
                "accepted_extra": int(accepted_extra),
                "accept_rate": float(accept_rate),
            }
        wall_s = time.perf_counter() - t0

        gen_bytes = bytes(ctx_out[prompt_len:prompt_len + max_new])
        bytes_generated = len(gen_bytes)
        ms_per_byte = (wall_s * 1000.0) / max(1, bytes_generated)
        text = (bytes(ctx_in) + gen_bytes).decode("utf-8", errors="replace")

        stats = {
            "mode": mode,
            "k": self.k,
            "bytes_generated": bytes_generated,
            "n_forward": n_fwd,
            "wall_s": wall_s,
            "ms_per_byte": ms_per_byte,
            "gen_bytes": gen_bytes,  # raw bytes for byte-exact comparisons
        }
        stats.update(extra_stats)
        return text, stats
