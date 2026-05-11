# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Wraps any Veritate-family model and decodes under a Constraint. The model
#   may be canonical Veritate, RoPE85M, the 800M plugin, or any nn.Module that
#   returns a tensor of shape [B, T, 256] (or a tuple whose first element is).
# - Decode loop applies `constraint.mask()` as a -inf logit mask, picks greedy
#   or sampled, steps the constraint, and halts when `constraint.done()` or
#   `len(out) >= max_new`. If every byte is masked we raise.
# veritate_mri/decode/constrained.py
# ------------------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .constraints import Constraint, JSONConstraint


@dataclass
class ConstrainedStats:
    bytes_generated: int = 0
    forward_calls:   int = 0
    wall_time_s:     float = 0.0
    halted_by_done:  bool = False
    halted_by_max:   bool = False
    halt_reason:     str = ""
    per_step_allowed_count: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "bytes_generated": self.bytes_generated,
            "forward_calls":   self.forward_calls,
            "wall_time_s":     self.wall_time_s,
            "halted_by_done":  self.halted_by_done,
            "halted_by_max":   self.halted_by_max,
            "halt_reason":     self.halt_reason,
            "avg_allowed":     (float(np.mean(self.per_step_allowed_count))
                                if self.per_step_allowed_count else 0.0),
        }


def _model_forward_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    out = model(tokens)
    if isinstance(out, tuple):
        out = out[0]
    return out


class ConstrainedDecoder:
    """Decode bytes from a model subject to a Constraint.

    Usage:
        dec = ConstrainedDecoder(model)
        text, stats = dec.generate(
            prompt   = b'{"name": "',
            constraint = JSONConstraint(),
            max_new   = 256,
            device    = "cpu",
            prime_constraint_with_prompt = True,
        )
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes,
        constraint: Constraint,
        max_new: int = 128,
        device: str = "cpu",
        sample: bool = False,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        prime_constraint_with_prompt: bool = True,
    ):
        if max_new < 1:
            raise ValueError(f"max_new must be >= 1, got {max_new}")

        self.model.eval()
        dev = torch.device(device)

        # Reset the constraint so the same object can be re-used across runs.
        constraint.reset()
        if prime_constraint_with_prompt and isinstance(constraint, JSONConstraint):
            constraint.prime(prompt)
        elif prime_constraint_with_prompt:
            # Generic priming -- only safe if we trust the prompt complies.
            for b in prompt:
                constraint.step(int(b) & 0xff)

        if seed is not None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(seed))
        else:
            gen = None

        ctx = torch.tensor(list(prompt), dtype=torch.long, device=dev).unsqueeze(0)
        produced = bytearray()
        stats = ConstrainedStats()

        max_seq = getattr(self.model, "seq", 4096)

        t0 = time.perf_counter()
        while len(produced) < max_new:
            if constraint.done():
                stats.halted_by_done = True
                stats.halt_reason = "constraint.done() before forward"
                break

            inp = ctx if ctx.size(1) <= max_seq else ctx[:, -max_seq:]
            logits = _model_forward_logits(self.model, inp)
            stats.forward_calls += 1
            last_logits = logits[0, -1, :].detach().to(torch.float32).cpu().numpy()  # [256]

            allowed = constraint.mask()
            if not allowed.any():
                stats.halt_reason = "constraint allowed no bytes"
                break
            stats.per_step_allowed_count.append(int(allowed.sum()))

            masked = np.where(allowed, last_logits, -np.inf)

            if sample:
                # Softmax with temperature, then sample.
                t = max(float(temperature), 1e-6)
                z = masked / t
                z = z - np.max(z)
                p = np.exp(z)
                s = p.sum()
                if not np.isfinite(s) or s <= 0.0:
                    # Fallback: uniform over allowed.
                    p = allowed.astype(np.float64)
                    s = p.sum()
                p = p / s
                # Use torch for seeded sampling so we can be reproducible.
                p_t = torch.from_numpy(p.astype(np.float32))
                if gen is not None:
                    nxt = int(torch.multinomial(p_t, 1, generator=gen).item())
                else:
                    nxt = int(torch.multinomial(p_t, 1).item())
            else:
                nxt = int(np.argmax(masked))

            # Sanity: chosen byte must be allowed.
            if not allowed[nxt]:
                raise RuntimeError(
                    f"picked byte {nxt} but constraint disallowed it -- bug"
                )

            produced.append(nxt)
            constraint.step(nxt)
            ctx = torch.cat(
                [ctx, torch.tensor([[nxt]], dtype=torch.long, device=dev)],
                dim=1,
            )

            if constraint.done():
                stats.halted_by_done = True
                stats.halt_reason = "constraint.done() after step"
                break

        if not stats.halted_by_done and len(produced) >= max_new:
            stats.halted_by_max = True
            stats.halt_reason = stats.halt_reason or "max_new reached"

        stats.bytes_generated = len(produced)
        stats.wall_time_s = time.perf_counter() - t0

        return bytes(produced).decode("latin-1"), stats
