# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Best-of-N orchestrator. Drives N independent samples through a Brain-shaped
#   backend, scores each, returns the best.
# - The scorer is pluggable. Two ship-ready scorers:
#     SelfGradingScorer  — mean per-byte NLL under the SAME model (S34 floor).
#     ConstantScorer     — placeholder for unit testing (returns 0 / random).
#   A trained PRM (W06) will be a third scorer that consumes per-byte signals
#   (entropy, lens consistency, residual norm) without needing a re-forward.
# - For now this is SEQUENTIAL sampling. A future MS-class wins by running N
#   forwards in parallel via batched-decode, but that's an inference-engine
#   change. The scaffold returns the same answer either way.
# - Compose this with AgentLoop: wrap each agent turn in a Best-of-N for the
#   JSON emit. The schema validator picks among N JSON objects rather than 1.
# veritate_mri/agent/best_of_n.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class Sample:
    """One sampled continuation."""
    seed:       int
    bytes_out:  bytes
    score:      float = math.nan          # lower = better by convention
    diagnostics: dict = field(default_factory=dict)


@dataclass
class BestOfNResult:
    """The whole BoN run."""
    best:    Optional[Sample] = None
    samples: List[Sample] = field(default_factory=list)
    elapsed_s: float = 0.0
    K:       int = 0


def self_grading_scorer(backend, prompt: str, sample_bytes: bytes,
                        temperature: float = 1.0, top_k_sample: int = 256) -> float:
    """Mean per-byte NLL of `sample_bytes` given `prompt`, scored by `backend`.
    Lower = the model is more confident in this continuation. S34 showed this
    self-grading signal yields ~45% relative gain at K=16 on the 85M."""
    import torch
    import torch.nn.functional as F

    m = backend.model
    seq = m.seq
    prompt_bytes = prompt.encode("utf-8")
    if len(prompt_bytes) >= seq:
        prompt_bytes = prompt_bytes[-(seq - 1):]
    full_bytes = bytes(prompt_bytes) + sample_bytes
    if len(full_bytes) >= seq:
        full_bytes = full_bytes[-(seq - 1):]
    if not sample_bytes:
        return math.inf  # empty samples can't be scored

    ids = torch.tensor([list(full_bytes)], dtype=torch.long)
    with torch.no_grad():
        logits, _ = m(ids)
    # logits[0, T-1] predicts byte at position T (i.e., the NEXT byte beyond what we fed).
    # We want NLL of sample_bytes[i] = byte at position (len(prompt_bytes) + i).
    # That is predicted by logits[0, len(prompt_bytes) + i - 1].
    p_start = len(prompt_bytes)
    nlls = []
    for i in range(len(sample_bytes)):
        pos = p_start + i - 1
        if pos < 0 or pos >= ids.size(1):
            continue
        full_p = F.softmax(logits[0, pos], dim=-1)
        b = sample_bytes[i]
        nll = -float(torch.log(full_p[b].clamp(min=1e-12)))
        nlls.append(nll)
    if not nlls:
        return math.inf
    return sum(nlls) / len(nlls)


def constant_scorer(*args, **kwargs) -> float:
    return 0.0


def run_best_of_n(backend, prompt: str, K: int = 16,
                  max_new: int = 128,
                  temperature: float = 0.8,
                  top_k_sample: int = 40,
                  seed_base: int = 0,
                  scorer: Optional[Callable[[Any, str, bytes], float]] = None,
                  constraint_factory: Optional[Callable[[], Any]] = None,
                  ) -> BestOfNResult:
    """Sample K independent continuations, score each, return best.
    `scorer(backend, prompt, sample_bytes) -> float` lower = better.
    `constraint_factory()` returns a fresh constraint per sample if grammar-
    constrained sampling is wanted (e.g., JSONConstraint for tool calls)."""
    import torch

    if scorer is None:
        scorer = self_grading_scorer

    res = BestOfNResult(K=K)
    t0 = time.time()

    for k in range(K):
        torch.manual_seed(seed_base + k)
        constraint = constraint_factory() if constraint_factory else None
        gen = backend.stream(prompt,
                             temperature=temperature,
                             top_k_sample=top_k_sample,
                             max_new=max_new,
                             constraint=constraint)
        out = bytearray()
        for ev in gen:
            kind = ev.get("kind")
            if kind == "token":
                b = ev.get("byte")
                if isinstance(b, int):
                    out.append(b & 0xff)
            elif kind == "fast_byte":
                b = ev.get("byte")
                if isinstance(b, int):
                    out.append(b & 0xff)
            elif kind in ("stop", "error"):
                break
        s = Sample(seed=seed_base + k, bytes_out=bytes(out))
        try:
            s.score = scorer(backend, prompt, s.bytes_out)
        except Exception as e:
            s.score = math.inf
            s.diagnostics["score_error"] = f"{type(e).__name__}: {e}"
        res.samples.append(s)

    res.elapsed_s = time.time() - t0
    if res.samples:
        finite = [s for s in res.samples if math.isfinite(s.score)]
        res.best = min(finite, key=lambda s: s.score) if finite else res.samples[0]
    return res
