# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - CPU-only sub-30-second smoke for the eval harness on a tiny random-init model.
#   Checks score_sequence range + determinism, MMLU and HellaSwag end-to-end on the
#   shipped sample sets. Run: python -m veritate_mri.eval._smoke
# veritate_mri/eval/_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

from __future__ import annotations

import math
import os
import sys
import time

import torch

MRI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MRI_ROOT not in sys.path:
    sys.path.insert(0, MRI_ROOT)

from eval.hellaswag import run_hellaswag  # noqa: E402
from eval.mmlu import run_mmlu  # noqa: E402
from eval.score import score_sequence  # noqa: E402
from veritate_core.model import Veritate  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

TINY_SHAPE = {"vocab": 256, "hidden": 32, "layers": 2, "ffn": 64, "heads": 4, "seq": 512}
UNIFORM_NATS = -math.log(256)
RANDOM_INIT_BAND = 1.5

# ------------------------------------------------------------------------------------
# Functions


def build_tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    return Veritate(**TINY_SHAPE)


def smoke_score_sequence():
    model = build_tiny_model(seed=42)
    score = score_sequence(model, b"Q: 1+1?", b" 2")
    assert isinstance(score, float) and math.isfinite(score), f"bad score: {score}"
    lower, upper = UNIFORM_NATS - RANDOM_INIT_BAND, UNIFORM_NATS + RANDOM_INIT_BAND
    assert lower < score < upper, f"score {score} outside random-init band [{lower}, {upper}]"
    score2 = score_sequence(model, b"Q: 1+1?", b" 2")
    assert abs(score - score2) < 1e-6, f"non-deterministic: {score} vs {score2}"
    print(f"  [ok] score_sequence -> {score:.4f} nats/byte (uniform floor {UNIFORM_NATS:.4f})")
    return score


def smoke_mmlu():
    model = build_tiny_model(seed=42)
    result = run_mmlu(model, limit=4, mode="text", verbose=False)
    assert result["n"] == 4, f"expected n=4, got {result['n']}"
    assert 0.0 <= result["accuracy"] <= 1.0, f"accuracy out of range: {result['accuracy']}"
    result_full = run_mmlu(model, mode="text", verbose=False)
    assert result_full["n"] >= 10, f"expected ~20 sample items, got {result_full['n']}"
    print(f"  [ok] MMLU smoke (4 items) -> acc {result['accuracy']:.3f}; "
          f"full sample ({result_full['n']} items) -> acc {result_full['accuracy']:.3f} (chance ~0.25)")
    return result_full["accuracy"]


def smoke_hellaswag():
    model = build_tiny_model(seed=42)
    result = run_hellaswag(model, verbose=False)
    assert result["n"] == 2, f"expected n=2 sample items, got {result['n']}"
    assert result["accuracy"] in (0.0, 0.5, 1.0), f"accuracy {result['accuracy']} invalid for n=2"
    print(f"  [ok] HellaSwag smoke (2 items) -> acc {result['accuracy']:.3f} (chance ~0.25)")
    return result["accuracy"]


def main():
    t0 = time.perf_counter()
    print("=== eval smoke (CPU) ===")
    s = smoke_score_sequence()
    m = smoke_mmlu()
    h = smoke_hellaswag()
    print(f"=== smoke OK in {time.perf_counter() - t0:.1f}s ===")
    print(f"summary: score={s:.4f}  mmlu_acc={m:.3f}  hellaswag_acc={h:.3f}")


if __name__ == "__main__":
    main()
