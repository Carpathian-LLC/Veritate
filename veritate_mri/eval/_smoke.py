# ------------------------------------------------------------------------------------
# veritate_mri/eval/_smoke.py
# ------------------------------------------------------------------------------------
# CPU-only sub-30-second smoke test for the dashboard-facing eval harness.
#
# What it checks:
#   1. Imports resolve from the new `veritate_mri.eval` path.
#   2. `score_sequence` returns a finite float in the expected range (between
#      -ln(256) ~ -5.55 nats/byte for a uniform model and 0 for a perfect one),
#      and that scoring the same data twice gives identical results (determinism).
#   3. MMLU runs end-to-end on the shipped sample data and returns a valid
#      accuracy in [0, 1]. For a random-init model, expected accuracy is ~25%
#      (4-way chance) — anything close to that confirms the framework is unbiased.
#   4. HellaSwag runs end-to-end on 2 hand-built items and likewise returns
#      accuracy in [0, 1].
#   5. The high-level `run_suites_on_model` wrapper produces the same report
#      shape the Flask endpoint hands back to the dashboard.
#
# Uses a tiny random-init VeritateRoPE85M-shape model (hidden=32, layers=2,
# ffn=64, heads=4) so the whole thing finishes in seconds on CPU.
#
# Run:
#   python -m veritate_mri.eval._smoke
# ------------------------------------------------------------------------------------

from __future__ import annotations

import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_tiny_model(seed: int = 0):
    """A random-init, VeritateRoPE85M-shape tiny model. CPU-only."""
    from experiments.v2.rope_85m.model_rope85m import VeritateRoPE85M
    torch.manual_seed(seed)
    return VeritateRoPE85M(
        vocab=256, hidden=32, layers=2, ffn=64, heads=4, seq=512,
    )


def smoke_score_sequence():
    from veritate_mri.eval.score import score_sequence
    model = build_tiny_model(seed=42)
    score = score_sequence(model, b"Q: 1+1?", b" 2")
    assert isinstance(score, float), f"score is not float: {type(score)}"
    assert math.isfinite(score), f"score is not finite: {score}"
    lower = -math.log(256) - 1.5
    upper = -math.log(256) + 1.5
    assert lower < score < upper, (
        f"score {score} outside reasonable random-init band [{lower}, {upper}]"
    )
    score2 = score_sequence(model, b"Q: 1+1?", b" 2")
    assert abs(score - score2) < 1e-6, f"non-deterministic: {score} vs {score2}"
    print(f"  [ok] score_sequence -> {score:.4f} nats/byte (uniform floor = {-math.log(256):.4f})")
    return score


def smoke_mmlu():
    from veritate_mri.eval.mmlu import run_mmlu
    model = build_tiny_model(seed=42)
    result = run_mmlu(model, limit=4, mode="text", verbose=False)
    n = result["n"]
    acc = result["accuracy"]
    assert n == 4, f"expected n=4, got {n}"
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"
    result_full = run_mmlu(model, mode="text", verbose=False)
    n_full = result_full["n"]
    acc_full = result_full["accuracy"]
    assert n_full >= 10, f"expected ~20 sample items, got {n_full}"
    assert "by_subject" in result_full
    print(f"  [ok] MMLU smoke ({n} items) -> acc {acc:.3f}; "
          f"full sample ({n_full} items) -> acc {acc_full:.3f} "
          f"(chance ~{1/4:.2f})")
    return acc_full


def smoke_hellaswag():
    from veritate_mri.eval.hellaswag import run_hellaswag
    model = build_tiny_model(seed=42)
    result = run_hellaswag(model, verbose=False)
    n = result["n"]
    acc = result["accuracy"]
    assert n == 2, f"expected n=2 sample items, got {n}"
    assert acc in (0.0, 0.5, 1.0), f"accuracy {acc} not in {{0,0.5,1}} for n=2"
    print(f"  [ok] HellaSwag smoke ({n} items) -> acc {acc:.3f} "
          f"(chance ~{1/4:.2f})")
    return acc


def smoke_run_suites_wrapper():
    """The dashboard endpoint calls this wrapper; make sure the report shape is
    what the front-end expects (a `suites` dict keyed by suite name)."""
    from veritate_mri.eval.run_eval import run_suites_on_model
    model = build_tiny_model(seed=42)
    report = run_suites_on_model(model, suites=["mmlu"], limit=4)
    assert "suites" in report, f"missing 'suites' key: {report.keys()}"
    assert "mmlu" in report["suites"], f"missing mmlu in suites: {report['suites'].keys()}"
    s = report["suites"]["mmlu"]
    for k in ("suite", "n", "accuracy", "by_subject", "elapsed_s"):
        assert k in s, f"missing key in mmlu report: {k}"
    print(f"  [ok] run_suites_on_model wrapper -> suites={list(report['suites'].keys())}")


def smoke_endpoint_import():
    """The Flask app imports the run_suites entry at request time. Make sure the
    same import path works from a cold interpreter (no app context needed)."""
    # Just exercise the import; the actual route is exercised in tests/.
    import importlib
    m = importlib.import_module("veritate_mri.eval.run_eval")
    assert hasattr(m, "run_suites_on_model"), "run_suites_on_model not exported"
    print(f"  [ok] endpoint import path resolves: {m.__name__}")


def main():
    t0 = time.perf_counter()
    print("=== veritate_mri.eval smoke (CPU) ===")
    s = smoke_score_sequence()
    m = smoke_mmlu()
    h = smoke_hellaswag()
    smoke_run_suites_wrapper()
    smoke_endpoint_import()
    dt = time.perf_counter() - t0
    print(f"=== smoke OK in {dt:.1f}s ===")
    print(f"summary: score={s:.4f}  mmlu_acc={m:.3f}  hellaswag_acc={h:.3f}")
    chance_band = (0.0, 0.6)  # generous; n=20 has wide CI on random accuracy
    if not (chance_band[0] <= m <= chance_band[1]):
        print(f"  [warn] MMLU acc {m:.3f} outside expected chance band {chance_band}; "
              "could be small-n noise but worth a look.")


if __name__ == "__main__":
    main()
