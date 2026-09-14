# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the eval harness end to end on a real (tiny, random-init) model and the shipped sample
#   sets: no stubs anywhere in the path. The rest of the eval tests stub the model to pin
#   dispatch and arithmetic; this one catches what those cannot, a break between the
#   scorer, the sample files and the model. A random-init model must score at the uniform
#   byte floor and pick at chance, which is also the null every eval number is read
#   against. Was veritate_mri/eval/_smoke.py, a script nobody ran (rule 22); it runs in
#   under a second here.
# tests/mri/test_eval_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import torch
from eval.hellaswag import run_hellaswag
from eval.mmlu import run_mmlu
from eval.score import score_sequence

from veritate_core.model import Veritate

# ------------------------------------------------------------------------------------
# Constants

TINY_SHAPE = {"vocab": 256, "hidden": 32, "layers": 2, "ffn": 64, "heads": 4, "seq": 512}
UNIFORM_NATS = -math.log(256)
RANDOM_INIT_BAND = 1.5

# ------------------------------------------------------------------------------------
# Functions


def _tiny_model(seed=42):
    torch.manual_seed(seed)
    return Veritate(**TINY_SHAPE)


def test_an_untrained_model_scores_at_the_uniform_byte_floor():
    """The null for every eval number: 256 equally likely bytes. A scorer that drifts off
    this on random weights is measuring its own bug."""
    score = score_sequence(_tiny_model(), b"Q: 1+1?", b" 2")
    assert math.isfinite(score)
    assert UNIFORM_NATS - RANDOM_INIT_BAND < score < UNIFORM_NATS + RANDOM_INIT_BAND


def test_scoring_the_same_sequence_twice_gives_the_same_number():
    """Grading is bare-greedy and must be reproducible; a sampled path here would make
    every eval a coin toss."""
    model = _tiny_model()
    first = score_sequence(model, b"Q: 1+1?", b" 2")
    assert abs(first - score_sequence(model, b"Q: 1+1?", b" 2")) < 1e-6


def test_mmlu_runs_over_the_shipped_samples_and_honours_the_limit():
    model = _tiny_model()
    assert run_mmlu(model, limit=4, mode="text", verbose=False)["n"] == 4
    full = run_mmlu(model, mode="text", verbose=False)
    assert full["n"] >= 10
    assert 0.0 <= full["accuracy"] <= 1.0


def test_hellaswag_runs_over_the_shipped_samples():
    result = run_hellaswag(_tiny_model(), verbose=False)
    assert result["n"] == 2
    assert result["accuracy"] in (0.0, 0.5, 1.0)
