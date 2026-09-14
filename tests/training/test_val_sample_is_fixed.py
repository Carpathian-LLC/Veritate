# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - a validation loader's generator advances with every draw, so before this fix each
#   evaluation in a run scored DIFFERENT windows and the run's val rows were not comparable
#   to each other. Measured on exp_wm_0905 (2026-09-09, failures.md): the same weights read
#   0.427959 on the first evaluation's windows and 0.382096 on the fifth's, 12% apart, which
#   the run reported as a 10.7% improvement over 200 steps whose true effect was +0.04%.
#   evaluate() now re-seeds the loader first. The training loader must NOT be reset, or a
#   run would train on the same batch forever.
# tests/training/test_val_sample_is_fixed.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
import torch
from torch import nn
from training import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Constants

WINDOW, BATCH, SEED = 16, 2, 7

# ------------------------------------------------------------------------------------
# Functions


class _Model(nn.Module):
    """evaluate() reads .parameters() for the device; one is enough."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))


def _bin(tmp_path):
    p = tmp_path / "val.bin"
    p.write_bytes(bytes(range(256)) * 40)
    return str(p)


def test_a_loader_advances_between_draws(tmp_path):
    """The training loader's whole job: every step sees new data."""
    draw, _n = vt.make_data_loader(_bin(tmp_path), WINDOW, BATCH, SEED)
    assert not torch.equal(draw()[0], draw()[0])


def test_reset_returns_the_loader_to_its_first_windows(tmp_path):
    draw, _n = vt.make_data_loader(_bin(tmp_path), WINDOW, BATCH, SEED)
    first = draw()[0]
    draw(), draw()
    draw.reset()
    assert torch.equal(draw()[0], first)


def test_two_evaluations_of_the_same_weights_agree(tmp_path):
    """The defect, in one assertion: without the reset the second evaluation scored other
    windows and reported a different loss for weights that had not changed."""
    draw, _n = vt.make_data_loader(_bin(tmp_path), WINDOW, BATCH, SEED)
    seen = []

    def _chunked(model, toks, tgts, seq, amp, **kw):
        seen.append(toks.clone())
        return float(toks.float().mean())

    original = vt.chunked_step
    vt.chunked_step = _chunked
    try:
        first = vt.evaluate(_Model(), draw, 2, WINDOW, None, 1, device_type="cpu")
        second = vt.evaluate(_Model(), draw, 2, WINDOW, None, 1, device_type="cpu")
    finally:
        vt.chunked_step = original
    assert first == second
    assert torch.equal(seen[0], seen[2]) and torch.equal(seen[1], seen[3])


def test_a_loader_without_a_reset_still_evaluates(tmp_path):
    """The image record loader is a plain closure; evaluate must not require the hook."""
    arr = np.frombuffer(bytes(range(256)) * 4, dtype=np.uint8).astype(np.int64)

    def plain():
        t = torch.from_numpy(arr[: BATCH * WINDOW].reshape(BATCH, WINDOW))
        return t, t

    original = vt.chunked_step
    vt.chunked_step = lambda *a, **k: 1.0
    try:
        assert vt.evaluate(_Model(), plain, 2, WINDOW, None, 1, device_type="cpu") == 1.0
    finally:
        vt.chunked_step = original
