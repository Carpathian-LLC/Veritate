# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the CLI over training/fuse.py, the step that pulls a consolidated checkpoint back
#   toward its pre-sleep weights (theta <- alpha*theta_ft + (1-alpha)*theta_prev) so sleep
#   damage is bounded after the fact. The module owns the arithmetic; what is pinned here
#   is that the CLI hands it the arguments in the right order and that alpha is mandatory,
#   since a defaulted alpha silently decides how much of a night's consolidation survives.
# tests/training/test_fuse_checkpoints_cli.py
# ------------------------------------------------------------------------------------
# Imports:

import sys

import pytest
from tools import fuse_checkpoints as fc

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def _fuse(model, base_step, tuned_step, alpha, out_step=None):
        calls.append((model, base_step, tuned_step, alpha, out_step))
        return 1300, {"fused": 200, "passed_through": 3}

    monkeypatch.setattr(fc.fusemod, "fuse", _fuse)
    return calls


def test_the_base_and_tuned_steps_keep_their_order(recorded, monkeypatch):
    """Swapping them fuses toward the wrong side and undoes the run instead of tempering
    it, with no error either way."""
    monkeypatch.setattr(sys, "argv", ["fuse", "wren1_3", "1200", "1250", "--alpha", "0.7"])
    assert fc.main() == 0
    assert recorded == [("wren1_3", 1200, 1250, 0.7, None)]


def test_the_out_step_is_passed_through(recorded, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["fuse", "m", "10", "20", "--alpha", "0.5", "--out-step", "99"])
    fc.main()
    assert recorded[0][4] == 99


def test_alpha_is_mandatory(monkeypatch):
    """A default would quietly decide how much of the consolidation survives."""
    monkeypatch.setattr(sys, "argv", ["fuse", "m", "10", "20"])
    with pytest.raises(SystemExit):
        fc.main()


def test_it_reports_what_was_fused(recorded, monkeypatch, capsys):
    """The counts are the only evidence the fuse touched the tensors it should have."""
    monkeypatch.setattr(sys, "argv", ["fuse", "m", "10", "20", "--alpha", "0.25"])
    fc.main()
    out = capsys.readouterr().out
    assert "step_1300.pt" in out and "200 fused" in out and "3 passed through" in out
