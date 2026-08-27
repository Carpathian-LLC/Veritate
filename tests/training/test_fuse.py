# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers training/fuse.py, the "merge fuse" step of the IDEA 20 T3 m1 spec. Fusion is
#   the lever that lets consolidation run at a usable learning rate: the run bounds its
#   own damage afterwards instead of avoiding it with a rate so low it needs 129 epochs.
# - the endpoints are the load-bearing cases. alpha=0 must reproduce the base EXACTLY
#   and alpha=1 the tuned model exactly; a fusion that drifts at its endpoints is
#   silently lossy everywhere in between.
# tests/training/test_fuse.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch
from training import fuse as fusemod

# ------------------------------------------------------------------------------------
# Functions


def _sd(v, dtype=torch.float32):
    return {"w": torch.full((2, 2), float(v), dtype=dtype),
            "b": torch.tensor([float(v)], dtype=dtype)}


def test_alpha_one_is_the_tuned_model_exactly():
    """An endpoint that drifts makes every interior alpha untrustworthy."""
    out, _ = fusemod.fuse_states(_sd(1), _sd(9), 1.0)
    assert torch.equal(out["w"], _sd(9)["w"]) and torch.equal(out["b"], _sd(9)["b"])


def test_alpha_zero_is_the_base_exactly():
    """The other endpoint."""
    out, _ = fusemod.fuse_states(_sd(1), _sd(9), 0.0)
    assert torch.equal(out["w"], _sd(1)["w"])


def test_interpolates_elementwise():
    """theta <- alpha*tuned + (1-alpha)*base."""
    out, stats = fusemod.fuse_states(_sd(0), _sd(10), 0.7)
    assert torch.allclose(out["w"], torch.full((2, 2), 7.0))
    assert stats["fused"] == 2 and stats["passed_through"] == 0


def test_dtype_is_preserved():
    """A bf16 checkpoint must stay bf16, or the resume path sees a different model."""
    out, _ = fusemod.fuse_states(_sd(0, torch.bfloat16), _sd(10, torch.bfloat16), 0.5)
    assert out["w"].dtype == torch.bfloat16


def test_non_float_and_mismatched_tensors_take_the_tuned_side():
    """Integer buffers and shape changes cannot be interpolated; passing them through
    keeps the fused checkpoint loadable rather than failing the whole merge."""
    base = {"n": torch.tensor([1, 2, 3]), "w": torch.zeros(2, 2)}
    tuned = {"n": torch.tensor([7, 8, 9]), "w": torch.ones(3, 3), "new": torch.ones(2)}
    out, stats = fusemod.fuse_states(base, tuned, 0.5)
    assert torch.equal(out["n"], tuned["n"])
    assert torch.equal(out["w"], tuned["w"])
    assert torch.equal(out["new"], tuned["new"])
    assert stats["fused"] == 0 and stats["passed_through"] == 3


def test_qat_base_prefix_is_stripped_on_both_sides():
    """A QAT-wrapped checkpoint nests the trunk under `base.`; resume strips it, so
    fusion has to compare the same key space or every tensor passes through unfused."""
    base = {"base.w": torch.zeros(2, 2)}
    tuned = {"base.w": torch.full((2, 2), 10.0)}
    out, stats = fusemod.fuse_states(base, tuned, 0.5)
    assert stats["fused"] == 1
    assert torch.allclose(out["w"], torch.full((2, 2), 5.0))


def test_alpha_outside_the_unit_interval_is_refused():
    """An alpha above 1 extrapolates past the tuned model; that is a different
    operation and must be asked for explicitly, not reached by a typo."""
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            fusemod.fuse_states(_sd(0), _sd(1), bad)
