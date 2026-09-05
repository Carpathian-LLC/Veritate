# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The masked_grid lever's guard rails. Every rejection here is a run that would
#   otherwise train on half an image, or on a trunk that gathers at text byte
#   boundaries an image code stream does not have, and report a plausible loss while
#   doing it.
# - Also pins that the lever is strictly opt-in: the default objective builds exactly
#   the causal model every existing run builds, with an unchanged state dict.
# tests/training/test_image_objective.py
# ------------------------------------------------------------------------------------
# Imports:

from types import SimpleNamespace

import pytest
from training import veritate_trainer

from veritate_core.model import Veritate

# ------------------------------------------------------------------------------------
# Constants

SEQ = 1024

# ------------------------------------------------------------------------------------
# Functions


def _args(**over):
    base = {"objective": "masked_grid", "image_code_bytes": 576, "seq": SEQ, "trunk": "dense"}
    base.update(over)
    return SimpleNamespace(**base)


def test_the_default_objective_is_next_byte_and_needs_nothing():
    """An existing run must not acquire a new obligation because this lever appeared."""
    objective, codes = veritate_trainer.check_objective(
        SimpleNamespace(seq=SEQ, trunk="hybrid", objective="next_byte", image_code_bytes=0))
    assert objective == "next_byte"
    assert codes == 0


def test_masked_grid_accepts_a_whole_image_in_the_window():
    """The happy path returns the geometry the loader needs."""
    assert veritate_trainer.check_objective(_args()) == ("masked_grid", 576)


def test_masked_grid_without_the_record_geometry_is_refused():
    """Without image_code_bytes the loader cannot find the image inside a record."""
    with pytest.raises(ValueError, match="image_code_bytes"):
        veritate_trainer.check_objective(_args(image_code_bytes=0))


def test_an_image_larger_than_the_window_is_refused():
    """Training on a truncated image reports a loss that means nothing."""
    with pytest.raises(ValueError, match="exceeds seq"):
        veritate_trainer.check_objective(_args(image_code_bytes=SEQ + 1))


def test_masked_grid_on_a_patched_trunk_is_refused():
    """Patched trunks gather on text byte boundaries; code streams have none."""
    with pytest.raises(ValueError, match="trunk=dense"):
        veritate_trainer.check_objective(_args(trunk="hybrid"))


def test_an_unknown_objective_names_the_valid_set():
    with pytest.raises(ValueError, match="unknown objective"):
        veritate_trainer.check_objective(_args(objective="diffusion"))


def test_attention_is_causal_unless_the_lever_asks_otherwise():
    """The flag carries no weights, so every existing checkpoint still loads."""
    causal = Veritate(256, 64, 2, 128, 4, 32)
    bidirectional = Veritate(256, 64, 2, 128, 4, 32, causal=False)
    assert causal.blocks[0].attn.causal is True
    assert bidirectional.blocks[0].attn.causal is False
    assert set(causal.state_dict()) == set(bidirectional.state_dict())
