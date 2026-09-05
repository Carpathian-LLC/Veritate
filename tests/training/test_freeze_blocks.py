# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers veritate_trainer.freeze_below / trainable_params, the freeze_blocks lever: a
#   resumed run trains only the top of the model. Freezing the blocks alone is not
#   enough, a trainable embedding below them makes backward walk every block for its
#   input gradient (cardinal 2026-09-02: blocks-only freeze of 10/20 saved 12%, with the
#   embeddings frozen the backward stops at block 10). The test pins that nothing below
#   the cut receives a gradient and that the optimizer sees only what trains.
# tests/training/test_freeze_blocks.py
# ------------------------------------------------------------------------------------
# Imports:
# ------------------------------------------------------------------------------------
import sys

import pytest
import torch
from training import veritate_trainer as vt

from veritate_core.model import Veritate

# ------------------------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------------------------


def _tiny():
    torch.manual_seed(0)
    return Veritate(vocab=256, hidden=16, layers=4, ffn=32, heads=2, seq=8)


def test_zero_freezes_nothing():
    """freeze_blocks 0 leaves every parameter, embeddings included, trainable."""
    m = _tiny()
    trainable, total = vt.freeze_below(m, 0)
    assert trainable == total
    assert all(p.requires_grad for p in m.parameters())


def test_the_cut_freezes_both_embeddings_and_the_blocks_below_it():
    """N freezes tok_emb, pos_emb and blocks[:N]; blocks[N:], the final norm and the head train."""
    m = _tiny()
    trainable, total = vt.freeze_below(m, 2)
    assert 0 < trainable < total
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert any(n.startswith("tok_emb.") for n in frozen)
    assert any(n.startswith("pos_emb.") for n in frozen)
    assert all(not p.requires_grad for p in m.blocks[0].parameters())
    assert all(not p.requires_grad for p in m.blocks[1].parameters())
    assert all(p.requires_grad for p in m.blocks[2].parameters())
    assert all(p.requires_grad for p in m.blocks[3].parameters())
    assert len(vt.trainable_params(m)) == sum(1 for p in m.parameters() if p.requires_grad)


def test_backward_never_reaches_below_the_cut():
    """After a loss.backward() no frozen parameter holds a grad and every trainable one does."""
    m = _tiny()
    vt.freeze_below(m, 2)
    toks = torch.randint(0, 256, (2, 8))
    _, loss = m(toks, toks)
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
        else:
            assert p.grad is None, name


def test_a_count_that_freezes_every_block_is_refused():
    """A freeze count is model-shape-specific; one that leaves no block trainable is a launch error,
    not a silent run that trains only the final norm."""
    m = _tiny()
    with pytest.raises(ValueError, match="no block trainable"):
        vt.freeze_below(m, 4)
    vt.freeze_below(m, 3)
    assert all(p.requires_grad for p in m.blocks[3].parameters())


def test_the_flag_parses(monkeypatch):
    """--freeze_blocks is a reserved int flag every launch may pass; it defaults to 0."""
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py", "--freeze_blocks", "12"])
    assert vt.parse_args({"description": "t", "defaults": {}}).freeze_blocks == 12
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py"])
    assert vt.parse_args({"description": "t", "defaults": {}}).freeze_blocks == 0
