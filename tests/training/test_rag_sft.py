# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the grounded (RAG) SFT job's pieces short of a training loop: the window loader
#   draws next-byte targets and refuses a corpus shorter than a window, the source
#   loader rebuilds the newest checkpoint or names the missing one, the checkpoint
#   writer stamps the provenance the next load reads, and the val loss averages only
#   finite readings.
# tests/training/test_rag_sft.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
import torch
from readers import paths
from training import rag_sft

from veritate_core.model import VOCAB_BYTE_LEVEL, Veritate

# ------------------------------------------------------------------------------------
# Constants

SHAPE = {"hidden": 16, "layers": 1, "ffn": 32, "heads": 2, "seq": 8}

# ------------------------------------------------------------------------------------
# Functions


def test_the_loader_draws_shifted_targets_and_refuses_a_short_corpus(tmp_path):
    corpus = tmp_path / "c.bin"
    corpus.write_bytes(bytes(range(64)))
    draw, n = rag_sft.make_loader(str(corpus), 8, 4, seed=1)
    assert n == 64
    toks, tgts = draw()
    assert toks.shape == tgts.shape == (4, 8) and toks.dtype == torch.int64
    assert torch.equal(tgts[:, :-1], toks[:, 1:]) and (tgts[:, -1] == toks[:, -1] + 1).all()
    (tmp_path / "tiny.bin").write_bytes(b"123456789")
    with pytest.raises(ValueError, match="corpus too small"):
        rag_sft.make_loader(str(tmp_path / "tiny.bin"), 8, 1, seed=1)


def test_the_source_is_the_newest_checkpoint_or_a_named_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="src"):
        rag_sft.load_source("src")
    src = Veritate(vocab=VOCAB_BYTE_LEVEL, **SHAPE)
    os.makedirs(paths.checkpoints_dir("src"))
    for step in (5, 40):
        torch.save({"model": src.state_dict(), "args": {**SHAPE, "heads": 2, "name": "src"}},
                   paths.checkpoint_path("src", step))
    model, args = rag_sft.load_source("src")
    assert args["name"] == "src" and isinstance(model, Veritate)
    toks = torch.randint(0, VOCAB_BYTE_LEVEL, (1, 8))
    assert torch.equal(model.eval()(toks)[0], src.eval()(toks)[0])


def test_a_checkpoint_carries_its_grounded_provenance(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag_sft.save_mod, "save",
                        lambda model, name, step, optimizer, args: seen.update(name=name, step=step, args=args) or "p")
    out = rag_sft.save_checkpoint("model", {"name": "src", "seq": 8}, "opt", "child", 7, "grounded_v1", "desc")
    assert out == "p" and seen["name"] == "child" and seen["step"] == 7
    assert seen["args"] == {"name": "src", "seq": 8, "description": "desc", "corpus": "grounded_v1",
                            "grounded_sft_from": "src"}


def test_val_loss_averages_finite_readings_only():
    losses = iter([torch.tensor(2.0), torch.tensor(float("nan")), torch.tensor(4.0)])

    class M(torch.nn.Module):
        def forward(self, x, y):
            return None, next(losses)
    m = M()
    draw = lambda: (torch.zeros(1, 2, dtype=torch.int64), torch.zeros(1, 2, dtype=torch.int64))  # noqa: E731
    assert rag_sft.eval_val_loss(m, draw, "cpu", 3) == 3.0 and m.training
    assert rag_sft.eval_val_loss(m, draw, "cpu", 0) is None
