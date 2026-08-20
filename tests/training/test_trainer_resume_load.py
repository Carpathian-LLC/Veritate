# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - load_resume_state uses strict=False because QAT seeds a new model from a plain
#   checkpoint and legitimately owns tensors the source lacks. The cost is that a
#   genuinely wrong architecture also loads in silence: resuming wren_base into a
#   trunk built four blocks too deep left 54,649,152 parameters at random init and
#   printed nothing.
# - these pin that a plain resume refuses on any missing tensor, that QAT-style
#   loads still tolerate them, and that a clean load stays quiet.
# tests/training/test_trainer_resume_load.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys

import pytest
import torch
from torch import nn

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINER = os.path.join(REPO, "veritate_mri", "training", "veritate_trainer.py")

# ------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def trainer():
    if os.path.join(REPO, "veritate_mri") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
    spec = importlib.util.spec_from_file_location("veritate_trainer_load_test", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Net(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(blocks))


@pytest.fixture
def ckpt(tmp_path, trainer, monkeypatch):
    """Write a checkpoint holding `blocks` linear layers, keyed like the trainer's."""
    def _make(blocks=4, name="faker", step=100, prefix=""):
        path = tmp_path / f"step_{step}.pt"
        sd = {prefix + k: v for k, v in _Net(blocks).state_dict().items()}
        torch.save({"model": sd, "step": step, "optimizer": {"state": {}}}, path)
        monkeypatch.setattr(trainer.paths, "checkpoint_path",
                            lambda n, s, _p=str(path): _p)
        return name
    return _make


# ------------------------------------------------------------------------------------
# Tests


def test_clean_load_is_silent(trainer, ckpt, capsys):
    name = ckpt(blocks=4)
    trainer.load_resume_state(_Net(4), name, 100, "cpu", require_complete=True)
    assert capsys.readouterr().out == ""


def test_returns_the_optimizer_state(trainer, ckpt):
    name = ckpt(blocks=4)
    assert trainer.load_resume_state(_Net(4), name, 100, "cpu") == {"state": {}}


def test_model_too_deep_refuses(trainer, ckpt):
    """The wren_base failure: 6-block model, 4-block checkpoint, 2 blocks random."""
    name = ckpt(blocks=4)
    with pytest.raises(SystemExit) as e:
        trainer.load_resume_state(_Net(6), name, 100, "cpu", require_complete=True)
    assert "random init" in str(e.value)


def test_refusal_counts_the_stranded_parameters(trainer, ckpt):
    """Two 8x8 linears left unloaded is 128 parameters; say the number."""
    name = ckpt(blocks=4)
    with pytest.raises(SystemExit) as e:
        trainer.load_resume_state(_Net(6), name, 100, "cpu", require_complete=True)
    assert "128" in str(e.value)


def test_qat_style_load_tolerates_missing(trainer, ckpt, capsys):
    """require_complete defaults off: seeding a richer model is a real use."""
    trainer.load_resume_state(_Net(6), ckpt(blocks=4), 100, "cpu")
    assert "not in the checkpoint" in capsys.readouterr().out


def test_extra_checkpoint_tensors_are_reported_not_fatal(trainer, ckpt, capsys):
    """Unexpected keys strand nothing, so they warn rather than stop the run."""
    name = ckpt(blocks=6)
    trainer.load_resume_state(_Net(4), name, 100, "cpu", require_complete=True)
    assert "no home" in capsys.readouterr().out


def test_base_prefix_is_stripped(trainer, ckpt, capsys):
    """Sidecar checkpoints store the trunk under base.*; that is a complete load."""
    name = ckpt(blocks=4, prefix="base.")
    trainer.load_resume_state(_Net(4), name, 100, "cpu", require_complete=True)
    assert capsys.readouterr().out == ""


def test_weights_actually_land(trainer, ckpt, tmp_path, monkeypatch):
    """Guard the whole point: a silent no-op load would pass every test above."""
    src = _Net(2)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.5)
    path = tmp_path / "step_1.pt"
    torch.save({"model": src.state_dict(), "step": 1}, path)
    monkeypatch.setattr(trainer.paths, "checkpoint_path", lambda n, s: str(path))
    dst = _Net(2)
    trainer.load_resume_state(dst, "faker", 1, "cpu", require_complete=True)
    assert all(torch.allclose(p, torch.full_like(p, 0.5)) for p in dst.parameters())
