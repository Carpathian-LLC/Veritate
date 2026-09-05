# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - a rolling prune deletes checkpoints under a live server, so a step cached at
#   boot (cfg DEFAULT_STEP) or held in a stale browser dropdown can name a file
#   that no longer exists. covers the reader's resolution and the pytorch load
#   funnel that consumes it; Brain is stubbed so no torch import or weights load.
# tests/mri/test_checkpoint_step_resolve.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys
import types

from readers import checkpoints, paths
from routes import _brain

# ------------------------------------------------------------------------------------
# Functions

def _model(tmp_path, monkeypatch, steps):
    """Model dir holding one empty .pt per step."""
    d = tmp_path / "models" / "m" / "checkpoints"
    d.mkdir(parents=True)
    for s in steps:
        (d / f"step_{s}.pt").write_bytes(b"")
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    return "m"


def test_present_step_is_kept(tmp_path, monkeypatch):
    """A step still on disk resolves to itself."""
    name = _model(tmp_path, monkeypatch, [0, 5000, 5250])
    assert checkpoints.resolve_step(name, 5000) == 5000


def test_pruned_step_resolves_to_newest(tmp_path, monkeypatch):
    """A pruned step resolves to the newest checkpoint on disk."""
    name = _model(tmp_path, monkeypatch, [0, 5000, 5250])
    assert checkpoints.resolve_step(name, 4750) == 5250


def test_step_zero_survives_resolution(tmp_path, monkeypatch):
    """Step 0 (the grown seed) is a real step, not a falsy miss."""
    name = _model(tmp_path, monkeypatch, [0, 5000])
    assert checkpoints.resolve_step(name, 0) == 0


def test_no_checkpoints_keeps_request(tmp_path, monkeypatch):
    """With nothing on disk the request survives so the error names what was asked."""
    name = _model(tmp_path, monkeypatch, [])
    assert checkpoints.resolve_step(name, 4750) == 4750


def test_load_falls_back_from_pruned_step(tmp_path, monkeypatch):
    """load_pytorch_brain loads the newest checkpoint when the asked-for one is gone."""
    name = _model(tmp_path, monkeypatch, [0, 5000, 5250])
    loaded = []

    class _Brain:
        def __init__(self, ck, threads=None, memory=None):
            loaded.append(ck)

    mod = types.ModuleType("inference.backends.pytorch")
    mod.Brain = _Brain
    mod.load_memory = lambda p: None
    monkeypatch.setitem(sys.modules, "inference.backends.pytorch", mod)

    _, out_name, out_step = _brain.load_pytorch_brain(name, 4750, 1)
    assert (out_name, out_step) == (name, 5250)
    assert os.path.basename(loaded[0]) == "step_5250.pt"
