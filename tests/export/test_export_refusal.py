# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - non-canonical trunk refusals in the .bin exporters (rule 40b): RoPE buffers,
#   a missing learned pos_emb, and MTP heads raise ValueError naming the variant
#   before anything is written, so no corrupt .bin lands on disk.
# - MTP alone is exportable through export_checkpoint since the v12 format
#   shipped; the refusal there covers MTP + MoE, and ternary still refuses MTP.
# tests/export/test_export_refusal.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import json
import os

import pytest
import torch
from readers import paths
from training import export

from veritate_core.model import Veritate
from veritate_core.model_rope import VeritateRoPE

# ------------------------------------------------------------------------------------
# Constants

VOCAB  = 256
HIDDEN = 16
LAYERS = 2
FFN    = 32
HEADS  = 2
SEQ    = 64
STEP   = 3
NAME   = "tiny_variant"

SHAPE = {"vocab": VOCAB, "hidden": HIDDEN, "layers": LAYERS,
         "ffn": FFN, "heads": HEADS, "seq": SEQ}
MOE_BLOCK  = {"n_experts": 2, "router_topk": 1}
ROPE_BUFFER_KEY = "rope_cos"
BIN_GLOB = "*.bin"

# ------------------------------------------------------------------------------------
# Functions

def _seed_model(tmp_path, monkeypatch, state, mega=None):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    ckpt_dir = tmp_path / NAME / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"model": state}, ckpt_dir / f"step_{STEP}.pt")
    cfg = {"shape": dict(SHAPE), "training_args": {"trunk": "dense"}}
    if mega is not None:
        cfg["mega"] = mega
    with open(tmp_path / NAME / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return str(tmp_path / NAME)


def _rope_state(with_buffer):
    torch.manual_seed(0)
    state = VeritateRoPE(VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ).state_dict()
    if with_buffer:
        state[ROPE_BUFFER_KEY] = torch.zeros(SEQ, HIDDEN // HEADS)
    return state


def _mtp_state():
    torch.manual_seed(0)
    state = Veritate(VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ).state_dict()
    state["mtp.norms.0.weight"]      = torch.ones(HIDDEN)
    state["mtp.transforms.0.weight"] = torch.zeros(HIDDEN, HIDDEN)
    return state


def _bins(model_dir):
    return glob.glob(os.path.join(model_dir, BIN_GLOB))


def test_export_refuses_rope_buffered_checkpoint(tmp_path, monkeypatch):
    """export_checkpoint raises ValueError naming RoPE for a rope-buffered checkpoint."""
    _seed_model(tmp_path, monkeypatch, _rope_state(with_buffer=True))
    with pytest.raises(ValueError, match="not supported for RoPE models"):
        export.export_checkpoint(NAME, STEP)


def test_export_writes_no_bin_for_rope_checkpoint(tmp_path, monkeypatch):
    """The RoPE refusal leaves no .bin in the model dir."""
    model_dir = _seed_model(tmp_path, monkeypatch, _rope_state(with_buffer=True))
    with pytest.raises(ValueError):
        export.export_checkpoint(NAME, STEP)
    assert _bins(model_dir) == []


def test_export_refuses_checkpoint_without_pos_emb(tmp_path, monkeypatch):
    """export_checkpoint raises ValueError naming the missing pos_emb for a RoPE trunk."""
    _seed_model(tmp_path, monkeypatch, _rope_state(with_buffer=False))
    with pytest.raises(ValueError, match="missing pos_emb"):
        export.export_checkpoint(NAME, STEP)


def test_export_refuses_mtp_with_moe(tmp_path, monkeypatch):
    """export_checkpoint raises ValueError naming MTP and MoE for a mixed variant."""
    _seed_model(tmp_path, monkeypatch, _mtp_state(), mega=MOE_BLOCK)
    with pytest.raises(ValueError, match="both MTP and MoE"):
        export.export_checkpoint(NAME, STEP)


def test_export_writes_no_bin_for_mtp_moe(tmp_path, monkeypatch):
    """The MTP + MoE refusal leaves no .bin in the model dir."""
    model_dir = _seed_model(tmp_path, monkeypatch, _mtp_state(), mega=MOE_BLOCK)
    with pytest.raises(ValueError):
        export.export_checkpoint(NAME, STEP)
    assert _bins(model_dir) == []


def test_export_mtp_dense_takes_the_v12_path(tmp_path, monkeypatch):
    """A dense MTP checkpoint is not refused: it exports at the v12 format version."""
    _seed_model(tmp_path, monkeypatch, _mtp_state())
    assert export.export_checkpoint(NAME, STEP)["version"] == export.VERITATE_MODEL_VERSION_MTP


def test_ternary_refuses_rope_buffered_checkpoint(tmp_path, monkeypatch):
    """export_checkpoint_ternary raises ValueError naming RoPE for a rope-buffered checkpoint."""
    _seed_model(tmp_path, monkeypatch, _rope_state(with_buffer=True))
    with pytest.raises(ValueError, match="not supported for RoPE models"):
        export.export_checkpoint_ternary(NAME, STEP)


def test_ternary_refuses_mtp_checkpoint(tmp_path, monkeypatch):
    """export_checkpoint_ternary raises ValueError naming MTP for an MTP-head checkpoint."""
    _seed_model(tmp_path, monkeypatch, _mtp_state())
    with pytest.raises(ValueError, match="not supported for MTP models"):
        export.export_checkpoint_ternary(NAME, STEP)


def test_ternary_writes_no_bin_for_mtp_checkpoint(tmp_path, monkeypatch):
    """The ternary MTP refusal leaves no .bin in the model dir."""
    model_dir = _seed_model(tmp_path, monkeypatch, _mtp_state())
    with pytest.raises(ValueError):
        export.export_checkpoint_ternary(NAME, STEP)
    assert _bins(model_dir) == []
