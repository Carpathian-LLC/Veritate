# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - v13 hybrid exporter round-trip: header, extension fields, boundary table,
#   tensor payload sizes, and refusal paths. spec at
#   developer_documentation/engine/engine_v13_hybrid.md.
# tests/export/test_export_v13.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import struct
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if p not in sys.path:
        sys.path.insert(0, p)

from readers import paths
from training import export
from veritate_core.model_patched import (N_LOCAL_DEC, N_LOCAL_ENC, PATCH_STRIDE,
                                         VeritatePatched, _boundary_table)

# ------------------------------------------------------------------------------------
# Constants

VOCAB   = 256
HIDDEN  = 32
GLOBAL  = 3
FFN     = 64
HEADS   = 4
SEQ     = 64
NAME    = "tiny_hybrid"
STEP    = 10
HEADER_BYTES = struct.calcsize(export.HEADER_FMT)
EXT_BYTES    = struct.calcsize(export.HYBRID_HEADER_EXT_FMT)

# ------------------------------------------------------------------------------------
# Functions


def _make_model_dir(tmp_path, trunk="hybrid", state_rule="gla"):
    torch.manual_seed(0)
    model = VeritatePatched(VOCAB, HIDDEN, GLOBAL, FFN, HEADS, SEQ,
                            global_mixer="recurrent", state_rule="gla")
    mdir = tmp_path / NAME / "checkpoints"
    mdir.mkdir(parents=True)
    torch.save({"model": model.state_dict()}, mdir / f"step_{STEP}.pt")
    cfg = {
        "shape": {"vocab": VOCAB, "hidden": HIDDEN, "layers": GLOBAL,
                  "ffn": FFN, "heads": HEADS, "seq": SEQ},
        "training_args": {"trunk": trunk, "state_rule": state_rule},
    }
    with open(tmp_path / NAME / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return model


def _expected_tensor_floats():
    n_total = N_LOCAL_ENC + GLOBAL + N_LOCAL_DEC
    slots = SEQ // PATCH_STRIDE
    per_attn = HIDDEN + 3 * HIDDEN * HIDDEN + HIDDEN * HIDDEN + HIDDEN \
        + FFN * HIDDEN + HIDDEN * FFN
    per_rec = per_attn + 3 * HIDDEN * 4 + HEADS * HIDDEN + HEADS \
        + (HIDDEN // HEADS) + HIDDEN * HIDDEN
    return (VOCAB * HIDDEN + SEQ * HIDDEN + slots * HIDDEN
            + (N_LOCAL_ENC + N_LOCAL_DEC) * per_attn + GLOBAL * per_rec + HIDDEN)


def test_v13_roundtrip_fp32(tmp_path, monkeypatch):
    """v13 fp32 export writes a parseable header + exact payload byte count."""
    model = _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="fp32")
    assert res["version"] == export.VERITATE_MODEL_VERSION_HYBRID
    with open(res["path"], "rb") as f:
        magic, ver, vocab, hidden, layers, ffn, heads, seq = struct.unpack(
            export.HEADER_FMT, f.read(HEADER_BYTES))
        dtype, n_enc, n_glob, n_dec, stride, slots, ck, rule = struct.unpack(
            export.HYBRID_HEADER_EXT_FMT, f.read(EXT_BYTES))
        boundary = np.frombuffer(f.read(VOCAB), dtype=np.uint8)
        payload = f.read()
    assert (magic, ver) == (export.VERITATE_MODEL_MAGIC, 13)
    assert (vocab, hidden, layers, ffn, heads, seq) == \
        (VOCAB, HIDDEN, N_LOCAL_ENC + GLOBAL + N_LOCAL_DEC, FFN, HEADS, SEQ)
    assert (dtype, n_enc, n_glob, n_dec, stride, slots, ck, rule) == \
        (0, N_LOCAL_ENC, GLOBAL, N_LOCAL_DEC, PATCH_STRIDE, SEQ // PATCH_STRIDE, 4, 0)
    assert np.array_equal(boundary, _boundary_table().numpy().astype(np.uint8))
    assert len(payload) == _expected_tensor_floats() * 4
    tok = np.frombuffer(payload[:VOCAB * HIDDEN * 4], dtype="<f4").reshape(VOCAB, HIDDEN)
    ref = model.tok_emb.weight.detach().numpy()
    assert np.array_equal(tok, ref)


def test_v13_fp16_payload_half_size(tmp_path, monkeypatch):
    """dtype=fp16 halves the tensor payload and flags dtype=1 in the header."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="fp16")
    expected = HEADER_BYTES + EXT_BYTES + VOCAB + _expected_tensor_floats() * 2
    assert res["bytes"] == expected
    with open(res["path"], "rb") as f:
        f.seek(HEADER_BYTES)
        (dtype,) = struct.unpack("<i", f.read(4))
    assert dtype == 1


def _expected_int8_bytes():
    n_total = N_LOCAL_ENC + GLOBAL + N_LOCAL_DEC
    slots = SEQ // PATCH_STRIDE
    smalls = (VOCAB * HIDDEN + SEQ * HIDDEN + slots * HIDDEN + HIDDEN
              + n_total * 2 * HIDDEN
              + GLOBAL * (3 * HIDDEN * 4 + HEADS * HIDDEN + HEADS + HIDDEN // HEADS))
    big_attn_rows = 3 * HIDDEN + HIDDEN + FFN + HIDDEN
    big_rec_rows  = big_attn_rows + HIDDEN
    big_rows  = (N_LOCAL_ENC + N_LOCAL_DEC) * big_attn_rows + GLOBAL * big_rec_rows
    big_elems = ((N_LOCAL_ENC + N_LOCAL_DEC)
                 * (3 * HIDDEN * HIDDEN + HIDDEN * HIDDEN + 2 * FFN * HIDDEN)
                 + GLOBAL * (3 * HIDDEN * HIDDEN + 2 * HIDDEN * HIDDEN + 2 * FFN * HIDDEN))
    return smalls * 4 + big_elems + big_rows * 4


def test_v13_int8_layout(tmp_path, monkeypatch):
    """dtype=int8 writes q + fp32 row scales for big tensors, fp32 smalls, dtype=2."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="int8")
    expected = HEADER_BYTES + EXT_BYTES + VOCAB + _expected_int8_bytes()
    assert res["bytes"] == expected
    with open(res["path"], "rb") as f:
        f.seek(HEADER_BYTES)
        (dtype,) = struct.unpack("<i", f.read(4))
    assert dtype == 2


def test_v13_refuses_non_gla(tmp_path, monkeypatch):
    """state_rule != gla is refused."""
    _make_model_dir(tmp_path, state_rule="delta")
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="state_rule"):
        export.export_checkpoint(NAME, STEP)


def test_patched_trunk_still_refused(tmp_path, monkeypatch):
    """trunk=patched keeps the no-engine-format refusal."""
    _make_model_dir(tmp_path, trunk="patched")
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="trunk 'patched'"):
        export.export_checkpoint(NAME, STEP)


def test_ternary_refuses_hybrid_trunk(tmp_path, monkeypatch):
    """export_checkpoint_ternary refuses a hybrid-trunk checkpoint."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="trunk 'hybrid'"):
        export.export_checkpoint_ternary(NAME, STEP)
