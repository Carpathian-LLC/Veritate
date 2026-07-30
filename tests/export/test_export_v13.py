# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - v13 hybrid exporter round-trip: header, extension fields, boundary table,
#   tensor payload sizes, and refusal paths. spec at
#   documentation.md (engine).
# tests/export/test_export_v13.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import struct

import numpy as np
import pytest
import torch
from readers import paths
from training import export

from veritate_core.model_patched import N_LOCAL_DEC, N_LOCAL_ENC, PATCH_STRIDE, VeritatePatched, _boundary_table

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
    slots = SEQ // PATCH_STRIDE
    per_attn = HIDDEN + 3 * HIDDEN * HIDDEN + HIDDEN * HIDDEN + HIDDEN \
        + FFN * HIDDEN + HIDDEN * FFN
    per_rec = per_attn + 3 * HIDDEN * 4 + HEADS * HIDDEN + HEADS \
        + (HIDDEN // HEADS) + HIDDEN * HIDDEN
    return (VOCAB * HIDDEN + SEQ * HIDDEN + slots * HIDDEN
            + (N_LOCAL_ENC + N_LOCAL_DEC) * per_attn + GLOBAL * per_rec + HIDDEN)


@pytest.fixture
def fp32_export(tmp_path, monkeypatch):
    """Export a tiny hybrid checkpoint as fp32 and hand back the parsed file."""
    model = _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="fp32")
    with open(res["path"], "rb") as f:
        head = struct.unpack(export.HEADER_FMT, f.read(HEADER_BYTES))
        ext  = struct.unpack(export.HYBRID_HEADER_EXT_FMT, f.read(EXT_BYTES))
        boundary = np.frombuffer(f.read(VOCAB), dtype=np.uint8)
        payload  = f.read()
    return {"res": res, "head": head, "ext": ext, "boundary": boundary,
            "payload": payload, "model": model}


def test_v13_reports_the_hybrid_version(fp32_export):
    """export_checkpoint reports the hybrid format version."""
    assert fp32_export["res"]["version"] == export.VERITATE_MODEL_VERSION_HYBRID


def test_v13_header_magic_and_version(fp32_export):
    """The written header opens with the Veritate magic and format version 13."""
    assert fp32_export["head"][:2] == (export.VERITATE_MODEL_MAGIC, 13)


def test_v13_header_carries_the_model_shape(fp32_export):
    """The header records the flattened trunk shape (vocab, hidden, layers, ffn, heads, seq)."""
    assert fp32_export["head"][2:] == \
        (VOCAB, HIDDEN, N_LOCAL_ENC + GLOBAL + N_LOCAL_DEC, FFN, HEADS, SEQ)


def test_v13_header_extension_describes_the_hybrid_stack(fp32_export):
    """The hybrid header extension records dtype, layer split, stride, slots, chunk, and state rule."""
    assert fp32_export["ext"] == \
        (0, N_LOCAL_ENC, GLOBAL, N_LOCAL_DEC, PATCH_STRIDE, SEQ // PATCH_STRIDE, 4, 0)


def test_v13_writes_the_boundary_table(fp32_export):
    """The exported boundary table matches the model's own byte-boundary table."""
    assert np.array_equal(fp32_export["boundary"], _boundary_table().numpy().astype(np.uint8))


def test_v13_fp32_payload_byte_count(fp32_export):
    """The fp32 tensor payload is exactly four bytes per exported float."""
    assert len(fp32_export["payload"]) == _expected_tensor_floats() * 4


def test_v13_token_embedding_survives_the_roundtrip(fp32_export):
    """The token embedding reads back from the payload bit-identical to the checkpoint."""
    tok = np.frombuffer(fp32_export["payload"][:VOCAB * HIDDEN * 4],
                        dtype="<f4").reshape(VOCAB, HIDDEN)
    assert np.array_equal(tok, fp32_export["model"].tok_emb.weight.detach().numpy())


def _dtype_flag(path):
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        return struct.unpack("<i", f.read(4))[0]


def test_v13_fp16_payload_is_half_size(tmp_path, monkeypatch):
    """dtype=fp16 halves the tensor payload byte count."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="fp16")
    assert res["bytes"] == HEADER_BYTES + EXT_BYTES + VOCAB + _expected_tensor_floats() * 2


def test_v13_fp16_flags_dtype_one(tmp_path, monkeypatch):
    """dtype=fp16 records dtype flag 1 in the hybrid header extension."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="fp16")
    assert _dtype_flag(res["path"]) == 1


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


def test_v13_int8_payload_byte_count(tmp_path, monkeypatch):
    """dtype=int8 writes quantized big tensors plus fp32 row scales and fp32 smalls."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="int8")
    assert res["bytes"] == HEADER_BYTES + EXT_BYTES + VOCAB + _expected_int8_bytes()


def test_v13_int8_flags_dtype_two(tmp_path, monkeypatch):
    """dtype=int8 records dtype flag 2 in the hybrid header extension."""
    _make_model_dir(tmp_path)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    res = export.export_checkpoint(NAME, STEP, dtype="int8")
    assert _dtype_flag(res["path"]) == 2


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
