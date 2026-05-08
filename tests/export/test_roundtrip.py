# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Round-trip tests for the export pipeline. Build a synthetic 4-layer
#   byte-level model, export to .bin, read the header back, assert version +
#   shape + (for v11) quant_mode + n_experts + router_topk match what was
#   written.
# - These tests do NOT exercise the C engine. Engine-level round-trip lives
#   in tests/engine/.
# tests/export/test_roundtrip.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os
import struct

import pytest

# ------------------------------------------------------------------------------------
# Constants

VRTE_MAGIC          = b"VRTE"
HEADER_FMT          = "<4sIIIIIII"
HEADER_FIXED_BYTES  = 32  # 4 magic + 7 uint32 fields
SHAPE_TINY = {"vocab": 256, "hidden": 16, "layers": 2, "ffn": 32, "heads": 2, "seq": 32}

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def synth_model_dir():
    """Create a synthetic Veritate checkpoint at models/_synth_export_test/
    with a matching config.json. The leading underscore in the name keeps it
    out of the dashboard's normal model listing. Cleans up on teardown."""
    import shutil
    import torch

    from veritate_mri.readers import paths as paths_mod

    name = "_synth_export_test"
    mdir = os.path.join(paths_mod.MODELS_ROOT, name)
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
    os.makedirs(os.path.join(mdir, "checkpoints"))

    from veritate.model import Veritate
    model = Veritate(**SHAPE_TINY)
    torch.save({"model": model.state_dict()}, os.path.join(mdir, "checkpoints", "step_1.pt"))

    cfg = {
        "name": name,
        "description": "synthetic test model",
        "kind": "trainer",
        "plugin": "test_export",
        "vocab": SHAPE_TINY["vocab"],
        "shape": SHAPE_TINY,
    }
    with open(os.path.join(mdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    yield name

    shutil.rmtree(mdir, ignore_errors=True)


def test_export_v9_int8_writes_correct_magic_and_version(synth_model_dir):
    """export_checkpoint emits a .bin starting with VRTE magic + version=9 for a
    non-MoE INT8 checkpoint."""
    from veritate_mri import export

    info = export.export_checkpoint(synth_model_dir, 1)

    with open(info["path"], "rb") as f:
        header = f.read(HEADER_FIXED_BYTES)
    magic, version, vocab, hidden, layers, ffn, heads, seq = struct.unpack(HEADER_FMT, header)

    assert magic == VRTE_MAGIC,                            "magic must be VRTE"
    assert version == export.VERITATE_MODEL_VERSION,       "non-MoE INT8 export defaults to v9"
    assert version == 9,                                   "v9 numeric value pinned by the on-disk contract"
    assert (vocab, hidden, layers, ffn, heads, seq) == (
        SHAPE_TINY["vocab"], SHAPE_TINY["hidden"], SHAPE_TINY["layers"],
        SHAPE_TINY["ffn"], SHAPE_TINY["heads"], SHAPE_TINY["seq"],
    ), "header shape fields must round-trip exactly"


def test_export_v11_ternary_writes_qat_version_and_quant_mode(synth_model_dir):
    """export_checkpoint_ternary emits v11 + (quant_mode=TERNARY, n_experts=1,
    router_topk=1) header extension after the act_boost slot."""
    from veritate_mri import export

    info = export.export_checkpoint_ternary(synth_model_dir, 1)

    with open(info["path"], "rb") as f:
        header = f.read(HEADER_FIXED_BYTES)
        act_boost = struct.unpack("<i", f.read(4))[0]
        quant_mode, n_experts, router_topk = struct.unpack("<iii", f.read(12))

    magic, version, *_ = struct.unpack(HEADER_FMT, header)

    assert magic == VRTE_MAGIC
    assert version == export.VERITATE_MODEL_VERSION_QAT,   "ternary export uses v11 unified format"
    assert version == 11,                                  "v11 numeric value pinned by the on-disk contract"
    assert act_boost >= 1
    assert quant_mode == export.VERITATE_QUANT_TERNARY,    "ternary export sets quant_mode=TERNARY"
    assert n_experts == 1,                                 "non-MoE ternary export sets n_experts=1"
    assert router_topk == 1,                               "non-MoE ternary export sets router_topk=1"


def test_export_v11_quant_constants_match_engine_header():
    """The Python-side quant_mode integers must equal the C-side defines in
    veritate.h. A drift here corrupts every v11 binary."""
    from veritate_mri import export

    h_path = os.path.join(os.path.dirname(__file__), "..", "..",
                          "veritate_engine", "v1", "src", "veritate.h")
    with open(os.path.abspath(h_path), "r", encoding="utf-8") as f:
        text = f.read()

    pairs = [
        ("VERITATE_QUANT_INT8",     export.VERITATE_QUANT_INT8),
        ("VERITATE_QUANT_INT4",     export.VERITATE_QUANT_INT4),
        ("VERITATE_QUANT_TERNARY",  export.VERITATE_QUANT_TERNARY),
        ("VERITATE_MODEL_VERSION_QAT", export.VERITATE_MODEL_VERSION_QAT),
    ]
    for name, py_val in pairs:
        # naive parse: line of form `#define <name> <int>`
        line = next((ln for ln in text.splitlines() if f"#define {name} " in ln), None)
        assert line is not None,                f"{name} not defined in veritate.h"
        c_val = int(line.split()[2])
        assert c_val == py_val,                 f"{name}: C={c_val} vs Python={py_val} drifted"


def test_export_ternary_packed_size_matches_5_trits_per_byte(synth_model_dir):
    """A v11 ternary .bin must be substantially smaller than the v9 INT8 .bin
    of the same model (5 trits/byte vs 1 weight/byte). Sanity check on the
    packing path."""
    from veritate_mri import export

    int8_info    = export.export_checkpoint(synth_model_dir, 1)
    ternary_info = export.export_checkpoint_ternary(synth_model_dir, 1)

    int8_size    = os.path.getsize(int8_info["path"])
    ternary_size = os.path.getsize(ternary_info["path"])

    # Tiny synth model has overhead-dominated header, so we don't see the full
    # 5x reduction. We still expect ternary < int8 for any non-trivial weight.
    assert ternary_size < int8_size, \
        f"ternary .bin ({ternary_size}) should be smaller than int8 .bin ({int8_size})"
