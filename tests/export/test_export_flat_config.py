# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Older runs wrote a FLAT config.json: no "shape" block, no "training_args",
#   every field at top level. core_50m is one, and export refused it with
#   "missing shape field" because shape_from_config only ever looked in those two
#   places. Resolution order is now shape -> training_args -> top level.
# - The flat fixture below is core_50m's real config.json, trimmed.
# tests/export/test_export_flat_config.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

import pytest
from training import export as export_mod

# ------------------------------------------------------------------------------------
# Constants

NAME = "flatcfg"

FLAT_CFG = {
    "name": "core", "size": "50m", "corpus": "the_pile",
    "vocab": 256, "hidden": 640, "layers": 10, "ffn": 2560, "heads": 10, "seq": 2048,
    "precision": "bf16", "batch_size": 2, "total_steps": 11250585,
}

NESTED_CFG = {"shape": {"vocab": 256, "hidden": 640, "layers": 10,
                        "ffn": 2560, "heads": 10, "seq": 2048}}

TA_CFG = {"training_args": {"vocab": 256, "hidden": 640, "layers": 10,
                            "ffn": 2560, "heads": 10, "seq": 2048}}

# ------------------------------------------------------------------------------------
# Helpers


def _model(tmp_path, monkeypatch, cfg):
    d = tmp_path / NAME
    os.makedirs(d, exist_ok=True)
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    monkeypatch.setattr(export_mod.paths, "config_path",
                        lambda n, _d=str(d): os.path.join(_d, "config.json"))
    return str(d)


# ------------------------------------------------------------------------------------
# Resolution order


def test_flat_config_resolves_its_shape(tmp_path, monkeypatch):
    """The regression: core_50m's config has no shape and no training_args."""
    _model(tmp_path, monkeypatch, FLAT_CFG)
    assert export_mod.shape_from_config(NAME) == {
        "vocab": 256, "hidden": 640, "layers": 10,
        "ffn": 2560, "heads": 10, "seq": 2048,
    }


def test_nested_shape_still_wins(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch, {**NESTED_CFG, "hidden": 99})
    assert export_mod.shape_from_config(NAME)["hidden"] == 640


def test_training_args_beats_top_level(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch, {**TA_CFG, "hidden": 99})
    assert export_mod.shape_from_config(NAME)["hidden"] == 640


def test_a_genuinely_missing_field_still_raises(tmp_path, monkeypatch):
    cfg = {k: v for k, v in FLAT_CFG.items() if k != "heads"}
    _model(tmp_path, monkeypatch, cfg)
    with pytest.raises(ValueError, match="missing shape field: heads"):
        export_mod.shape_from_config(NAME)


def test_flat_config_still_validates_vocab(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch, {**FLAT_CFG, "vocab": 512})
    with pytest.raises(ValueError, match="vocab"):
        export_mod.shape_from_config(NAME)


def test_flat_config_still_validates_head_divisibility(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch, {**FLAT_CFG, "heads": 7})
    with pytest.raises(ValueError, match="not divisible"):
        export_mod.shape_from_config(NAME)
