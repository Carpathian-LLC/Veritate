# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - schema validation of the shipped veritate_mri/data/trainer_sizes.json. Guards
#   against a malformed hand-edit that only surfaces at runtime: the required shape
#   fields come from veritate_trainer.py::SHAPE_FIELDS, the single source of truth
#   for what a size must supply to build a model.
# tests/mri/test_trainer_sizes_schema.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

from readers import paths

TRAINER_DIR = os.path.join(paths.REPO_ROOT, "veritate_mri", "training")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Helpers

SIZES_PATH = os.path.join(paths.REPO_ROOT, "veritate_mri", "data", "trainer_sizes.json")


def _load_doc():
    with open(SIZES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------
# Top-level shape


def test_top_level_keys_present():
    """The shipped doc has shared_defaults, sizes, and default_size."""
    doc = _load_doc()
    assert "shared_defaults" in doc
    assert "sizes" in doc
    assert "default_size" in doc


def test_default_size_is_a_key_in_sizes():
    """default_size names an entry that actually exists in sizes."""
    doc = _load_doc()
    assert doc["default_size"] in doc["sizes"]


# ------------------------------------------------------------------------------------
# Per-size shape fields


def test_every_size_has_the_trainer_required_shape_fields():
    """Every size's shape dict has every field veritate_trainer.py's SHAPE_FIELDS needs."""
    doc = _load_doc()
    for name, entry in doc["sizes"].items():
        shape = entry.get("shape") or {}
        for field in vt.SHAPE_FIELDS:
            assert field in shape, f"{name} missing shape field {field}"


def test_every_shape_field_is_numeric_and_not_boolean():
    """layers/hidden/ffn/heads are numbers the trainer can build a model from, not bools."""
    doc = _load_doc()
    for name, entry in doc["sizes"].items():
        shape = entry["shape"]
        for field in vt.SHAPE_FIELDS:
            value = shape[field]
            assert isinstance(value, (int, float)), f"{name}.{field} is not numeric"
            assert not isinstance(value, bool), f"{name}.{field} is a bool, not numeric"
