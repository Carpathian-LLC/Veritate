# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Structural validation of every shipped trainers/<id>/manifest.json. The
#   dashboard form, size dropdown, and VRAM estimator all read these blind;
#   a malformed manifest breaks the Training tab silently.
# tests/plugin_contract/test_manifests.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os

import pytest

from veritate_mri.readers.trainers import PLUGINS_ROOT

# ------------------------------------------------------------------------------------
# Constants

SHAPE_FIELDS      = ("layers", "hidden", "ffn", "heads", "params")
REQUIRED_DEFAULTS = ("size", "precision", "version")


# ------------------------------------------------------------------------------------
# Functions

def _manifest_items():
    out = []
    for d in sorted(os.listdir(PLUGINS_ROOT)):
        p = os.path.join(PLUGINS_ROOT, d, "manifest.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                out.append((d, json.load(f)))
    return out


MANIFESTS = _manifest_items()
SIZED     = [(d, m) for d, m in MANIFESTS if m.get("sizes")]


@pytest.mark.parametrize("plugin_id,manifest", MANIFESTS, ids=[d for d, _ in MANIFESTS])
def test_manifest_kind_is_trainer(plugin_id, manifest):
    """Every manifest declares kind == 'trainer'."""
    assert manifest.get("kind") == "trainer"


@pytest.mark.parametrize("plugin_id,manifest", SIZED, ids=[d for d, _ in SIZED])
def test_sized_manifest_declares_required_defaults(plugin_id, manifest):
    """Manifests with a sizes table declare size, precision, and version defaults."""
    missing = [k for k in REQUIRED_DEFAULTS if k not in manifest.get("defaults", {})]
    assert not missing, f"{plugin_id} defaults missing {missing}"


@pytest.mark.parametrize("plugin_id,manifest", SIZED, ids=[d for d, _ in SIZED])
def test_default_size_is_in_sizes_table(plugin_id, manifest):
    """defaults.size names a key of the manifest's sizes table."""
    assert manifest["defaults"]["size"] in manifest["sizes"]


@pytest.mark.parametrize("plugin_id,manifest", SIZED, ids=[d for d, _ in SIZED])
def test_sizes_table_is_single_entry(plugin_id, manifest):
    """Each trainer is standalone at exactly one size."""
    assert len(manifest["sizes"]) == 1


@pytest.mark.parametrize("plugin_id,manifest", SIZED, ids=[d for d, _ in SIZED])
def test_size_shape_fields_complete(plugin_id, manifest):
    """Every sizes entry carries positive-int layers, hidden, ffn, heads, params."""
    bad = [f"{label}.{field}"
           for label, shape in manifest["sizes"].items()
           for field in SHAPE_FIELDS
           if not (isinstance(shape.get(field), int) and shape.get(field) > 0)]
    assert not bad, f"{plugin_id} bad shape fields: {bad}"


@pytest.mark.parametrize("plugin_id,manifest", SIZED, ids=[d for d, _ in SIZED])
def test_hidden_divisible_by_heads(plugin_id, manifest):
    """hidden % heads == 0 for every size (model construction raises otherwise)."""
    bad = [label for label, shape in manifest["sizes"].items()
           if shape["hidden"] % shape["heads"] != 0]
    assert not bad, f"{plugin_id} sizes with hidden not divisible by heads: {bad}"


@pytest.mark.parametrize("plugin_id,manifest", MANIFESTS, ids=[d for d, _ in MANIFESTS])
def test_trainer_shim_exists(plugin_id, manifest):
    """Every manifest sits next to a trainer.py entry point."""
    assert os.path.isfile(os.path.join(PLUGINS_ROOT, plugin_id, "trainer.py"))
