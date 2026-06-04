# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for runtime/settings.py mesh_* validation. Each test
#   monkeypatches SETTINGS_PATH to a tmp_path file so the real user settings
#   file is never touched. Cache is cleared between tests via the fixture.
# tests/mri/test_settings_mesh.py
# ------------------------------------------------------------------------------------
# Imports

import os
import random
import sys

import pytest

# ------------------------------------------------------------------------------------
# Constants

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect SETTINGS_PATH into tmp_path and reset cache per test."""
    random.seed(0)
    target = tmp_path / "mri_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", str(target))
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    yield target
    monkeypatch.setattr(settings_mod, "_CACHE", None)


def test_mesh_role_rejects_unknown(isolated_settings):
    """mesh_role not in VALID_MESH_ROLES raises ValueError."""
    with pytest.raises(ValueError):
        settings_mod._validate({"mesh_role": "captain"})


def test_mesh_role_accepts_off(isolated_settings):
    """mesh_role 'off' validates."""
    out = settings_mod._validate({"mesh_role": "off"})
    assert out["mesh_role"] == "off"


def test_mesh_role_accepts_node(isolated_settings):
    """mesh_role 'node' validates."""
    out = settings_mod._validate({"mesh_role": "node"})
    assert out["mesh_role"] == "node"


def test_mesh_role_accepts_hub(isolated_settings):
    """mesh_role 'hub' validates."""
    out = settings_mod._validate({"mesh_role": "hub"})
    assert out["mesh_role"] == "hub"


def test_mesh_role_accepts_both(isolated_settings):
    """mesh_role 'both' validates."""
    out = settings_mod._validate({"mesh_role": "both"})
    assert out["mesh_role"] == "both"


def test_mesh_role_is_lowercased_and_stripped(isolated_settings):
    """mesh_role '  NODE  ' normalizes to 'node'."""
    out = settings_mod._validate({"mesh_role": "  NODE  "})
    assert out["mesh_role"] == "node"


def test_mesh_hub_address_none_becomes_empty(isolated_settings):
    """mesh_hub_address None coerces to empty string."""
    out = settings_mod._validate({"mesh_hub_address": None})
    assert out["mesh_hub_address"] == ""


def test_mesh_hub_address_trailing_slash_stripped(isolated_settings):
    """mesh_hub_address strips trailing slash."""
    out = settings_mod._validate({"mesh_hub_address": "http://hub.example.com/"})
    assert out["mesh_hub_address"] == "http://hub.example.com"


def test_mesh_hub_address_non_string_raises(isolated_settings):
    """mesh_hub_address int raises ValueError."""
    with pytest.raises(ValueError):
        settings_mod._validate({"mesh_hub_address": 42})


def test_mesh_auth_token_none_becomes_empty(isolated_settings):
    """mesh_auth_token None coerces to empty string."""
    out = settings_mod._validate({"mesh_auth_token": None})
    assert out["mesh_auth_token"] == ""


def test_mesh_auth_token_non_string_raises(isolated_settings):
    """mesh_auth_token list raises ValueError."""
    with pytest.raises(ValueError):
        settings_mod._validate({"mesh_auth_token": ["abc"]})
