# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Smoke tests for /mesh/* routes via Flask's test_client. Settings file is
#   redirected to a tmp_path so user data is never touched. No live network is
#   ever opened; test_connection is exercised only with an empty hub_address
#   which short-circuits before urlopen.
# tests/mri/test_mesh_routes.py
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
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with mesh settings file redirected to tmp_path."""
    random.seed(0)
    from runtime import settings as settings_mod
    target = tmp_path / "mri_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", str(target))
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    from veritate_mri.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_mesh_status_returns_expected_keys(client):
    """GET /mesh/status returns dict containing role + hub_address + has_token."""
    r = client.get("/mesh/status")
    body = r.get_json()
    assert r.status_code == 200 and set(["role", "hub_address", "has_token"]).issubset(body.keys())


def test_mesh_role_node_ok_restart_required(client):
    """POST /mesh/role node -> ok=True with restart_required=True."""
    r = client.post("/mesh/role", json={"role": "node"})
    body = r.get_json()
    assert body.get("ok") is True and body.get("restart_required") is True


def test_mesh_role_invalid_returns_400(client):
    """POST /mesh/role with an invalid role returns 400."""
    r = client.post("/mesh/role", json={"role": "captain"})
    assert r.status_code == 400


def test_mesh_token_regenerate_returns_token(client):
    """POST /mesh/token/regenerate returns ok=True with a non-empty token."""
    r = client.post("/mesh/token/regenerate")
    body = r.get_json()
    assert body.get("ok") is True and bool(body.get("token"))


def test_mesh_token_get_returns_has_token_key(client):
    """GET /mesh/token returns dict containing has_token."""
    r = client.get("/mesh/token")
    body = r.get_json()
    assert "has_token" in body


def test_mesh_test_connection_empty_address_is_not_ok(client):
    """POST /mesh/test_connection with empty hub_address returns ok=False."""
    r = client.post("/mesh/test_connection", json={"hub_address": "", "auth_token": ""})
    body = r.get_json()
    assert body.get("ok") is False
