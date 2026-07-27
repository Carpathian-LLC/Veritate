# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - bearer gate on the /mesh/* protocol routes. the gate itself lives in
#   veritate_mesh/hub.py; veritate_mri/routes/mesh_routes.py owns token rotation,
#   so invalidation is exercised across both: rotate, then replay the old token.
# - the settings store is redirected to tmp_path and the hub registry singleton is
#   replaced, so no real settings file and no cross-test state are touched.
# tests/mesh/test_mesh_auth.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import random

import pytest
from flask import Flask
from routes import mesh_routes
from runtime import settings as settings_mod

from veritate_mesh import hub
from veritate_mesh.registry import NodeRegistry

# ------------------------------------------------------------------------------------
# Constants

SEED = 1743

REAL_SETTINGS_PATH = settings_mod.SETTINGS_PATH
SETTINGS_FILENAME  = "mri_settings.json"
TOKEN_KEY          = "mesh_auth_token"

TOKEN       = "mesh-token-correct"
WRONG_TOKEN = "mesh-token-wrong"

GATED_PATH     = "/mesh/hub/nodes"
REGENERATE_PATH = "/mesh/token/regenerate"

HTTP_OK           = 200
HTTP_UNAUTHORIZED = 401

# ------------------------------------------------------------------------------------
# Functions

def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _read_bytes(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Mesh app whose token store is a fresh settings file under tmp_path."""
    random.seed(SEED)
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", str(tmp_path / SETTINGS_FILENAME))
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    monkeypatch.setattr(hub, "_REGISTRY", NodeRegistry())
    settings_mod.update({TOKEN_KEY: TOKEN})

    app = Flask(__name__)
    hub.register(app)
    mesh_routes.register(app)
    return app.test_client()


def test_wrong_token_rejected(client):
    """A gated mesh route with a wrong bearer token returns 401."""
    assert client.get(GATED_PATH, headers=_bearer(WRONG_TOKEN)).status_code == HTTP_UNAUTHORIZED


def test_missing_token_rejected(client):
    """A gated mesh route with no Authorization header returns 401."""
    assert client.get(GATED_PATH).status_code == HTTP_UNAUTHORIZED


def test_correct_token_accepted(client):
    """A gated mesh route with the configured bearer token returns 200."""
    assert client.get(GATED_PATH, headers=_bearer(TOKEN)).status_code == HTTP_OK


def test_regenerate_invalidates_old_token(client):
    """After token regeneration the previous token is rejected with 401."""
    client.post(REGENERATE_PATH)
    assert client.get(GATED_PATH, headers=_bearer(TOKEN)).status_code == HTTP_UNAUTHORIZED


def test_regenerated_token_accepted(client):
    """The token returned by regeneration is accepted on a gated mesh route."""
    new_token = client.post(REGENERATE_PATH).get_json()["token"]
    assert client.get(GATED_PATH, headers=_bearer(new_token)).status_code == HTTP_OK


def test_regenerate_writes_to_redirected_store(client, tmp_path):
    """Token regeneration persists the new token to the tmp_path settings file."""
    new_token = client.post(REGENERATE_PATH).get_json()["token"]
    stored = json.loads((tmp_path / SETTINGS_FILENAME).read_text(encoding="utf-8"))
    assert stored[TOKEN_KEY] == new_token


def test_real_settings_file_untouched(client):
    """Token regeneration leaves the real dashboard settings file byte-identical."""
    before = _read_bytes(REAL_SETTINGS_PATH)
    client.post(REGENERATE_PATH)
    assert _read_bytes(REAL_SETTINGS_PATH) == before
