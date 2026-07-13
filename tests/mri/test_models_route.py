# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for /v1/models (OpenAI-compatible list). readers are stubbed so no
#   filesystem models or checkpoint loads are needed (rule 33); assertions cover the
#   payload envelope, extra-field passthrough, and newest-first order.
# tests/mri/test_models_route.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from flask import Flask

from routes import models_routes

# ------------------------------------------------------------------------------------
# Constants

ROWS = [
    {"name": "old_model", "step": 5, "is_current": False, "plugin": "",
     "n_params": 800000000, "hidden": 1536, "layers": 24, "description": "",
     "mtime": 100, "capabilities": {"chat": True}},
    {"name": "new_model", "step": 9, "is_current": True, "plugin": "",
     "n_params": None, "hidden": 512, "layers": 8, "description": "",
     "mtime": 200, "capabilities": {}},
]

# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, rows):
    """Minimal app with models_routes registered and _model_rows stubbed."""
    monkeypatch.setattr(models_routes, "_model_rows", lambda: sorted(rows, key=lambda r: -r["mtime"]))
    app = Flask(__name__)
    models_routes.register(app)
    return app.test_client()


def test_v1_models_envelope(monkeypatch):
    """GET /v1/models returns the OpenAI list envelope with object:'list'."""
    body = _client(monkeypatch, ROWS).get("/v1/models").get_json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_v1_models_entry_shape(monkeypatch):
    """Each entry has id/object/created/owned_by and the preserved extras."""
    entry = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"][0]
    assert entry["object"] == "model"
    assert entry["owned_by"] == models_routes.OWNER
    for k in ("id", "created", "n_params", "hidden", "layers", "capabilities", "is_current"):
        assert k in entry


def test_v1_models_id_and_created_from_row(monkeypatch):
    """id maps from name and created is the int mtime of the model."""
    top = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"][0]
    assert top["id"] == "new_model"
    assert top["created"] == 200


def test_v1_models_sorted_newest_first(monkeypatch):
    """Entries are ordered newest-first by created, matching /pytorch-models."""
    data = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"]
    assert [e["id"] for e in data] == ["new_model", "old_model"]
