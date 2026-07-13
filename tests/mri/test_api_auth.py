# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - enforcement tests for the optional API-key gate (api_auth_routes). settings
#   are stubbed so no on-disk state is touched (rule 33). Covers open-when-unset,
#   401 without/with-wrong key, 200 with correct key, counter increment, and that
#   non-protected paths stay open even when a key is set.
# tests/mri/test_api_auth.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from flask import Flask

from routes import api_auth_routes

# ------------------------------------------------------------------------------------
# Constants

KEY = "vrt_testkey_abc123"


# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, key):
    """Minimal app with the gate registered and settings stubbed to `key`."""
    counter = {"hits": 0}
    monkeypatch.setattr(api_auth_routes.settings_mod, "get", lambda: {"api_key": key})
    def _record():
        counter["hits"] += 1
    monkeypatch.setattr(api_auth_routes.settings_mod, "record_api_key_use", _record)

    app = Flask(__name__)
    api_auth_routes.register(app)
    app.add_url_rule("/generate", "generate", lambda: {"ok": True})
    app.add_url_rule("/v1/models", "v1_models", lambda: {"ok": True})
    app.add_url_rule("/settings", "settings", lambda: {"ok": True})
    return app.test_client(), counter


def test_open_when_unset(monkeypatch):
    """Unset key leaves a protected endpoint open (200, no auth)."""
    client, _ = _client(monkeypatch, "")
    assert client.get("/generate").status_code == 200


def test_missing_key_rejected(monkeypatch):
    """Set key + no Authorization header returns 401."""
    client, _ = _client(monkeypatch, KEY)
    assert client.get("/generate").status_code == 401


def test_wrong_key_rejected(monkeypatch):
    """Set key + wrong Bearer token returns 401."""
    client, _ = _client(monkeypatch, KEY)
    r = client.get("/generate", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_correct_key_allowed(monkeypatch):
    """Set key + correct Bearer token returns 200."""
    client, _ = _client(monkeypatch, KEY)
    r = client.get("/generate", headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200


def test_correct_key_increments_counter(monkeypatch):
    """A successful authed request records one key use."""
    client, counter = _client(monkeypatch, KEY)
    client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert counter["hits"] == 1


def test_non_protected_path_open_when_set(monkeypatch):
    """A non-programmatic path (/settings) stays open even with a key set."""
    client, _ = _client(monkeypatch, KEY)
    assert client.get("/settings").status_code == 200
