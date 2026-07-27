# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for GET/POST /settings. the settings file and the module cache are
#   redirected into tmp_path so the machine-local data/mri_settings.json is never
#   read or written; one test asserts the write landed in the temp file.
# - the first-run device name generator is stubbed to a constant to keep the
#   fixture deterministic (rule 48).
# tests/mri/test_settings_route.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import types

import pytest
from flask import Flask
from routes import settings_routes
from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

DEVICE_NAME  = "test-box"
FIRST_RUN_NAME = "fixed-box"
UNKNOWN_KEY  = "not_a_real_setting"
UNKNOWN_VALUE = "stored?"
OVERLONG_NAME = "x" * (settings_mod.DEVICE_NAME_MAX_LEN + 1)

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def env(monkeypatch, tmp_path):
    """Test client for /settings with the settings file redirected into tmp_path."""
    path = tmp_path / "mri_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", str(path))
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    monkeypatch.setattr(settings_mod, "_random_device_name", lambda: FIRST_RUN_NAME)
    app = Flask(__name__)
    settings_routes.register(app)
    return types.SimpleNamespace(client=app.test_client(), path=path)


def test_post_then_get_round_trips_value(env):
    """A value written by POST /settings is returned by the next GET /settings."""
    env.client.post("/settings", json={"device_name": DEVICE_NAME})
    assert env.client.get("/settings").get_json()["device_name"] == DEVICE_NAME


def test_post_returns_the_written_value(env):
    """POST /settings echoes the merged settings including the written value."""
    body = env.client.post("/settings", json={"device_name": DEVICE_NAME}).get_json()
    assert body["device_name"] == DEVICE_NAME


def test_post_writes_to_the_patched_settings_file(env):
    """The write lands in the tmp_path settings file, never the machine-local one."""
    env.client.post("/settings", json={"device_name": DEVICE_NAME})
    assert json.loads(env.path.read_text(encoding="utf-8"))["device_name"] == DEVICE_NAME


def test_invalid_value_for_known_key_returns_400(env):
    """A known key with an out-of-range value is rejected with 400."""
    resp = env.client.post("/settings", json={"device_name": OVERLONG_NAME})
    assert resp.status_code == 400


def test_unknown_key_is_not_persisted(env):
    """An unknown key posted to /settings never appears in the stored settings."""
    env.client.post("/settings", json={UNKNOWN_KEY: UNKNOWN_VALUE})
    assert UNKNOWN_KEY not in env.client.get("/settings").get_json()


@pytest.mark.xfail(reason="settings.update() drops unknown keys silently; no 400 is raised", strict=True)
def test_unknown_key_is_rejected(env):
    """An unknown key posted to /settings is rejected with 400 rather than accepted and dropped."""
    assert env.client.post("/settings", json={UNKNOWN_KEY: UNKNOWN_VALUE}).status_code == 400
