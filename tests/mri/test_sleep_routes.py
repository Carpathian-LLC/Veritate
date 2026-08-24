# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pins the sleep routes' model-parameter contract: /sleep/now and /sleep/wake
#   take a "model" (body or query); omitted, the only enrolled model is assumed;
#   with zero or several enrolled the request is a clean 400.
# tests/mri/test_sleep_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import sleep_routes
from runtime import settings as settings_mod
from training import sleep

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def env(monkeypatch):
    cfg = {"sleep_enabled": True, "sleep_models": ["toy"]}
    monkeypatch.setattr(settings_mod, "get", lambda: cfg)
    calls = {}
    monkeypatch.setattr(sleep, "maybe_sleep",
                        lambda force_idle=False, model=None: calls.setdefault("now", model) or "ok")
    monkeypatch.setattr(sleep, "wake",
                        lambda model: calls.setdefault("wake", model) or {"ok": True, "state": "awake"})
    app = Flask(__name__)
    sleep_routes.register(app)
    return app.test_client(), cfg, calls


def test_explicit_model_param_is_forwarded(env):
    """A "model" in the POST body reaches the controller for both actions."""
    client, cfg, calls = env
    cfg["sleep_models"] = ["toy", "quill"]
    assert client.post("/sleep/now", json={"model": "quill"}).status_code == 200
    assert calls["now"] == "quill"
    assert client.post("/sleep/wake", json={"model": "toy"}).status_code == 200
    assert calls["wake"] == "toy"


def test_omitted_model_defaults_to_only_enrollment(env):
    """No model + exactly one enrolled: that model is assumed."""
    client, _, calls = env
    assert client.post("/sleep/now", json={}).status_code == 200
    assert calls["now"] == "toy"
    assert client.post("/sleep/wake", json={}).status_code == 200
    assert calls["wake"] == "toy"


def test_omitted_model_with_several_enrolled_is_400(env):
    """No model + several enrolled: clean 400 naming the choices, controller
    never called."""
    client, cfg, calls = env
    cfg["sleep_models"] = ["toy", "quill"]
    r = client.post("/sleep/now", json={})
    assert r.status_code == 400 and "model" in r.get_json()["error"]
    r = client.post("/sleep/wake", json={})
    assert r.status_code == 400 and "quill" in r.get_json()["error"]
    assert calls == {}


def test_omitted_model_with_none_enrolled_is_400(env):
    """No model + nothing enrolled: clean 400, controller never called."""
    client, cfg, calls = env
    cfg["sleep_models"] = []
    r = client.post("/sleep/now", json={})
    assert r.status_code == 400 and "sleep_models" in r.get_json()["error"]
    assert calls == {}


def test_status_serves_multi_model_payload(env, monkeypatch):
    """GET /sleep proxies the controller's multi-model payload."""
    client, _, _ = env
    payload = {"enabled": True, "state": "awake", "models": [{"name": "toy"}]}
    monkeypatch.setattr(sleep, "status", lambda: payload)
    assert client.get("/sleep").get_json() == payload
