# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - /lifecycle/* are thin proxies onto runtime.lifecycle with the live app config; the
#   lifecycle module is stubbed so no process is restarted or killed here.
# tests/mri/test_lifecycle_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import lifecycle_routes

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch):
    calls = []
    for fn in ("restart", "kill", "soft_reload"):
        monkeypatch.setattr(lifecycle_routes.lifecycle, fn,
                            lambda cfg, fn=fn: (calls.append((fn, cfg.get("MARK"))), {"ok": True, "did": fn})[1])
    app = Flask(__name__)
    app.config["MARK"] = "this-config"
    lifecycle_routes.register(app)
    return app.test_client(), calls


@pytest.mark.parametrize("action", ["restart", "kill", "soft_reload"])
def test_each_lifecycle_action_proxies_with_the_app_config(rig, action):
    """POST /lifecycle/<action> calls lifecycle.<action>(current_app.config) and returns its answer."""
    client, calls = rig
    body = client.post(f"/lifecycle/{action}").get_json()
    assert body == {"ok": True, "did": action}
    assert calls == [(action, "this-config")]


def test_lifecycle_actions_are_post_only(rig):
    """A GET must not restart or kill anything."""
    client, calls = rig
    assert client.get("/lifecycle/kill").status_code == 405
    assert calls == []
