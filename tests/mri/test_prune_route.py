# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POST /models/checkpoints/prune. The route's one job beyond wiring is the default:
#   dry_run is TRUE unless the caller says otherwise, so a malformed or replayed
#   request plans instead of deleting. That default is what the test pins.
# tests/mri/test_prune_route.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import time

import pytest
from flask import Flask
from readers import checkpoints, paths
from routes import models_routes

# ------------------------------------------------------------------------------------
# Constants

MODEL = "toy_route"
STEPS = list(range(500, 5001, 500))
OLD_S = 86_400

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    os.makedirs(paths.checkpoints_dir(MODEL))
    with open(paths.config_path(MODEL), "w") as f:
        f.write("{}")
    old = time.time() - OLD_S
    for step in STEPS:
        path = checkpoints.path_for(MODEL, step)
        with open(path, "wb") as f:
            f.write(b"x" * 32)
        os.utime(path, (old, old))
    app = Flask(__name__)
    models_routes.register(app)
    return app.test_client()


def _post(client, body):
    return client.post("/models/checkpoints/prune", json=body)


def test_dry_run_is_the_default(client):
    """No dry_run key must mean plan, never delete."""
    r = _post(client, {"name": MODEL, "keep_every": 5000, "keep_last": 1})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert checkpoints.list_steps(MODEL) == STEPS


def test_explicit_dry_run_false_deletes(client):
    r = _post(client, {"name": MODEL, "keep_every": 5000, "keep_last": 1, "dry_run": False})
    assert r.get_json()["deleted"]
    assert checkpoints.list_steps(MODEL) == [5000]


def test_bad_name_is_a_400_not_a_500(client):
    r = _post(client, {"name": "nope"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_keep_last_zero_is_rejected_before_any_delete(client):
    r = _post(client, {"name": MODEL, "keep_last": 0, "dry_run": False})
    assert r.status_code == 400
    assert checkpoints.list_steps(MODEL) == STEPS


def test_plan_reports_the_hooks_it_preserves(client):
    r = _post(client, {"name": MODEL})
    assert "hooks_kept" in r.get_json()
