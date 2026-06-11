# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the grounded job runner: input validation and the atomic
#   single-job claim that prevents two concurrent POSTs from both spawning.
# tests/mri/test_grounded.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
for p in (REPO_ROOT, MRI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from routes import grounded_routes


# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def client():
    """Flask test client for the MRI app."""
    from veritate_mri.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Restore the runner to idle around every test."""
    grounded_routes._STATE.update(status="idle", phase=None)
    yield
    grounded_routes._STATE.update(status="idle", phase=None)


def test_claim_is_atomic_single_winner():
    """Two claims without an intervening reset: first wins, second is refused."""
    assert grounded_routes._claim("train") is True
    assert grounded_routes._claim("train") is False


def test_second_train_post_returns_409(client, monkeypatch):
    """A second /grounded/train while one is claimed returns 409 without spawning."""
    spawned = []
    monkeypatch.setattr(grounded_routes, "_spawn", lambda phase, argv: spawned.append(phase))
    ok = client.post("/grounded/train", json={"source": "src", "name": "dst", "steps": 5})
    assert ok.status_code == 200
    busy = client.post("/grounded/train", json={"source": "src", "name": "dst", "steps": 5})
    assert busy.status_code == 409
    assert spawned == ["train"]


def test_train_requires_positive_steps(client):
    """/grounded/train rejects non-positive steps with 400 before claiming."""
    r = client.post("/grounded/train", json={"source": "src", "name": "dst", "steps": 0})
    assert r.status_code == 400
    assert grounded_routes._STATE["status"] == "idle"
