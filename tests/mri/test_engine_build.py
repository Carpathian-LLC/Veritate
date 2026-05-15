# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - regression tests for the c engine rebuild flow shared by the Settings tab
#   "trigger build" button and the Generation tab inline "rebuild" link. both
#   POST /engine/build and both must honor the force flag so a user click is
#   never silently skipped when the binary looks fresh.
# tests/mri/test_engine_build.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys
import time

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def runner():
    """Import build_runner once. Module-scoped: module state is global."""
    if MRI_DIR not in sys.path:
        sys.path.insert(0, MRI_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from training import build_runner
    return build_runner


@pytest.fixture(scope="module")
def route_app():
    """Minimal Flask app with only engine_routes registered. Avoids importing
    the full veritate_mri.app (slow, side-effecty) just to exercise the
    /engine/build force-flag contract."""
    if MRI_DIR not in sys.path:
        sys.path.insert(0, MRI_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from flask import Flask
    from routes import engine_routes
    app = Flask(__name__)
    app.config["TESTING"] = True
    engine_routes.register(app)
    return app


@pytest.fixture(scope="module")
def client(route_app):
    with route_app.test_client() as c:
        yield c


def _reset_runner(runner, status):
    runner._STATE["status"] = status
    runner._STATE["error"]  = None


def _stub_thread(runner, monkeypatch):
    """Replace threading.Thread so start() does not actually spawn a build."""
    started = {"count": 0}
    class FakeThread:
        def __init__(self, *a, **kw): pass
        def start(self): started["count"] += 1
    monkeypatch.setattr(runner.threading, "Thread", FakeThread)
    return started


def test_start_no_force_skips_when_binary_present_and_fresh(runner, monkeypatch):
    """start(force=False) returns STATUS_OK without spawning a build when the binary is fresh."""
    started = _stub_thread(runner, monkeypatch)
    monkeypatch.setattr(runner, "_refresh_present", lambda: True)
    monkeypatch.setattr(runner, "_binary_is_stale", lambda: False)
    _reset_runner(runner, runner.STATUS_IDLE)
    state = runner.start(force=False)
    assert state["status"] == runner.STATUS_OK
    assert started["count"] == 0


def test_start_force_always_spawns_build(runner, monkeypatch):
    """start(force=True) always spawns the build, even if the binary is fresh."""
    started = _stub_thread(runner, monkeypatch)
    monkeypatch.setattr(runner, "_refresh_present", lambda: True)
    monkeypatch.setattr(runner, "_binary_is_stale", lambda: False)
    _reset_runner(runner, runner.STATUS_IDLE)
    state = runner.start(force=True)
    assert state["status"] == runner.STATUS_BUILDING
    assert started["count"] == 1


def test_start_status_building_set_synchronously(runner, monkeypatch):
    """STATUS_BUILDING is set before start() returns when a build will run.
    Closes the race where a caller polling /engine/status immediately after
    POST /engine/build would otherwise observe a transient ok/idle state and
    skip the wait loop."""
    _stub_thread(runner, monkeypatch)
    monkeypatch.setattr(runner, "_refresh_present", lambda: False)
    monkeypatch.setattr(runner, "_binary_is_stale", lambda: False)
    _reset_runner(runner, runner.STATUS_IDLE)
    state = runner.start(force=False)
    assert state["status"] == runner.STATUS_BUILDING


def test_start_noop_when_already_building(runner, monkeypatch):
    """A second start() while a build is in flight does not spawn another thread."""
    started = _stub_thread(runner, monkeypatch)
    _reset_runner(runner, runner.STATUS_BUILDING)
    state = runner.start(force=True)
    assert state["status"] == runner.STATUS_BUILDING
    assert started["count"] == 0


def test_engine_build_route_without_force_can_skip(client, runner, monkeypatch):
    """POST /engine/build with no body: route reaches build_runner.start(force=False)."""
    seen = {}
    def fake_start(force=False):
        seen["force"] = force
        return {"status": "ok"}
    monkeypatch.setattr(runner, "start", fake_start)
    r = client.post("/engine/build")
    assert r.status_code == 200
    assert seen["force"] is False


def test_engine_build_route_with_force_true_propagates(client, runner, monkeypatch):
    """POST /engine/build {"force":true} reaches build_runner.start(force=True).
    This is the contract shared by the Settings button and the Generation rebuild
    link: a user click must always trigger a real build."""
    seen = {}
    def fake_start(force=False):
        seen["force"] = force
        return {"status": "building"}
    monkeypatch.setattr(runner, "start", fake_start)
    r = client.post("/engine/build", json={"force": True})
    assert r.status_code == 200
    assert seen["force"] is True


def test_engine_build_route_with_force_false_skips(client, runner, monkeypatch):
    """Explicit force:false also propagates (defensive: ensures bool coercion)."""
    seen = {}
    def fake_start(force=False):
        seen["force"] = force
        return {"status": "ok"}
    monkeypatch.setattr(runner, "start", fake_start)
    r = client.post("/engine/build", json={"force": False})
    assert r.status_code == 200
    assert seen["force"] is False
