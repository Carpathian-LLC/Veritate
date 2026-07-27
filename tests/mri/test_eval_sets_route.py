# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for /eval_sets: POST regenerates the smartness-meter eval sets, GET polls
#   the job, and a second POST is refused while one is in flight.
# - rebuild_all is stubbed with a recorder and the job runs inline instead of on a
#   thread, so nothing writes to veritate_mri/data/eval and ordering is deterministic
#   with no sleeps (rule 48).
# tests/mri/test_eval_sets_route.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib
import types

import pytest
from flask import Flask
from routes import runs_routes
from training.builders import eval as eval_builders

# ------------------------------------------------------------------------------------
# Constants

BUILT = [{"builder": "build_math_eval", "rc": 0}]

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def sets(monkeypatch):
    """Test client for /eval_sets with the builders stubbed and the job run inline."""
    state = types.SimpleNamespace(calls=0, client=None, during=None, during_result=None)

    def _rebuild_all():
        state.calls += 1
        if state.during is not None:
            call, state.during = state.during, None
            state.during_result = call()
        return BUILT

    monkeypatch.setattr(eval_builders, "rebuild_all", _rebuild_all)
    monkeypatch.setattr(runs_routes, "_eval_sets_spawn", runs_routes._eval_sets_job)
    runs_routes._EVAL_SETS_STATE.update(runs_routes.EVAL_SETS_IDLE)
    app = Flask(__name__)
    runs_routes.register(app)
    state.client = app.test_client()
    yield state
    runs_routes._EVAL_SETS_STATE.update(runs_routes.EVAL_SETS_IDLE)


@pytest.mark.parametrize("builder", eval_builders.BUILDERS)
def test_every_named_builder_is_callable(builder):
    """Each name in BUILDERS imports and exposes the main() rebuild_all drives."""
    assert callable(importlib.import_module(f"{eval_builders.__name__}.{builder}").main)


def test_post_returns_200(sets):
    """POST /eval_sets starts a rebuild and returns 200."""
    assert sets.client.post("/eval_sets").status_code == 200


def test_post_runs_every_builder(sets):
    """POST /eval_sets drives the eval-set builder module."""
    sets.client.post("/eval_sets")
    assert sets.calls == 1


def test_post_reports_the_builders_it_owns(sets):
    """POST /eval_sets names the eval sets it regenerates."""
    body = sets.client.post("/eval_sets").get_json()
    assert body["known_builders"] == list(eval_builders.BUILDERS)


def test_status_reports_the_finished_job(sets):
    """GET /eval_sets reports the completed rebuild with no error."""
    sets.client.post("/eval_sets")
    body = sets.client.get("/eval_sets").get_json()
    assert body["running"] is False and body["error"] is None


def test_status_lists_what_was_rebuilt(sets):
    """GET /eval_sets carries the per-builder result of the last rebuild."""
    sets.client.post("/eval_sets")
    assert sets.client.get("/eval_sets").get_json()["builders"] == BUILT


def test_second_post_during_a_rebuild_returns_409(sets):
    """POST /eval_sets while a rebuild holds the claim returns 409."""
    sets.during = lambda: sets.client.post("/eval_sets")
    sets.client.post("/eval_sets")
    assert sets.during_result.status_code == 409


def test_second_post_during_a_rebuild_runs_no_second_job(sets):
    """POST /eval_sets while a rebuild holds the claim starts no second job."""
    sets.during = lambda: sets.client.post("/eval_sets")
    sets.client.post("/eval_sets")
    assert sets.calls == 1


def test_failed_rebuild_reports_the_error(sets, monkeypatch):
    """A builder that raises leaves the job stopped with the error in status."""
    def _boom():
        raise RuntimeError("source passage missing")

    monkeypatch.setattr(eval_builders, "rebuild_all", _boom)
    sets.client.post("/eval_sets")
    body = sets.client.get("/eval_sets").get_json()
    assert body["running"] is False and "source passage missing" in body["error"]
