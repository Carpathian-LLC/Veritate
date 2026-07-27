# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - state-machine tests for /rag/build_corpus and /rag/train: one job at a time,
#   409 while a job holds the claim, and the claim releasing when the job ends.
#   nothing is executed: subprocess.Popen is a recorder and the job runs inline
#   instead of on a thread, which makes ordering deterministic with no sleeps
#   (rule 48).
# - both spawned entry points must ship in the repo, so the flow works on a
#   fresh clone; test_entry_points_ship_in_the_platform is that guard.
# - the recorder can run a caller-supplied request while the job still holds the
#   claim, which is how the concurrent-request case is exercised.
# - the module-level run state and log path are reset around every test (rule 49).
# tests/mri/test_rag_state.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import types

import pytest
from conftest import REPO_ROOT
from flask import Flask
from routes import rag_routes

# ------------------------------------------------------------------------------------
# Constants

SOURCE_MODEL = "fixture_model"
N_FACTS      = 1
GITIGNORED   = os.path.join(REPO_ROOT, "experiments")

IDLE_STATE = {"status": rag_routes.STATUS_IDLE, "phase": None, "started_at": None,
              "finished_at": None, "exit_code": None}

# ------------------------------------------------------------------------------------
# Functions

class _FakeProc:
    def __init__(self, argv):
        self.argv       = list(argv)
        self.returncode = 0

    def wait(self):
        return self.returncode


@pytest.fixture
def rag(monkeypatch, tmp_path):
    """Test client for the rag routes with Popen recorded, jobs run inline, temp log file."""
    state = types.SimpleNamespace(spawned=[], during=None, during_result=None, client=None)

    def _popen(argv, **_kw):
        proc = _FakeProc(argv)
        state.spawned.append(proc)
        if state.during is not None:
            call, state.during = state.during, None
            state.during_result = call()
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(rag_routes, "LOG_FILE", str(tmp_path / "rag_run.log"))
    monkeypatch.setattr(rag_routes, "_spawn", rag_routes._run_steps)
    rag_routes._STATE.update(IDLE_STATE)
    app = Flask(__name__)
    rag_routes.register(app)
    state.client = app.test_client()
    yield state
    rag_routes._STATE.update(IDLE_STATE)


def _build(rag):
    return rag.client.post("/rag/build_corpus", json={"n_facts": N_FACTS})


def _train(rag):
    return rag.client.post("/rag/train", json={"source": SOURCE_MODEL})


def _train_argvs(rag):
    return [p.argv for p in rag.spawned if rag_routes.TRAIN_SCRIPT in p.argv]


@pytest.mark.parametrize("script", [rag_routes.BUILD_SCRIPT, rag_routes.TRAIN_SCRIPT])
def test_entry_points_ship_in_the_platform(script):
    """Every rag entry point is a shipped platform file, never gitignored scratch."""
    assert os.path.isfile(script) and not script.startswith(GITIGNORED)


def test_train_during_active_build_returns_409(rag):
    """POST /rag/train while a build holds the run claim returns 409."""
    rag.during = lambda: _train(rag)
    _build(rag)
    assert rag.during_result.status_code == 409


def test_train_during_active_build_spawns_no_process(rag):
    """POST /rag/train while a build holds the run claim launches no second process."""
    rag.during = lambda: _train(rag)
    _build(rag)
    assert _train_argvs(rag) == []


def test_build_releases_the_claim_when_it_ends(rag):
    """A finished build leaves the state machine out of the running status."""
    _build(rag)
    assert rag.client.get("/rag/status").get_json()["running"] is False


def test_train_accepted_after_build_ends(rag):
    """POST /rag/train after the build finished returns 200."""
    _build(rag)
    assert _train(rag).status_code == 200


def test_train_spawns_the_trainer_after_build_ends(rag):
    """POST /rag/train after the build finished launches the rag trainer."""
    _build(rag)
    _train(rag)
    assert len(_train_argvs(rag)) == 1


def test_train_without_source_returns_400(rag):
    """POST /rag/train with no source model returns 400."""
    assert rag.client.post("/rag/train", json={}).status_code == 400


def test_train_without_source_spawns_nothing(rag):
    """POST /rag/train with no source model launches no process."""
    rag.client.post("/rag/train", json={})
    assert rag.spawned == []


def test_build_without_facts_returns_400(rag):
    """POST /rag/build_corpus with a non-positive n_facts returns 400."""
    assert rag.client.post("/rag/build_corpus", json={"n_facts": 0}).status_code == 400
