# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for POST /trainers/run, the only sanctioned training launch
#   (preflight rule 24a). subprocess.Popen is replaced by a recorder so no trainer
#   process is ever spawned; the pid file and run log are redirected into tmp_path.
# - covers argv/env passthrough of model_type, rule 24a's omitted-model_type
#   default (VERITATE_MODEL_TYPE stays unset, so the trainer falls back to
#   language), and that an invalid body launches nothing at all.
# tests/mri/test_trainers_run_route.py
# ------------------------------------------------------------------------------------
# Imports:

import subprocess
import threading
import types

import pytest
from flask import Flask
from routes import trainers_routes
from runtime import heartbeat
from training import trainer_runner

# ------------------------------------------------------------------------------------
# Constants

TRAINER_ID  = "fixture/trainer"
UNKNOWN_ID  = "fixture/absent"
RUN_ARGS    = {"size": "5m", "total_steps": 1}
FINISH_SECS = 10.0
FAKE_PID    = 424242

IDLE_STATE = {"status": trainer_runner.STATUS_IDLE, "plugin_id": None, "args": None,
              "started_at": None, "finished_at": None, "exit_code": None}

# ------------------------------------------------------------------------------------
# Functions

class _FakeProc:
    def __init__(self, argv, env):
        self.argv       = list(argv)
        self.env        = dict(env or {})
        self.pid        = FAKE_PID
        self.returncode = 0

    def wait(self):
        return self.returncode


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.fixture
def runner(monkeypatch, tmp_path):
    """Test client for /trainers/run with Popen recorded, one fixture trainer, temp run state."""
    script = tmp_path / "trainer.py"
    script.write_text("", encoding="utf-8")
    plugin = {"id": TRAINER_ID, "file": script.name, "path": str(script),
              "manifest": {"kind": "trainer", "defaults": {}},
              "bundle_dir": None, "bundle_corpus_dir": None}
    state = types.SimpleNamespace(spawned=[], launched=threading.Event(),
                                  finished=threading.Event(), client=None)

    def _popen(argv, **kw):
        proc = _FakeProc(argv, kw.get("env"))
        state.spawned.append(proc)
        state.launched.set()
        return proc

    plain_set = trainer_runner._set

    def _set_spy(**kw):
        plain_set(**kw)
        if kw.get("status") in (trainer_runner.STATUS_OK, trainer_runner.STATUS_FAILED):
            state.finished.set()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(trainer_runner, "_set", _set_spy)
    monkeypatch.setattr(trainer_runner, "PID_FILE", str(tmp_path / "plugin_pid.json"))
    monkeypatch.setattr(trainer_runner, "RUN_LOG_FILE", str(tmp_path / "plugin_run.log"))
    monkeypatch.setattr(trainer_runner.plugins_reader, "scan", lambda: [plugin])
    monkeypatch.setattr(trainer_runner.plugins_reader, "update_defaults", lambda *a, **k: False)
    monkeypatch.setattr(trainer_runner.settings_mod, "get", dict)
    monkeypatch.setattr(heartbeat, "record_training_event", lambda *a, **k: None)
    monkeypatch.delenv(trainer_runner.MODEL_TYPE_ENV, raising=False)
    trainer_runner._STATE.update(IDLE_STATE)
    app = Flask(__name__)
    trainers_routes.register(app)
    state.client = app.test_client()
    yield state
    if state.launched.is_set():
        state.finished.wait(FINISH_SECS)
    trainer_runner._STATE.update(IDLE_STATE)


def _post(runner, body):
    return runner.client.post("/trainers/run", json=body)


def _launch(runner, args):
    resp = _post(runner, {"id": TRAINER_ID, "args": args})
    runner.launched.wait(FINISH_SECS)
    return resp


def test_valid_body_returns_200(runner):
    """POST /trainers/run with a known trainer id returns 200."""
    assert _launch(runner, RUN_ARGS).status_code == 200


def test_valid_body_reports_ok(runner):
    """POST /trainers/run with a known trainer id reports ok=true."""
    assert _launch(runner, RUN_ARGS).get_json()["ok"] is True


def test_valid_body_spawns_one_process(runner):
    """A valid launch spawns exactly one trainer process."""
    _launch(runner, RUN_ARGS)
    assert len(runner.spawned) == 1


def test_model_type_code_reaches_argv(runner):
    """model_type=code in the body arrives on the trainer argv as --model_type code."""
    _launch(runner, {**RUN_ARGS, "model_type": "code"})
    assert _flag_value(runner.spawned[0].argv, "--model_type") == "code"


def test_model_type_code_exported_to_env(runner):
    """model_type=code exports VERITATE_MODEL_TYPE=code to the trainer process."""
    _launch(runner, {**RUN_ARGS, "model_type": "code"})
    assert runner.spawned[0].env[trainer_runner.MODEL_TYPE_ENV] == "code"


def test_omitted_model_type_leaves_env_unset(runner):
    """An omitted model_type exports no VERITATE_MODEL_TYPE, the rule 24a silent language default."""
    _launch(runner, RUN_ARGS)
    assert trainer_runner.MODEL_TYPE_ENV not in runner.spawned[0].env


def test_omitted_model_type_leaves_argv_flagless(runner):
    """An omitted model_type puts no --model_type flag on the trainer argv."""
    _launch(runner, RUN_ARGS)
    assert _flag_value(runner.spawned[0].argv, "--model_type") is None


def test_missing_id_returns_400(runner):
    """POST /trainers/run without an id returns 400."""
    assert _post(runner, {"args": RUN_ARGS}).status_code == 400


def test_missing_id_spawns_nothing(runner):
    """POST /trainers/run without an id spawns no process."""
    _post(runner, {"args": RUN_ARGS})
    assert runner.spawned == []


def test_unknown_trainer_reports_not_ok(runner):
    """POST /trainers/run with an unregistered trainer id reports ok=false."""
    assert _post(runner, {"id": UNKNOWN_ID, "args": RUN_ARGS}).get_json()["ok"] is False


def test_unknown_trainer_spawns_nothing(runner):
    """POST /trainers/run with an unregistered trainer id spawns no process."""
    _post(runner, {"id": UNKNOWN_ID, "args": RUN_ARGS})
    assert runner.spawned == []


def test_unknown_trainer_leaves_runner_idle(runner):
    """A rejected launch releases the run claim so the next launch is not blocked."""
    _post(runner, {"id": UNKNOWN_ID, "args": RUN_ARGS})
    assert trainer_runner.state()["status"] == trainer_runner.STATUS_IDLE


def test_a_name_whose_previous_attempt_never_saved_a_checkpoint_may_be_relaunched(runner, monkeypatch, tmp_path):
    """Out of memory at step 1 leaves config.json and no weights. That is not a model to
    protect; the same name launches again. Weights on disk keep the 409."""
    import json
    import os

    from readers import paths
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    name, size = "retry_me", "20m"
    composed = f"{name}_{size}"
    os.makedirs(paths.model_dir(composed))
    with open(paths.config_path(composed), "w", encoding="utf-8") as handle:
        json.dump({"name": composed}, handle)
    resp = _launch(runner, {"name": name, "size": size})
    assert resp.status_code == 200 and resp.get_json()["ok"]
    os.makedirs(paths.checkpoints_dir(composed), exist_ok=True)
    with open(paths.checkpoint_path(composed, 1), "wb") as handle:
        handle.write(b"weights")
    resp = _post(runner, {"id": TRAINER_ID, "args": {"name": name, "size": size}})
    assert resp.status_code == 409
    assert "already exists" in resp.get_json()["error"]
