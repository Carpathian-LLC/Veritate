# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for /teacher/target_status, the guard behind the Distillation tab's
#   "training is running on that machine" confirm. The question it answers is
#   narrow: will these teacher calls land on a box that is already training?
# - A hosted API never contends. A local provider on this box contends only while
#   trainer_runner reports running. A local provider pointed at ANOTHER box is
#   unknowable from here and must report None, not a reassuring False.
# - trainer_runner.state and settings are stubbed; nothing reads real settings or
#   touches a real trainer (rule 48).
# tests/mri/test_teacher_target_status.py
# ------------------------------------------------------------------------------------
# Imports:

import socket

import pytest
from flask import Flask
from routes import teacher_routes

# ------------------------------------------------------------------------------------
# Constants

OLLAMA_LOCAL = {
    "teacher_provider": "ollama",
    "teacher_model": "qwen2.5:14b-instruct",
    "teacher_configs": {"ollama": {"base_url": "http://localhost:11434", "model": "", "api_key": ""}},
}
OLLAMA_REMOTE = {
    "teacher_provider": "ollama",
    "teacher_model": "qwen2.5:14b-instruct",
    "teacher_configs": {"ollama": {"base_url": "http://cardinal-01.local:11434", "model": "", "api_key": ""}},
}
CLOUD = {
    "teacher_provider": "carpathian",
    "teacher_model": "some-model",
    "teacher_configs": {"carpathian": {"base_url": "https://api.carpathian.ai/ai/v1",
                                       "model": "", "api_key": "k"}},
}
RUNNING_STATE = {
    "status": "running",
    "plugin_id": "veritate_trainer",
    "args": {"name": "wren", "size": "200m"},
    "started_at": 1755000000,
}
IDLE_STATE = {"status": "idle", "plugin_id": None, "args": None, "started_at": None}

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def client(monkeypatch):
    """Test client for /teacher/target_status with settings and trainer state stubbed."""
    state = {"settings": dict(OLLAMA_LOCAL), "trainer": dict(IDLE_STATE)}

    monkeypatch.setattr(teacher_routes.settings_mod, "get", lambda: state["settings"])
    monkeypatch.setattr(teacher_routes.trainer_runner, "state", lambda: state["trainer"])

    app = Flask(__name__)
    teacher_routes.register(app)
    c = app.test_client()

    def _get(settings=None, trainer=None):
        if settings is not None:
            state["settings"] = dict(settings)
        if trainer is not None:
            state["trainer"] = dict(trainer)
        r = c.get("/teacher/target_status")
        assert r.status_code == 200
        return r.get_json()

    return _get


def test_local_teacher_idle_trainer_does_not_contend(client):
    """Teacher on this box with no run active: safe to start, no confirm."""
    d = client(OLLAMA_LOCAL, IDLE_STATE)
    assert d["targets_this_machine"] is True
    assert d["training_active"] is False
    assert d["contention"] is False
    assert d["run"] is None


def test_local_teacher_running_trainer_contends(client):
    """Teacher on this box while a run is training: this is the confirm case."""
    d = client(OLLAMA_LOCAL, RUNNING_STATE)
    assert d["targets_this_machine"] is True
    assert d["training_active"] is True
    assert d["contention"] is True
    assert d["contention_kind"] == teacher_routes.CONTENTION_LOCAL_RUN


def test_contending_run_is_named_so_the_confirm_can_quote_it(client):
    """The confirm has to say WHICH run it would be competing with."""
    run = client(OLLAMA_LOCAL, RUNNING_STATE)["run"]
    assert run["name"] == "wren"
    assert run["size"] == "200m"
    assert run["plugin_id"] == "veritate_trainer"
    assert run["started_at"] == RUNNING_STATE["started_at"]


def test_cloud_teacher_never_contends_even_while_training(client):
    """A hosted API uses none of this machine, so a live run is irrelevant."""
    d = client(CLOUD, RUNNING_STATE)
    assert d["kind"] == "api"
    assert d["targets_this_machine"] is False
    assert d["training_active"] is False
    assert d["contention"] is False


def test_remote_local_teacher_reports_unknown_not_false(client):
    """A local provider on another box: we cannot see its trainer. Saying False
    there would be a guess dressed as a fact, so it reports None."""
    d = client(OLLAMA_REMOTE, RUNNING_STATE)
    assert d["kind"] == "local"
    assert d["targets_this_machine"] is False
    assert d["training_active"] is None
    assert d["contention"] is False
    assert "not visible" in d["reason"]


def test_this_machine_by_hostname_also_counts_as_local(client, monkeypatch):
    """base_url naming this box by hostname is the same machine as localhost."""
    host = socket.gethostname().split(".")[0]
    cfg = {
        "teacher_provider": "ollama",
        "teacher_model": "m",
        "teacher_configs": {"ollama": {"base_url": f"http://{host}:11434", "model": "", "api_key": ""}},
    }
    d = client(cfg, RUNNING_STATE)
    assert d["targets_this_machine"] is True
    assert d["contention"] is True


def test_unconfigured_teacher_does_not_contend(client):
    """No teacher selected: nothing to guard, and no exception either."""
    d = client({"teacher_provider": "", "teacher_configs": {}}, RUNNING_STATE)
    assert d["kind"] == ""
    assert d["contention"] is False


def test_trainer_state_failure_never_blocks_the_guard(client, monkeypatch):
    """The guard must degrade to 'no contention', never to a 500 that stops a start."""
    def _boom():
        raise RuntimeError("trainer state unavailable")

    monkeypatch.setattr(teacher_routes.trainer_runner, "state", _boom)
    d = client(OLLAMA_LOCAL, None)
    assert d["contention"] is False
    assert d["run"] is None


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:11434", "localhost"),
    ("http://127.0.0.1:11434", "127.0.0.1"),
    ("https://api.carpathian.ai/ai/v1", "api.carpathian.ai"),
    ("", ""),
    ("not a url", ""),
])
def test_target_host_parses_the_base_url(url, expected):
    """Host extraction is what decides local vs remote, so it is pinned."""
    assert teacher_routes._target_host(url) == expected


def test_status_separates_call_counts_from_record_counts(tmp_path):
    """state.json counts calls; samples.jsonl counts records. The status route
    must publish both under names that cannot be confused."""
    import json as _json

    from routes import teacher_routes as tr
    (tmp_path / "state.json").write_text(_json.dumps({
        "completed": 2, "failed": 5, "remaining": ["a", "b", "c"],
        "authoring": {"records": 21},
    }))
    (tmp_path / "samples.jsonl").write_text("{}\n" * 21)
    counts = tr._read_state_counts(str(tmp_path))
    assert counts["completed"] == 21          # records
    assert counts["calls_ok"] == 2            # calls
    assert counts["calls_failed"] == 5
    assert counts["calls_remaining"] == 3
