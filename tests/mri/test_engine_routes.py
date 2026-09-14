# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the C-engine routes: build status carries the live exe and subprocess flag, engine
#   and model listings mark the current one and skip what is not on disk, and /c-config
#   refuses missing files, closes the old subprocess and installs the new one (or leaves
#   none on a failed spawn). Readers, the build runner and the subprocess are stubbed.
# tests/mri/test_engine_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from flask import Flask
from routes import engine_routes

# ------------------------------------------------------------------------------------
# Functions


class _Proc:
    pid = 4242


SPAWNED = []


class _Sub:
    def __init__(self, exe, model):
        self.exe, self.model, self.closed, self.proc = exe, model, False, _Proc()
        SPAWNED.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    exe_a, exe_b = tmp_path / "engine_a", tmp_path / "engine_b"
    exe_a.write_bytes(b"x")
    exe_b.write_bytes(b"xy")
    monkeypatch.setattr(engine_routes.engine, "engines",
                        lambda: [{"name": "a", "path": str(exe_a)}, {"name": "b", "path": str(exe_b)},
                                 {"name": "gone", "path": str(tmp_path / "missing")}])
    mdir = tmp_path / "models"
    for name, size in (("old", 3), ("new", 5)):
        (mdir / name).mkdir(parents=True)
        (mdir / name / "model.bin").write_bytes(b"b" * size)
    os.utime(mdir / "new" / "model.bin", (200, 200))
    os.utime(mdir / "old" / "model.bin", (100, 100))
    monkeypatch.setattr(engine_routes.models, "list_models", lambda: ["old", "nobin", "new"])
    monkeypatch.setattr(engine_routes.binr, "exists", lambda n: n != "nobin")
    monkeypatch.setattr(engine_routes.binr, "header", lambda n: ("int8", 13))
    monkeypatch.setattr(engine_routes.binr, "act_boost", lambda n: 2 if n == "old" else None)
    monkeypatch.setattr(engine_routes.paths, "bin_path", lambda n: str(mdir / n / "model.bin"))
    monkeypatch.setattr(engine_routes.cfg_reader, "training_kind", lambda n: ("chat", "gelu"))
    monkeypatch.setattr(engine_routes.cfg_reader, "qat_enabled", lambda n: n == "old")
    monkeypatch.setattr(engine_routes.cfg_reader, "description", lambda n: "desc " + n)
    monkeypatch.setattr(engine_routes.build_runner, "state", lambda: {"status": "idle"})
    monkeypatch.setattr(engine_routes.build_runner, "start", lambda force: {"ok": True, "force": force})
    monkeypatch.setattr(engine_routes, "CTracedSubprocess", _Sub)
    SPAWNED.clear()
    app = Flask(__name__)
    app.config.update(C_EXE=str(exe_a), C_MODEL=str(mdir / "old" / "model.bin"), C_SUBPROCESS=None)
    engine_routes.register(app)
    return app, app.test_client(), str(exe_b), str(mdir / "new" / "model.bin")


def test_status_carries_the_build_state_the_exe_and_the_subprocess_flag(rig):
    app, client, _exe_b, _new = rig
    assert client.get("/engine/status").get_json() == {"status": "idle", "c_subprocess_running": False,
                                                       "c_exe": app.config["C_EXE"]}


def test_build_trigger_passes_force_through(rig):
    _app, client, _exe_b, _new = rig
    assert client.post("/engine/build", json={"force": True}).get_json() == {"ok": True, "force": True}
    assert client.post("/engine/build").get_json()["force"] is False


def test_engine_listing_marks_the_current_one_and_skips_what_is_not_on_disk(rig):
    _app, client, _exe_b, _new = rig
    rows = client.get("/c-engines").get_json()["engines"]
    assert [r["name"] for r in rows] == ["a", "b"]
    assert [r["is_current"] for r in rows] == [True, False]
    assert rows[1]["size"] == 2


def test_model_listing_is_newest_first_with_only_the_models_that_have_a_bin(rig):
    _app, client, _exe_b, _new = rig
    rows = client.get("/c-models").get_json()["models"]
    assert [r["name"] for r in rows] == ["new", "old"]
    assert rows[1]["is_current"] and not rows[0]["is_current"]
    assert rows[1] | {"precision": "int8", "bin_version": 13, "act_boost": 2, "qat_enabled": True} == rows[1]


def test_c_config_refuses_a_missing_exe_or_model(rig):
    _app, client, _exe_b, _new = rig
    assert client.post("/c-config", json={"exe": "/nowhere"}).status_code == 400
    assert client.post("/c-config", json={"model": "/nowhere.bin"}).status_code == 400
    assert SPAWNED == []


def test_c_config_closes_the_old_subprocess_and_installs_the_new_one(rig):
    app, client, exe_b, new = rig
    old = _Sub("x", "y")
    app.config["C_SUBPROCESS"] = old
    body = client.post("/c-config", json={"exe": exe_b, "model": new}).get_json()
    assert body["ok"] and body["c_model_dir"] == "new" and body["c_model_precision"] == "int8"
    assert old.closed
    assert app.config["C_SUBPROCESS"] is SPAWNED[-1]
    assert (app.config["C_EXE"], app.config["C_MODEL"]) == (exe_b, new)


def test_a_failed_spawn_is_a_500_and_leaves_no_subprocess(rig, monkeypatch):
    app, client, exe_b, new = rig

    class Boom:
        def __init__(self, exe, model):
            raise OSError("cannot exec")
    monkeypatch.setattr(engine_routes, "CTracedSubprocess", Boom)
    res = client.post("/c-config", json={"exe": exe_b, "model": new})
    assert res.status_code == 500 and "cannot exec" in res.get_json()["error"]
    assert app.config["C_SUBPROCESS"] is None
