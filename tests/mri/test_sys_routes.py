# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - /sys/*, /heartbeat/*, /app/update_* and /versions: the minimal-mode contract with
#   the launcher, the flag toggle a relaunch passes to lifecycle, the specs fallback, the
#   dep snapshot bundled into a detect, the reload after a pulled update (and its error
#   surfacing), and the versions file served raw. Every runtime module is stubbed.
# tests/mri/test_sys_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import sys_routes

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(sys_routes.sys_metrics, "snapshot", lambda: {"cpu": 1})
    monkeypatch.setattr(sys_routes.sys_metrics, "load_specs", lambda: None)
    monkeypatch.setattr(sys_routes.sys_metrics, "detect_and_save", lambda: {"detected": True})
    monkeypatch.setattr(sys_routes.heartbeat_mod, "arm_hw_redump", lambda: calls.setdefault("redump", True))
    monkeypatch.setattr(sys_routes.heartbeat_mod, "status", lambda: {"tier": "full"})
    monkeypatch.setattr(sys_routes.heartbeat_mod, "send_now", lambda: True)
    monkeypatch.setattr(sys_routes.lifecycle, "restart_with_flag_toggle",
                        lambda cfg, add_flags, remove_flags: {"add": list(add_flags), "remove": list(remove_flags)})
    monkeypatch.setattr(sys_routes.lifecycle, "restart", lambda cfg: calls.setdefault("restart", True))
    monkeypatch.setattr(sys_routes.app_sync_mod, "pull_update",
                        lambda force, ignore_training: {"ok": True, "force": force, "ignore_training": ignore_training})
    monkeypatch.setattr(sys_routes.app_sync_mod, "switch_channel", lambda ch: {"channel": ch})
    monkeypatch.setattr(sys_routes.paths, "VERSIONS_JSON_PATH", str(tmp_path / "versions.json"))
    monkeypatch.delenv(sys_routes.MINIMAL_ENV, raising=False)
    app = Flask(__name__)
    sys_routes.register(app)
    return app.test_client(), calls, tmp_path


def test_mode_reads_the_launcher_env_var(rig, monkeypatch):
    client, _calls, _tmp = rig
    assert client.get("/sys/mode").get_json() == {"minimal": False}
    monkeypatch.setenv(sys_routes.MINIMAL_ENV, sys_routes.MINIMAL_ON)
    assert client.get("/sys/mode").get_json() == {"minimal": True}


def test_relaunch_toggles_the_minimal_flag_in_the_requested_direction(rig):
    client, _calls, _tmp = rig
    flag = sys_routes.MINIMAL_FLAG
    assert client.post("/sys/mode/relaunch", json={"minimal": True}).get_json() == {"add": [flag], "remove": []}
    assert client.post("/sys/mode/relaunch", json={}).get_json() == {"add": [], "remove": [flag]}


def test_specs_report_not_detected_when_nothing_is_saved(rig):
    client, _calls, _tmp = rig
    assert client.get("/sys/specs").get_json() == {"detected": False}


def test_detect_rearms_the_hardware_dump_and_bundles_the_dep_snapshot(rig, monkeypatch):
    client, calls, _tmp = rig
    from veritate_core.plugin import deps as deps_mod
    monkeypatch.setattr(deps_mod, "status_snapshot", lambda: {"missing": ["x"]})
    body = client.post("/sys/detect").get_json()
    assert body == {"detected": True, "deps": {"missing": ["x"]}}
    assert calls["redump"] is True


def test_heartbeat_send_reports_the_outcome_beside_the_status(rig):
    client, _calls, _tmp = rig
    assert client.post("/heartbeat/send").get_json() == {"ok": True, "tier": "full"}


def test_update_pull_passes_the_flags_and_reloads_only_when_asked(rig):
    client, calls, _tmp = rig
    body = client.post("/app/update_pull", json={"force": True, "ignore_training": True}).get_json()
    assert body == {"ok": True, "force": True, "ignore_training": True} and "restart" not in calls
    client.post("/app/update_pull", json={"reload": True})
    assert calls["restart"] is True


def test_a_failed_reload_after_a_pull_is_reported_not_raised(rig, monkeypatch):
    client, _calls, _tmp = rig

    def boom(cfg):
        raise OSError("no exec")
    monkeypatch.setattr(sys_routes.lifecycle, "restart", boom)
    body = client.post("/app/update_pull", json={"reload": True}).get_json()
    assert body["ok"] and "no exec" in body["reload_error"]


def test_update_channel_is_lower_cased(rig):
    client, _calls, _tmp = rig
    assert client.post("/app/update_channel", json={"channel": "Beta"}).get_json() == {"channel": "beta"}


def test_versions_is_served_raw_or_404s(rig):
    client, _calls, tmp = rig
    assert client.get("/versions").status_code == 404
    (tmp / "versions.json").write_text('{"build": 7}')
    res = client.get("/versions")
    assert res.status_code == 200 and res.get_json() == {"build": 7}
