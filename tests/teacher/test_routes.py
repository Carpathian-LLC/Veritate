# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests for the teacher HTTP routes. builds a minimal Flask app and registers
#   teacher_routes onto it; isolates the settings file under tmp_path so each
#   test starts from a clean state. mocks Client.complete and SynthJob.run to
#   keep tests offline and deterministic.
# tests/teacher/test_routes.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os
import sys
import time

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MRI_DIR = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask

# ------------------------------------------------------------------------------------
# Constants

TEACHER_KEY_ENV = "VERITATE_TEACHER_API_KEY"
SECRET_API_KEY = "sk-test-do-not-leak-XYZ123"
DEFAULT_TIMEOUT_S = 5

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Repoint settings storage to tmp_path and clear the module cache so each
    test starts with DEFAULTS."""
    from runtime import settings as settings_mod
    settings_path = str(tmp_path / "mri_settings.json")
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    monkeypatch.delenv(TEACHER_KEY_ENV, raising=False)
    yield settings_mod
    monkeypatch.setattr(settings_mod, "_CACHE", None)


@pytest.fixture
def teacher_pkgs(monkeypatch):
    """Force `teacher` to resolve to veritate_mri/teacher (not tests/teacher)
    when teacher_routes imports it. pytest's package collection binds the
    name to tests/teacher by default; we override here."""
    import importlib
    for name in list(sys.modules):
        if name == "teacher" or name.startswith("teacher."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    t = importlib.import_module("veritate_mri.teacher")
    s = importlib.import_module("veritate_mri.teacher.synth")
    tc = importlib.import_module("veritate_mri.teacher.test_connection")
    sys.modules["teacher"] = t
    sys.modules["teacher.synth"] = s
    sys.modules["teacher.test_connection"] = tc
    sys.modules["teacher.providers"] = importlib.import_module("veritate_mri.teacher.providers")
    sys.modules["teacher.client"] = importlib.import_module("veritate_mri.teacher.client")
    sys.modules["teacher.quality"] = importlib.import_module("veritate_mri.teacher.quality")
    yield {"teacher": t, "synth": s, "test_connection": tc}


@pytest.fixture
def client(isolated_settings, teacher_pkgs):
    """Flask test client with teacher_routes mounted on a fresh app."""
    from routes import teacher_routes
    app = Flask(__name__)
    app.config["TESTING"] = True
    teacher_routes.register(app)
    with app.test_client() as c:
        yield c


def test_get_returns_providers_and_no_key(client):
    """GET /teacher returns provider list and has_api_key=False with env cleared."""
    r = client.get("/teacher")
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body["providers"], list)
    assert body["has_api_key"] is False


def test_get_has_api_key_true_when_env_set(client, monkeypatch):
    """env var VERITATE_TEACHER_API_KEY makes has_api_key true."""
    monkeypatch.setenv(TEACHER_KEY_ENV, SECRET_API_KEY)
    r = client.get("/teacher")
    body = r.get_json()
    assert body["has_api_key"] is True


def test_post_persists_provider(client):
    """POST /teacher with teacher_provider=ollama reflects on subsequent GET."""
    r = client.post("/teacher", json={"teacher_provider": "ollama"})
    assert r.status_code == 200
    assert r.get_json()["provider"] == "ollama"
    r2 = client.get("/teacher")
    assert r2.get_json()["provider"] == "ollama"


def test_provider_switch_restores_saved_key(client, isolated_settings):
    """switching provider away and back restores its stored key via teacher_configs."""
    client.post("/teacher", json={"teacher_provider": "openai",
                                  "teacher_model": "gpt-4o",
                                  "teacher_api_key": SECRET_API_KEY})
    r = client.post("/teacher", json={"teacher_provider": "ollama",
                                      "teacher_model": "qwen2.5:72b"})
    body = r.get_json()
    assert body["has_api_key"] is False
    assert body["configs"]["openai"] == {"model": "gpt-4o", "base_url": "", "has_key": True}
    r2 = client.post("/teacher", json={"teacher_provider": "openai",
                                       "teacher_model": "gpt-4o"})
    assert r2.get_json()["has_api_key"] is True
    assert isolated_settings.get()["teacher_api_key"] == SECRET_API_KEY


def test_post_invalid_concurrency_returns_400(client):
    """POST /teacher with out-of-range concurrency returns 400 + error message."""
    r = client.post("/teacher", json={"teacher_max_concurrency": 9999})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_local_concurrency_capped(teacher_pkgs):
    """local providers clamp a high global concurrency to the safe ceiling; cloud keeps it."""
    from routes import teacher_routes
    from teacher import providers
    high = {"teacher_max_concurrency": 64}
    assert teacher_routes._resolve_concurrency(high, "ollama") == providers.LOCAL_MAX_CONCURRENCY
    assert teacher_routes._resolve_concurrency(high, "openai") == 64
    assert teacher_routes._resolve_concurrency({"teacher_max_concurrency": 2}, "ollama") == 2


def test_synth_delete_removes_dir(client, teacher_pkgs, monkeypatch, tmp_path):
    """POST /teacher/synth/delete removes an existing job dir; unknown and traversal ids 404."""
    from routes import teacher_routes
    monkeypatch.setattr(teacher_routes, "REPO_ROOT", str(tmp_path))
    job_dir = tmp_path / teacher_routes.SYNTH_JOBS_DIR / "abc123"
    job_dir.mkdir(parents=True)
    (job_dir / "samples.jsonl").write_text("{}\n")
    r = client.post("/teacher/synth/delete", json={"job_id": "abc123"})
    assert r.status_code == 200
    assert not job_dir.exists()
    assert client.post("/teacher/synth/delete", json={"job_id": "nope"}).status_code == 404
    assert client.post("/teacher/synth/delete", json={"job_id": "../mri_settings.json"}).status_code == 404


def test_test_unknown_provider_returns_ok_false(client, teacher_pkgs, monkeypatch):
    """POST /teacher/test for an unreachable provider returns ok=False."""
    def fake_test(provider_id, model=None, base_url=None, api_key=None):
        return {"ok": False, "latency_ms": 1, "error": "auth: bad", "model": model or ""}

    monkeypatch.setattr(teacher_pkgs["test_connection"], "test", fake_test)
    r = client.post("/teacher/test", json={"provider": "openai", "model": "gpt-4o"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is False


def test_synth_start_empty_prompts_returns_400(client):
    """POST /teacher/synth/start with empty prompts returns 400."""
    client.post("/teacher", json={"teacher_provider": "ollama", "teacher_model": "x"})
    r = client.post("/teacher/synth/start", json={"prompts": []})
    assert r.status_code == 400


def test_synth_status_unknown_job_returns_404(client):
    """GET /teacher/synth/status with unknown job_id returns 404."""
    r = client.get("/teacher/synth/status?job_id=nope")
    assert r.status_code == 404
    assert r.get_json()["error"] == "unknown job"


def test_synth_start_then_status(client, teacher_pkgs, monkeypatch, tmp_path):
    """synth/start launches a job; status returns running/completed counts."""
    captured = {}

    class FakeJob:
        def __init__(self, *a, **kw):
            captured["args"] = a
            captured["kwargs"] = kw
            self.output_dir = a[4]

        def run(self):
            samples = os.path.join(self.output_dir, "samples.jsonl")
            with open(samples, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p0", "response": "ok"}) + "\n")

    monkeypatch.setattr(teacher_pkgs["synth"], "SynthJob", FakeJob)
    client.post("/teacher", json={"teacher_provider": "ollama", "teacher_model": "x"})
    out_dir = str(tmp_path / "job")
    r = client.post("/teacher/synth/start", json={
        "prompts": [{"id": "p0", "messages": [{"role": "user", "content": "hi"}]}],
        "output_dir": out_dir,
    })
    assert r.status_code == 200
    job_id = r.get_json()["job_id"]
    deadline = time.time() + DEFAULT_TIMEOUT_S
    while time.time() < deadline:
        s = client.get(f"/teacher/synth/status?job_id={job_id}")
        if not s.get_json()["running"]:
            break
    s = client.get(f"/teacher/synth/status?job_id={job_id}")
    body = s.get_json()
    assert body["job_id"] == job_id
    assert body["completed"] == 1


def test_seeds_list_returns_catalog(client):
    """GET /teacher/seeds returns version 1 and a non-empty seeds list."""
    r = client.get("/teacher/seeds")
    assert r.status_code == 200
    body = r.get_json()
    assert body["version"] == 1
    assert isinstance(body["seeds"], list) and len(body["seeds"]) > 0


def test_seeds_detail_returns_prompts(client):
    """GET /teacher/seeds/math_word_problems returns prompt list with the right count."""
    r = client.get("/teacher/seeds/math_word_problems")
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == len(body["prompts"]) > 0


def test_seeds_catalog_counts_match_files(client):
    """every catalog entry's jsonl file exists and its line count equals the entry's count."""
    from routes.teacher_routes import SEEDS_DIR
    seeds = client.get("/teacher/seeds").get_json()["seeds"]
    for s in seeds:
        with open(os.path.join(SEEDS_DIR, s["file"]), encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        assert n == s["count"], s["id"]


def test_seeds_grouped_entries_carry_tier(client):
    """every grouped catalog entry has a tier in {easy, basic, advanced}."""
    seeds = client.get("/teacher/seeds").get_json()["seeds"]
    grouped = [s for s in seeds if s.get("group")]
    assert grouped
    assert all(s.get("tier") in {"easy", "basic", "advanced"} for s in grouped)


def test_seeds_unknown_id_returns_404(client):
    """GET /teacher/seeds/<unknown> returns 404."""
    r = client.get("/teacher/seeds/unknown_id")
    assert r.status_code == 404


def test_api_key_value_never_in_response(client, monkeypatch):
    """api key value is absent from every endpoint's response body and keys."""
    monkeypatch.setenv(TEACHER_KEY_ENV, SECRET_API_KEY)
    client.post("/teacher", json={"teacher_api_key": SECRET_API_KEY,
                                  "teacher_provider": "openai",
                                  "teacher_model": "gpt-4o"})
    r = client.get("/teacher")
    body = r.get_json()
    assert "teacher_api_key" not in body
    assert SECRET_API_KEY not in r.get_data(as_text=True)
    r2 = client.post("/teacher", json={"teacher_model": "gpt-4o-mini"})
    assert SECRET_API_KEY not in r2.get_data(as_text=True)
    assert "teacher_api_key" not in r2.get_json()
