# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the front-door chat endpoint: prompt formats, the
#   stream-collect helper, cloud-fallback routing, opt-in BM25 retrieval, and the
#   knowledge-base upload. Model load + real generation need torch and are
#   stubbed; the teacher call is never hit (mocked per preflight rule 48).
# tests/mri/test_hybrid.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import os
import sys

import pytest

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
for p in (REPO_ROOT, MRI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from routes import hybrid_routes


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
def clear_kb_cache():
    """Drop the cached corpus/index around every test so they stay isolated."""
    hybrid_routes.reset_index_cache()
    yield
    hybrid_routes.reset_index_cache()


def test_build_prompt_grounds_context_and_message():
    """build_prompt places the facts in a context block ahead of the user turn."""
    p = hybrid_routes.build_prompt("What is the capital?", ["France's capital is Paris."])
    assert "context: France's capital is Paris." in p
    assert "<|user|>\nWhat is the capital?" in p
    assert p.rstrip().endswith("<|assistant|>")


def test_build_plain_prompt_has_no_context_block():
    """build_plain_prompt is a bare user turn with no context line."""
    p = hybrid_routes.build_plain_prompt("hello")
    assert "context:" not in p
    assert p.startswith("<|user|>\nhello")
    assert p.rstrip().endswith("<|assistant|>")


def test_collect_joins_token_bytes_and_stops():
    """collect accumulates token/fast_byte bytes and halts on a stop event."""
    events = [
        {"kind": "meta"},
        {"kind": "token", "byte": ord("h")},
        {"kind": "fast_byte", "byte": ord("i")},
        {"kind": "stop"},
        {"kind": "token", "byte": ord("X")},
    ]
    assert hybrid_routes.collect(events) == "hi"


def test_reset_index_cache_clears_state():
    """reset_index_cache drops the cached corpus + index so the next load reloads."""
    hybrid_routes._STATE["chunks"] = ["stale"]
    hybrid_routes._STATE["bm25"] = object()
    hybrid_routes.reset_index_cache()
    assert hybrid_routes._STATE["chunks"] is None
    assert hybrid_routes._STATE["bm25"] is None


def test_kb_upload_rejects_missing_file(client):
    """POST /hybrid/kb/upload with no file returns 400."""
    r = client.post("/hybrid/kb/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_kb_upload_then_bm25_retrieve(client, tmp_path, monkeypatch):
    """An uploaded file is chunked into the corpus and found by BM25 keyword search."""
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "kb"))
    hybrid_routes.reset_index_cache()
    data = {"file": (io.BytesIO(b"The capital of France is Paris. "
                                b"Berlin is the capital of Germany."), "geo.txt")}
    r = client.post("/hybrid/kb/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True and j["n_files"] == 1 and j["n_chunks"] >= 1
    texts, scores = hybrid_routes.retrieve("france capital", k=2)
    assert any("Paris" in t for t in texts)


def test_retrieve_empty_corpus_returns_nothing(tmp_path, monkeypatch):
    """retrieve over an empty knowledge base returns no hits, no error."""
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "empty"))
    hybrid_routes.reset_index_cache()
    assert hybrid_routes.retrieve("anything") == ([], [])


def test_hybrid_chat_rejects_empty_message(client):
    """POST /hybrid/chat with no message returns 400 before any routing."""
    r = client.post("/hybrid/chat", json={"message": "  "})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_cloud_fallback_calls_public_model(client, monkeypatch):
    """No local model routes to the always-available public Carpathian model."""
    monkeypatch.setattr("runtime.ai_assist.chat",
                        lambda message, system=None: {"ok": True, "answer": "public says hi"})
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "cloud"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["answer"] == "public says hi"
    assert d["backend"] == "cloud"
    assert d["model"] == hybrid_routes.CLOUD_LABEL


def test_cloud_fallback_unconfigured_returns_503(client, monkeypatch):
    """Public model with no key configured returns 503, not a crash."""
    monkeypatch.setattr("runtime.ai_assist.chat",
                        lambda message, system=None: {"ok": False, "error": "no public model key configured"})
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "cloud"})
    assert r.status_code == 503
    assert r.get_json()["ok"] is False


def test_health_reports_configured_teacher(client, monkeypatch):
    """/hybrid/health surfaces the configured teacher label for the dropdown."""
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "openai", "teacher_model": "gpt-4o"})
    d = client.get("/hybrid/health").get_json()
    assert d["teacher"] == {"configured": True, "label": "openai: gpt-4o"}


def test_teacher_model_routes_to_teacher(client, monkeypatch):
    """Selecting the configured teacher routes to its provider/model and key."""
    monkeypatch.delenv("VERITATE_TEACHER_API_KEY", raising=False)
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "openai", "teacher_model": "gpt-4o-mini",
                                 "teacher_api_key": "sk-test", "teacher_base_url": ""})
    monkeypatch.setattr("teacher.client.complete",
                        lambda provider, model, messages, **o: "teacher says hi")
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "teacher"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["backend"] == "teacher"
    assert d["answer"] == "teacher says hi"
    assert d["model"] == "openai: gpt-4o-mini"


def test_teacher_unconfigured_returns_503(client, monkeypatch):
    """Selecting teacher with nothing configured returns 503, not the public model."""
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "", "teacher_model": ""})
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "teacher"})
    assert r.status_code == 503


def test_local_no_rag_builds_plain_prompt(client, monkeypatch):
    """use_rag=false sends a plain prompt and never calls retrieval."""
    seen = {}
    def fake_gen(cfg, backend, prompt):
        seen["prompt"] = prompt
        return "ANSWER"
    monkeypatch.setattr(hybrid_routes, "is_local_model", lambda name: True)
    monkeypatch.setattr(hybrid_routes, "_ensure_pytorch", lambda cfg, name: None)
    monkeypatch.setattr(hybrid_routes, "retrieve",
                        lambda *a, **k: pytest.fail("retrieval must not run when use_rag is false"))
    monkeypatch.setattr(hybrid_routes, "_generate_local", fake_gen)
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "m1",
                                          "backend": "pytorch", "use_rag": False})
    assert r.status_code == 200
    d = r.get_json()
    assert d["answer"] == "ANSWER"
    assert d["confident"] is False
    assert d["sources"] == []
    assert seen["prompt"] == hybrid_routes.build_plain_prompt("hi")


def test_local_with_rag_builds_context_prompt(client, monkeypatch):
    """use_rag=true with a confident hit injects the retrieved fact as context."""
    seen = {}
    def fake_gen(cfg, backend, prompt):
        seen["prompt"] = prompt
        return "Paris"
    monkeypatch.setattr(hybrid_routes, "is_local_model", lambda name: True)
    monkeypatch.setattr(hybrid_routes, "has_corpus", lambda: True)
    monkeypatch.setattr(hybrid_routes, "retrieve", lambda message, k=3: (["Paris is the capital."], [4.2]))
    monkeypatch.setattr(hybrid_routes, "_ensure_pytorch", lambda cfg, name: None)
    monkeypatch.setattr(hybrid_routes, "_generate_local", fake_gen)
    r = client.post("/hybrid/chat", json={"message": "capital?", "model": "m1", "use_rag": True})
    assert r.status_code == 200
    d = r.get_json()
    assert d["confident"] is True
    assert d["sources"] == [{"text": "Paris is the capital.", "score": 4.2}]
    assert "context: Paris is the capital." in seen["prompt"]
