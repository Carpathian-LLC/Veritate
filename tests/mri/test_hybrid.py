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
    """reset_index_cache drops every cached per-scope corpus + index."""
    hybrid_routes._STATE["all"] = {"chunks": ["stale"], "bm25": object()}
    hybrid_routes.reset_index_cache()
    assert hybrid_routes._STATE == {}


def test_kb_upload_rejects_missing_file(client):
    """POST /hybrid/kb/upload with no file returns 400."""
    r = client.post("/hybrid/kb/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_kb_upload_then_bm25_retrieve(client, tmp_path, monkeypatch):
    """An uploaded file is chunked into the corpus and found by BM25 keyword search."""
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "kb"))
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(tmp_path / "noplatform"))
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
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(tmp_path / "noplatform"))
    hybrid_routes.reset_index_cache()
    assert hybrid_routes.retrieve("anything") == ([], [])


def test_platform_kb_grounds_without_uploads(tmp_path, monkeypatch):
    """The shipped platform KB grounds chat even with no user uploads."""
    pdir = tmp_path / "platform"; pdir.mkdir()
    (pdir / "veritate.md").write_text("Veritate uses byte-level tokenization and runs on consumer hardware.")
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(pdir))
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "nokb"))
    hybrid_routes.reset_index_cache()
    assert hybrid_routes.has_corpus() is True
    texts, _scores = hybrid_routes.retrieve("byte level tokenization")
    assert any("byte-level" in t for t in texts)


def test_retrieve_scope_platform_excludes_user_uploads(tmp_path, monkeypatch):
    """scope='platform' retrieves only platform docs, not the user's uploads."""
    pdir = tmp_path / "platform"; pdir.mkdir()
    udir = tmp_path / "user"; udir.mkdir()
    (pdir / "p.md").write_text("Veritate ships nineteen trainers from 10M to 1T parameters.")
    (udir / "u.md").write_text("My private note about trainers and parameters.")
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(pdir))
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(udir))
    hybrid_routes.reset_index_cache()
    texts, _scores = hybrid_routes.retrieve("trainers parameters", scope="platform")
    assert any("nineteen trainers" in t for t in texts)
    assert not any("private note" in t for t in texts)


def test_veritate_docs_pick_grounds_on_platform(client, tmp_path, monkeypatch):
    """The veritate_docs pick routes to the public model and grounds on the platform KB."""
    seen = {}
    def fake_chat(message, system=None, history=None):
        seen.update(system=system)
        return {"ok": True, "answer": "from docs"}
    monkeypatch.setattr("runtime.ai_assist.chat", fake_chat)
    pdir = tmp_path / "platform"; pdir.mkdir()
    (pdir / "v.md").write_text("Veritate trainers range from 10M to 1T parameters.")
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(pdir))
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "nokb"))
    hybrid_routes.reset_index_cache()
    r = client.post("/hybrid/chat", json={"message": "what trainers exist?", "model": "veritate_docs"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["model"] == "Veritate (platform docs)"
    assert d["backend"] == "cloud"
    assert d["confident"] is True
    assert "10M to 1T" in seen["system"]


def test_veritate_docs_includes_training_logs(client, tmp_path, monkeypatch):
    """The veritate_docs mode folds recent Training-tab log lines into the grounding."""
    seen = {}
    def fake_chat(message, system=None, history=None):
        seen.update(system=system)
        return {"ok": True, "answer": "ok"}
    monkeypatch.setattr("runtime.ai_assist.chat", fake_chat)
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(tmp_path / "p"))
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "u"))
    hybrid_routes.reset_index_cache()
    from runtime import logs as logmod
    logmod.info("plugin:veritate_300m", "step 1200 loss 1.83 lr 3e-4")
    r = client.post("/hybrid/chat", json={"message": "how is my run going?", "model": "veritate_docs"})
    assert r.status_code == 200
    d = r.get_json()
    assert any("step 1200 loss 1.83" in src["text"] for src in d["sources"])
    assert "step 1200 loss 1.83" in (seen.get("system") or "")


def test_use_logs_flag_grounds_on_logs(client, tmp_path, monkeypatch):
    """use_logs folds training logs into grounding for any model, not just veritate_docs."""
    seen = {}
    def fake_chat(message, system=None, history=None):
        seen.update(system=system)
        return {"ok": True, "answer": "ok"}
    monkeypatch.setattr("runtime.ai_assist.chat", fake_chat)
    monkeypatch.setattr(hybrid_routes, "PLATFORM_KB_DIR", str(tmp_path / "p"))
    monkeypatch.setattr(hybrid_routes, "KB_DIR", str(tmp_path / "u"))
    hybrid_routes.reset_index_cache()
    from runtime import logs as logmod
    logmod.info("plugin:veritate_80m", "step 900 train_loss 2.01")
    r = client.post("/hybrid/chat", json={"message": "status?", "model": "cloud", "use_logs": True})
    assert r.status_code == 200
    assert "step 900 train_loss 2.01" in (seen.get("system") or "")


def test_context_meter_in_response(client, monkeypatch):
    """Each reply carries a context gauge with a 0..1 fullness pct and the model's char budget."""
    monkeypatch.setattr("runtime.ai_assist.chat",
                        lambda message, system=None, history=None: {"ok": True, "answer": "hi"})
    r = client.post("/hybrid/chat", json={"message": "hello", "model": "cloud"})
    ctx = r.get_json()["context"]
    assert 0.0 <= ctx["pct"] <= 1.0
    assert ctx["char_limit"] > 0 and "turns" in ctx and "chars" in ctx


def test_hybrid_chat_rejects_empty_message(client):
    """POST /hybrid/chat with no message returns 400 before any routing."""
    r = client.post("/hybrid/chat", json={"message": "  "})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_cloud_fallback_calls_public_model(client, monkeypatch):
    """No local model routes to the always-available public Carpathian model."""
    monkeypatch.setattr("runtime.ai_assist.chat",
                        lambda message, system=None, history=None: {"ok": True, "answer": "public says hi"})
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
                        lambda message, system=None, history=None: {"ok": False, "error": "no public model key configured"})
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "cloud"})
    assert r.status_code == 503
    assert r.get_json()["ok"] is False


def test_chat_returns_compacted_memory(client, monkeypatch):
    """A chat turn echoes the conversation memory with the new user+assistant turns appended."""
    monkeypatch.setattr("runtime.ai_assist.chat",
                        lambda message, system=None, history=None: {"ok": True, "answer": "A2"})
    r = client.post("/hybrid/chat", json={"message": "Q2", "model": "cloud",
                                          "history": [{"role": "user", "content": "Q1"},
                                                      {"role": "assistant", "content": "A1"}]})
    assert r.status_code == 200
    mem = r.get_json()["memory"]
    assert mem["turns"][-2:] == [{"role": "user", "content": "Q2"},
                                 {"role": "assistant", "content": "A2"}]


def test_remote_model_grounded_with_rag_and_history(client, monkeypatch):
    """RAG facts and prior turns both reach a remote model: facts in the system, history forwarded."""
    seen = {}
    def fake_chat(message, system=None, history=None):
        seen.update(message=message, system=system, history=history)
        return {"ok": True, "answer": "grounded"}
    monkeypatch.setattr("runtime.ai_assist.chat", fake_chat)
    monkeypatch.setattr(hybrid_routes, "has_corpus", lambda: True)
    monkeypatch.setattr(hybrid_routes, "retrieve", lambda m, k=3, scope="all": (["Veritate is byte-level."], [5.0]))
    r = client.post("/hybrid/chat", json={"message": "what is veritate?", "model": "cloud", "use_rag": True,
                                          "history": [{"role": "user", "content": "hi"},
                                                      {"role": "assistant", "content": "hello"}]})
    assert r.status_code == 200
    assert r.get_json()["confident"] is True
    assert "Veritate is byte-level." in seen["system"]
    assert seen["history"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_compaction_summarizes_over_budget(client, monkeypatch):
    """Past the model's context budget, older turns fold into a summary and only the tail is kept."""
    def fake_chat(message, system=None, history=None):
        if system == hybrid_routes.SUMMARY_SYSTEM:
            return {"ok": True, "answer": "COMPACT SUMMARY"}
        return {"ok": True, "answer": "answer"}
    monkeypatch.setattr("runtime.ai_assist.chat", fake_chat)
    monkeypatch.setattr(hybrid_routes, "context_limit_chars", lambda *a: 20)
    hist = [{"role": ("user" if i % 2 == 0 else "assistant"), "content": f"turn{i}"} for i in range(14)]
    r = client.post("/hybrid/chat", json={"message": "next", "model": "cloud", "history": hist})
    assert r.status_code == 200
    mem = r.get_json()["memory"]
    assert mem["summary"] == "COMPACT SUMMARY"
    assert len(mem["turns"]) == hybrid_routes.CTX_KEEP_TAIL_TURNS


def test_context_limit_is_model_aware():
    """Cloud window is far larger than a local byte model's seq-based budget."""
    cloud = hybrid_routes.context_limit_chars("remote", "carpathian", "")
    local = hybrid_routes.context_limit_chars("local", "", "no_such_model")
    assert cloud == int(hybrid_routes.CONTEXT_TOKENS_BY_PROVIDER["carpathian"]
                        * hybrid_routes.CHARS_PER_TOKEN * hybrid_routes.CONTEXT_MEMORY_FRACTION)
    assert local == max(hybrid_routes.MIN_CONTEXT_CHARS,
                        int(hybrid_routes.DEFAULT_LOCAL_SEQ * hybrid_routes.CONTEXT_MEMORY_FRACTION))
    assert cloud > local


def test_remote_models_lists_cloud_and_api_teacher(client, monkeypatch):
    """/hybrid/models lists the public model plus each configured API teacher's saved model."""
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "openai",
                                 "teacher_configs": {"openai": {"model": "gpt-4o", "api_key": "sk", "base_url": ""}}})
    d = client.get("/hybrid/models").get_json()
    ids = [m["id"] for m in d["models"]]
    assert "cloud" in ids
    assert "teacher:openai:gpt-4o" in ids


def test_remote_models_enumerates_local_submodels(client, monkeypatch):
    """A local provider is enumerated live so every installed sub-model is offered."""
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "ollama",
                                 "teacher_configs": {"ollama": {"model": "", "api_key": "", "base_url": ""}}})
    monkeypatch.setattr("teacher.client.Client.list_models",
                        lambda self: ["qwen2.5:7b-instruct", "llama3:8b"])
    d = client.get("/hybrid/models").get_json()
    ids = [m["id"] for m in d["models"]]
    assert "teacher:ollama:qwen2.5:7b-instruct" in ids
    assert "teacher:ollama:llama3:8b" in ids


def test_teacher_prefix_routes_to_selected_provider(client, monkeypatch):
    """A teacher:<provider>:<model> id routes to that provider/model with its saved key."""
    monkeypatch.delenv("VERITATE_TEACHER_API_KEY", raising=False)
    seen = {}
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "ollama",
                                 "teacher_configs": {"openai": {"model": "gpt-4o-mini", "api_key": "sk-test", "base_url": ""}}})
    def fake_complete(provider, model, messages, **o):
        seen.update(provider=provider, model=model, api_key=o.get("api_key"))
        return "teacher says hi"
    monkeypatch.setattr("teacher.client.complete", fake_complete)
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "teacher:openai:gpt-4o-mini"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["backend"] == "teacher"
    assert d["model"] == "openai: gpt-4o-mini"
    assert seen == {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"}


def test_legacy_teacher_id_routes_to_active_teacher(client, monkeypatch):
    """The bare 'teacher' id still routes to the active provider via its flat key slot."""
    monkeypatch.delenv("VERITATE_TEACHER_API_KEY", raising=False)
    monkeypatch.setattr("runtime.settings.get",
                        lambda: {"teacher_provider": "openai", "teacher_model": "gpt-4o-mini",
                                 "teacher_api_key": "sk-test", "teacher_base_url": "",
                                 "teacher_configs": {}})
    monkeypatch.setattr("teacher.client.complete",
                        lambda provider, model, messages, **o: "teacher says hi")
    r = client.post("/hybrid/chat", json={"message": "hi", "model": "teacher"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["backend"] == "teacher"
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
    monkeypatch.setattr(hybrid_routes, "retrieve", lambda message, k=3, scope="all": (["Paris is the capital."], [4.2]))
    monkeypatch.setattr(hybrid_routes, "_ensure_pytorch", lambda cfg, name: None)
    monkeypatch.setattr(hybrid_routes, "_generate_local", fake_gen)
    r = client.post("/hybrid/chat", json={"message": "capital?", "model": "m1", "use_rag": True})
    assert r.status_code == 200
    d = r.get_json()
    assert d["confident"] is True
    assert d["sources"] == [{"text": "Paris is the capital.", "score": 4.2}]
    assert "context: Paris is the capital." in seen["prompt"]
