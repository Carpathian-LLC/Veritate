# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for POST /v1/chat/completions (OpenAI-compatible) and the chat-page
#   POST /hybrid/chat/stream SSE. _resolve_route is stubbed so no model loads (rule 33);
#   assertions cover the non-stream envelope, the buffered cloud/teacher SSE stream, the
#   local true per-token generator (stubbed byte-event stream, no torch), stop-marker
#   holdback, stream dispatch, body validation, the /hybrid/chat/stream delta+done frames,
#   its done-frame memory/context/sources, and that a local stream honors the selected
#   c engine.
# tests/mri/test_openai_chat.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from flask import Flask, Response

from routes import hybrid_routes as H

# ------------------------------------------------------------------------------------
# Constants

ANSWER       = "Hello there friend"
LOCAL_ANSWER = "Hello there friend, streamed byte by byte now."

# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, answer=ANSWER, raises=None, kind="remote"):
    """App with _resolve_route stubbed to a buffered complete() (cloud/teacher kind)."""
    def fake_resolve(cfg, model, backend):
        def complete(messages, system):
            if raises is not None:
                raise raises
            return answer
        return complete, "stub", "cloud", kind, 4096
    monkeypatch.setattr(H, "_resolve_route", fake_resolve)
    app = Flask(__name__)
    try:
        app.json.sort_keys = False
    except AttributeError:
        app.config["JSON_SORT_KEYS"] = False
    H.register(app)
    return app.test_client()


def _events(text):
    """Stubbed local generation stream as _stop_on_bytes hands it to the delta
    emitter: a meta frame, one token frame per UTF-8 byte, then a stop frame."""
    yield {"kind": "meta"}
    for b in text.encode("utf-8"):
        yield {"kind": "token", "byte": b}
    yield {"kind": "stop"}


def _content_deltas(resp):
    out = []
    for f in resp.get_data(as_text=True).split("\n\n"):
        f = f.strip()
        if not f.startswith("data: ") or f == "data: [DONE]":
            continue
        delta = json.loads(f[len("data: "):])["choices"][0]["delta"]
        if "content" in delta:
            out.append(delta["content"])
    return out


def test_non_stream_envelope(monkeypatch):
    """stream:false returns an OpenAI chat.completion with the assistant answer."""
    r = _client(monkeypatch).post("/v1/chat/completions",
                                  json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": ANSWER}
    assert body["choices"][0]["finish_reason"] == "stop"


def test_non_stream_usage_present(monkeypatch):
    """The usage block carries approximate prompt/completion/total token counts."""
    r = _client(monkeypatch).post("/v1/chat/completions",
                                  json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    u = r.get_json()["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_stream_is_sse_with_done(monkeypatch):
    """A buffered (cloud) stream returns text/event-stream ending with a stop chunk then [DONE]."""
    r = _client(monkeypatch).post("/v1/chat/completions",
                                  json={"model": "m", "stream": True,
                                        "messages": [{"role": "user", "content": "hi"}]})
    assert "text/event-stream" in r.content_type
    frames = [f for f in r.get_data(as_text=True).split("\n\n") if f.strip()]
    assert '"object": "chat.completion.chunk"' in frames[0]
    assert '"finish_reason": "stop"' in frames[-2]
    assert frames[-1] == "data: [DONE]"


def test_stream_content_reassembles(monkeypatch):
    """Concatenating a buffered stream's content deltas reproduces the full answer."""
    r = _client(monkeypatch).post("/v1/chat/completions",
                                  json={"model": "m", "stream": True,
                                        "messages": [{"role": "user", "content": "hi"}]})
    assert "".join(_content_deltas(r)) == ANSWER


def test_local_delta_stream_is_incremental():
    """The local token generator emits multiple content deltas as bytes arrive, reassembling to the answer."""
    deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER)))
    assert len(deltas) > 1
    assert "".join(deltas) == LOCAL_ANSWER


def test_local_delta_stream_holds_back_stop_marker():
    """A turn marker ending the byte stream is cut, never emitted in the content deltas."""
    deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER + "<|user|>")))
    text = "".join(deltas)
    assert "<|user|>" not in text
    assert text == LOCAL_ANSWER


def test_stream_local_dispatches_true_token_path(monkeypatch):
    """A local-kind streaming request is dispatched to the true per-token handler, not the buffered one."""
    monkeypatch.setattr(H, "_openai_stream_local",
                        lambda cfg, model, backend, conv, system: Response("TRUE_TOKEN",
                                                                           mimetype="text/event-stream"))
    r = _client(monkeypatch, kind="local").post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.get_data(as_text=True) == "TRUE_TOKEN"


def test_bad_body_is_400(monkeypatch):
    """A request with no messages array is a 400 invalid_request_error."""
    r = _client(monkeypatch).post("/v1/chat/completions", json={"model": "m"})
    assert r.status_code == 400
    assert r.get_json()["error"]["type"] == "invalid_request_error"


def test_unavailable_provider_is_503(monkeypatch):
    """A ChatUnavailable from the wrapped route maps to a 503."""
    r = _client(monkeypatch, raises=H.ChatUnavailable("no key")).post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def _hybrid_frames(resp):
    """Parse a /hybrid/chat/stream SSE body into its event dicts."""
    out = []
    for f in resp.get_data(as_text=True).split("\n\n"):
        f = f.strip()
        if f.startswith("data: "):
            out.append(json.loads(f[len("data: "):]))
    return out


def test_hybrid_stream_is_sse_and_reassembles(monkeypatch):
    """/hybrid/chat/stream is text/event-stream whose delta frames reassemble the answer."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert "text/event-stream" in r.content_type
    frames = _hybrid_frames(r)
    deltas = "".join(f["text"] for f in frames if f["kind"] == "delta")
    assert deltas == ANSWER


def test_hybrid_stream_done_carries_memory_and_context(monkeypatch):
    """The terminal done frame carries the full answer, the new turns, sources, and the context gauge."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    done = _hybrid_frames(r)[-1]
    assert done["kind"] == "done"
    assert done["answer"] == ANSWER
    assert [t["role"] for t in done["memory"]["turns"]] == ["user", "assistant"]
    assert done["context"]["turns"] == 2
    assert "sources" in done


def test_hybrid_stream_empty_message_is_400(monkeypatch):
    """An empty chat message is rejected before the stream opens."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "  "})
    assert r.status_code == 400


def test_hybrid_stream_unavailable_emits_error_frame(monkeypatch):
    """A ChatUnavailable after the stream opened surfaces as an error frame, not a 503."""
    r = _client(monkeypatch, raises=H.ChatUnavailable("no key")).post(
        "/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert r.status_code == 200
    errs = [f for f in _hybrid_frames(r) if f["kind"] == "error"]
    assert errs and errs[0]["error"] == "no key"


def test_hybrid_stream_local_honors_c_engine(monkeypatch):
    """A local stream with backend='c' decodes via _ensure_c (the selected engine), not pytorch."""
    from routes import backends_routes as B
    calls = []
    monkeypatch.setattr(H, "_ensure_c", lambda cfg, name: calls.append("c"))
    monkeypatch.setattr(H, "_ensure_pytorch", lambda cfg, name: calls.append("pytorch"))
    monkeypatch.setattr(H, "_local_events", lambda cfg, backend, prompt: (_events(LOCAL_ANSWER), []))
    monkeypatch.setattr(B, "_stop_on_bytes", lambda events, stop: events)
    r = _client(monkeypatch, kind="local").post(
        "/hybrid/chat/stream", json={"model": "m", "backend": "c", "message": "hi"})
    frames = _hybrid_frames(r)
    assert calls == ["c"]
    assert "".join(f["text"] for f in frames if f["kind"] == "delta") == LOCAL_ANSWER
    assert frames[-1]["kind"] == "done" and frames[-1]["answer"] == LOCAL_ANSWER
