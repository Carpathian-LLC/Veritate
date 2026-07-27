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

from flask import Flask, Response
from routes import hybrid_routes as H

# ------------------------------------------------------------------------------------
# Constants

ANSWER          = "Hello there friend"
LOCAL_ANSWER    = "Hello there friend, streamed byte by byte now."
CHATML_IM_END   = "<|im_end|>"

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


def _done_frame(monkeypatch):
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    return _hybrid_frames(r)[-1]


def test_non_stream_object_is_chat_completion(monkeypatch):
    """stream:false returns an object of type chat.completion."""
    assert _post_chat(monkeypatch)["object"] == "chat.completion"


def test_non_stream_carries_assistant_message(monkeypatch):
    """stream:false returns the assistant answer in choices[0].message."""
    body = _post_chat(monkeypatch)
    assert body["choices"][0]["message"] == {"role": "assistant", "content": ANSWER}


def test_non_stream_finish_reason_is_stop(monkeypatch):
    """stream:false reports finish_reason stop."""
    assert _post_chat(monkeypatch)["choices"][0]["finish_reason"] == "stop"


def test_non_stream_usage_totals_add_up(monkeypatch):
    """The usage block's total_tokens equals prompt plus completion tokens."""
    u = _post_chat(monkeypatch)["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_stream_content_type_is_sse(monkeypatch):
    """A buffered stream responds with content type text/event-stream."""
    assert "text/event-stream" in _post_chat_stream(monkeypatch).content_type


def test_stream_first_frame_is_a_chunk(monkeypatch):
    """A buffered stream's first frame is a chat.completion.chunk."""
    assert '"object": "chat.completion.chunk"' in _sse_frames(_post_chat_stream(monkeypatch))[0]


def test_stream_penultimate_frame_finishes_with_stop(monkeypatch):
    """A buffered stream's penultimate frame carries finish_reason stop."""
    assert '"finish_reason": "stop"' in _sse_frames(_post_chat_stream(monkeypatch))[-2]


def test_stream_terminates_with_done_sentinel(monkeypatch):
    """A buffered stream's last frame is the [DONE] sentinel."""
    assert _sse_frames(_post_chat_stream(monkeypatch))[-1] == "data: [DONE]"


def test_stream_content_reassembles(monkeypatch):
    """Concatenating a buffered stream's content deltas reproduces the full answer."""
    assert "".join(_content_deltas(_post_chat_stream(monkeypatch))) == ANSWER


def test_local_delta_stream_is_incremental():
    """The local token generator emits more than one content delta for one answer."""
    assert len(list(H._local_delta_stream(_events(LOCAL_ANSWER)))) > 1


def test_local_delta_stream_reassembles_answer():
    """The local token generator's deltas concatenate back to the full answer."""
    assert "".join(H._local_delta_stream(_events(LOCAL_ANSWER))) == LOCAL_ANSWER


def test_local_delta_stream_emits_no_delta_containing_stop_marker():
    """No content delta of a stream ending in <|im_end|> contains the marker."""
    deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER + CHATML_IM_END)))
    assert [d for d in deltas if CHATML_IM_END in d] == []


def test_local_delta_stream_cuts_trailing_stop_marker():
    """A trailing <|im_end|> is cut from the reassembled content, leaving the answer."""
    deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER + CHATML_IM_END)))
    assert "".join(deltas) == LOCAL_ANSWER


def test_local_delta_stream_never_leaks_partial_stop_marker():
    """A stream ending in <|im_end|> emits no content delta holding a marker prefix at the tail."""
    deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER + CHATML_IM_END)))
    text = "".join(deltas)
    assert not any(text.endswith(CHATML_IM_END[:i]) for i in range(1, len(CHATML_IM_END) + 1))


def test_local_delta_stream_cuts_every_chatml_stop_marker():
    """Each marker in STOP_MARKERS terminating the byte stream is cut from the content."""
    for marker in H.STOP_MARKERS:
        deltas = list(H._local_delta_stream(_events(LOCAL_ANSWER + marker)))
        assert "".join(deltas) == LOCAL_ANSWER, marker


def test_stream_local_dispatches_true_token_path(monkeypatch):
    """A local-kind streaming request is dispatched to the true per-token handler, not the buffered one."""
    monkeypatch.setattr(H, "_openai_stream_local",
                        lambda cfg, model, backend, conv, system, mri=False, gen_params=None: Response(
                            "TRUE_TOKEN", mimetype="text/event-stream"))
    r = _client(monkeypatch, kind="local").post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.get_data(as_text=True) == "TRUE_TOKEN"


def test_bad_body_is_400(monkeypatch):
    """A request with no messages array is a 400 invalid_request_error."""
    assert _client(monkeypatch).post("/v1/chat/completions", json={"model": "m"}).status_code == 400


def test_bad_body_error_type_is_invalid_request(monkeypatch):
    """A request with no messages array reports error.type invalid_request_error."""
    r = _client(monkeypatch).post("/v1/chat/completions", json={"model": "m"})
    assert r.get_json()["error"]["type"] == "invalid_request_error"


def test_unavailable_provider_is_503(monkeypatch):
    """A ChatUnavailable from the wrapped route maps to a 503."""
    r = _client(monkeypatch, raises=H.ChatUnavailable("no key")).post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def _post_chat(monkeypatch):
    """POST a non-stream /v1/chat/completions and return the parsed body."""
    return _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}).get_json()


def _post_chat_stream(monkeypatch):
    return _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})


def _sse_frames(resp):
    return [f for f in resp.get_data(as_text=True).split("\n\n") if f.strip()]


def _hybrid_frames(resp):
    """Parse a /hybrid/chat/stream SSE body into its event dicts."""
    out = []
    for f in resp.get_data(as_text=True).split("\n\n"):
        f = f.strip()
        if f.startswith("data: "):
            out.append(json.loads(f[len("data: "):]))
    return out


def test_hybrid_stream_content_type_is_sse(monkeypatch):
    """/hybrid/chat/stream responds with content type text/event-stream."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert "text/event-stream" in r.content_type


def test_hybrid_stream_deltas_reassemble(monkeypatch):
    """/hybrid/chat/stream delta frames concatenate back to the full answer."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert "".join(f["text"] for f in _hybrid_frames(r) if f["kind"] == "delta") == ANSWER


def test_hybrid_stream_terminal_frame_is_done(monkeypatch):
    """/hybrid/chat/stream's last frame is of kind done."""
    assert _done_frame(monkeypatch)["kind"] == "done"


def test_hybrid_done_frame_carries_answer(monkeypatch):
    """The done frame carries the full answer."""
    assert _done_frame(monkeypatch)["answer"] == ANSWER


def test_hybrid_done_frame_carries_both_turns(monkeypatch):
    """The done frame's memory carries the user turn then the assistant turn."""
    assert [t["role"] for t in _done_frame(monkeypatch)["memory"]["turns"]] == ["user", "assistant"]


def test_hybrid_done_frame_carries_context_gauge(monkeypatch):
    """The done frame's context gauge counts the two new turns."""
    assert _done_frame(monkeypatch)["context"]["turns"] == 2


def test_hybrid_done_frame_carries_sources(monkeypatch):
    """The done frame carries a sources key."""
    assert "sources" in _done_frame(monkeypatch)


def test_hybrid_stream_empty_message_is_400(monkeypatch):
    """An empty chat message is rejected before the stream opens."""
    r = _client(monkeypatch).post("/hybrid/chat/stream", json={"model": "m", "message": "  "})
    assert r.status_code == 400


def test_hybrid_stream_unavailable_emits_error_frame(monkeypatch):
    """A ChatUnavailable after the stream opened surfaces as an error frame, not a 503."""
    r = _client(monkeypatch, raises=H.ChatUnavailable("no key")).post(
        "/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert r.status_code == 200


def test_hybrid_stream_error_frame_carries_reason(monkeypatch):
    """A ChatUnavailable after the stream opened emits an error frame naming the reason."""
    r = _client(monkeypatch, raises=H.ChatUnavailable("no key")).post(
        "/hybrid/chat/stream", json={"model": "m", "message": "hi"})
    assert [f["error"] for f in _hybrid_frames(r) if f["kind"] == "error"] == ["no key"]


def test_hybrid_stream_local_honors_c_engine(monkeypatch):
    """A local stream with backend='c' decodes via _ensure_c (the selected engine), not pytorch."""
    from routes import backends_routes as B
    calls = []
    monkeypatch.setattr(H, "_ensure_c", lambda cfg, name: calls.append("c"))
    monkeypatch.setattr(H, "_ensure_pytorch", lambda cfg, name: calls.append("pytorch"))
    monkeypatch.setattr(H, "_local_events",
                        lambda cfg, backend, prompt, mri=False: (_events(LOCAL_ANSWER), []))
    monkeypatch.setattr(B, "_stop_on_bytes", lambda events, stop: events)
    _client(monkeypatch, kind="local").post(
        "/hybrid/chat/stream", json={"model": "m", "backend": "c", "message": "hi"})
    assert calls == ["c"]
