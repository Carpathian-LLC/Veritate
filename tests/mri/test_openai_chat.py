# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for POST /v1/chat/completions (OpenAI-compatible). _resolve_route is
#   stubbed so no model loads (rule 33); assertions cover the non-stream envelope, the
#   buffered cloud/teacher SSE stream, the local true per-token generator (stubbed
#   byte-event stream, no torch), stop-marker holdback, stream dispatch, and body
#   validation.
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


def _deltas(events):
    """Sanitized per-token text deltas: the text-only view of _local_stream_items,
    as the /v1 local streaming path consumes it."""
    return [v for tag, v in H._local_stream_items(events) if tag == "text"]


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
    assert len(_deltas(_events(LOCAL_ANSWER))) > 1


def test_local_delta_stream_reassembles_answer():
    """The local token generator's deltas concatenate back to the full answer."""
    assert "".join(_deltas(_events(LOCAL_ANSWER))) == LOCAL_ANSWER


def test_local_delta_stream_emits_no_delta_containing_stop_marker():
    """No content delta of a stream ending in <|im_end|> contains the marker."""
    deltas = _deltas(_events(LOCAL_ANSWER + CHATML_IM_END))
    assert [d for d in deltas if CHATML_IM_END in d] == []


def test_local_delta_stream_cuts_trailing_stop_marker():
    """A trailing <|im_end|> is cut from the reassembled content, leaving the answer."""
    assert "".join(_deltas(_events(LOCAL_ANSWER + CHATML_IM_END))) == LOCAL_ANSWER


def test_local_delta_stream_never_leaks_partial_stop_marker():
    """A stream ending in <|im_end|> emits no content delta holding a marker prefix at the tail."""
    text = "".join(_deltas(_events(LOCAL_ANSWER + CHATML_IM_END)))
    assert not any(text.endswith(CHATML_IM_END[:i]) for i in range(1, len(CHATML_IM_END) + 1))


def test_local_delta_stream_cuts_every_chatml_stop_marker():
    """Each marker in STOP_MARKERS terminating the byte stream is cut from the content."""
    for marker in H.STOP_MARKERS:
        assert "".join(_deltas(_events(LOCAL_ANSWER + marker))) == LOCAL_ANSWER, marker


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


def _real_route_client(monkeypatch, cloud_answer=None):
    """App with the REAL _resolve_route: no local models exist, and the public model is
    a stub that records whether it was reached."""
    calls = []
    monkeypatch.setattr(H, "is_local_model", lambda name: False)
    from runtime import ai_assist

    def fake_chat(text, system=None, history=None):
        calls.append(text)
        return {"ok": True, "answer": cloud_answer}
    monkeypatch.setattr(ai_assist, "chat", fake_chat)
    app = Flask(__name__)
    H.register(app)
    return app.test_client(), calls


def test_an_unknown_model_name_is_404_not_the_public_model(monkeypatch):
    """A model that is neither `cloud`, a teacher id, nor local is refused. Before this
    the route fell through to the public endpoint, and 50 fact statements meant for an
    empty local model dir were answered off-box on 2026-09-02."""
    client, calls = _real_route_client(monkeypatch)
    r = client.post("/v1/chat/completions",
                    json={"model": "no_such_model", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert r.get_json()["error"]["code"] == "model_not_found"
    assert calls == []


def test_the_mri_endpoint_refuses_an_unknown_model_the_same_way(monkeypatch):
    """Both OpenAI-shaped endpoints share the routing, so both refuse."""
    client, calls = _real_route_client(monkeypatch)
    r = client.post("/v1/chat/mri",
                    json={"model": "no_such_model", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404 and calls == []


def test_cloud_is_still_reachable_by_its_own_name(monkeypatch):
    """The public model is a deliberate choice, selected by name, never a fallback."""
    client, calls = _real_route_client(monkeypatch, cloud_answer="from the cloud")
    r = client.post("/v1/chat/completions",
                    json={"model": "cloud", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.get_json()["choices"][0]["message"]["content"] == "from the cloud"
    assert calls == ["hi"]


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
