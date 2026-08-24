# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route + unit tests for the opt-in per-byte MRI telemetry field on POST
#   /v1/chat/completions. Proves: flag off is today's
#   text-only behavior (no frames), flag on returns the same per-byte frame objects the
#   MRI view consumes (unfiltered), and an off-the-shelf OpenAI reader still reassembles
#   the answer with the flag on. _resolve_route / _local_events / _generate_local_mri are
#   stubbed so no model loads (rule 33); deterministic, CPU, no network.
# tests/mri/test_mri_optin.py
# ------------------------------------------------------------------------------------
# Imports:

import json

from flask import Flask
from routes import hybrid_routes as H

# ------------------------------------------------------------------------------------
# Constants

ANSWER = "hello world"
FRAMES = [
    {"kind": "meta", "checkpoint": "stub", "layers": 2},
    {"kind": "token", "byte": ord("h"), "ffn_full": [[7]], "attn": [], "lens": [],
     "cand": [], "confidence": 0.5, "entropy_bits": 1.25, "backend": "pytorch"},
    {"kind": "stop", "reason": "<|end|>"},
]

# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, kind="local", resp_backend="pytorch"):
    """App with _resolve_route stubbed (no model load). complete() returns ANSWER."""
    def fake_resolve(cfg, model, backend):
        def complete(messages, system):
            return ANSWER
        return complete, "stub", resp_backend, kind, 4096
    monkeypatch.setattr(H, "_resolve_route", fake_resolve)
    app = Flask(__name__)
    try:
        app.json.sort_keys = False
    except AttributeError:
        app.config["JSON_SORT_KEYS"] = False
    H.register(app)
    return app.test_client()


def _mri_events(text):
    """Fake full-telemetry stream: a meta frame, one token frame per byte carrying
    telemetry fields, then a stop frame. Stands in for Brain.stream / C trace=True."""
    def gen():
        yield {"kind": "meta", "checkpoint": "stub", "layers": 2}
        for b in text.encode("utf-8"):
            yield {"kind": "token", "byte": b, "ffn_full": [[b]], "attn": [],
                   "lens": [], "cand": [], "confidence": 0.5, "backend": "pytorch"}
        yield {"kind": "stop", "reason": "<|end|>"}
    return gen()


def _stub_local(monkeypatch, text=ANSWER):
    """Stub the local generation stream + backend ensure so the mri path runs without
    a model. _local_events yields the fake telemetry stream regardless of the flag."""
    monkeypatch.setattr(H, "_ensure_pytorch", lambda cfg, name: None)
    monkeypatch.setattr(H, "_ensure_c", lambda cfg, name: None)
    monkeypatch.setattr(H, "_local_events",
                        lambda cfg, backend, prompt, mri=False, gen=None: _mri_events(text))


def _chunks(resp):
    """Parse an OpenAI SSE body into its chunk objects (skipping [DONE])."""
    out = []
    for f in resp.get_data(as_text=True).split("\n\n"):
        f = f.strip()
        if f.startswith("data: ") and f != "data: [DONE]":
            out.append(json.loads(f[len("data: "):]))
    return out


# --- unit: the shared byte->text + frame emitter --------------------------------------

def test_stream_items_off_is_text_only():
    """emit_frames off yields only ('text', ...) segments reassembling to the answer."""
    items = list(H._local_stream_items(_mri_events(ANSWER)))
    assert {tag for tag, _ in items} == {"text"}
    assert "".join(v for _, v in items) == ANSWER


def test_stream_items_on_passes_frames_unfiltered():
    """emit_frames on yields every raw event as a ('frame', ev) unchanged, plus the text."""
    src = list(_mri_events(ANSWER))
    items = list(H._local_stream_items(iter(src), emit_frames=True))
    frames = [v for tag, v in items if tag == "frame"]
    assert frames == src  # same objects, no fields dropped (field symmetry)
    assert "".join(v for tag, v in items if tag == "text") == ANSWER


def test_generate_local_mri_returns_text_and_frames(monkeypatch):
    """_generate_local_mri returns the trimmed answer plus the captured per-byte frames."""
    _stub_local(monkeypatch)
    answer, frames = H._generate_local_mri({}, "m", "pytorch",
                                           [{"role": "user", "content": "hi"}], "")
    assert answer == ANSWER
    assert [f["kind"] for f in frames] == ["meta", "token", "token", "token", "token",
                                           "token", "token", "token", "token", "token",
                                           "token", "token", "stop"]


# --- /v1/chat/completions non-streaming ----------------------------------------------

def test_openai_nonstream_off_has_no_mri(monkeypatch):
    """Without the flag the completion is the plain OpenAI envelope, no mri key."""
    r = _client(monkeypatch).post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_json()
    assert body["object"] == "chat.completion"
    assert "mri" not in body


def test_openai_nonstream_on_attaches_frames(monkeypatch):
    """mri:true on a local model attaches the frames under the top-level mri key, envelope intact."""
    monkeypatch.setattr(H, "_generate_local_mri",
                        lambda cfg, model, backend, conv, system, gen=None: (ANSWER, FRAMES))
    r = _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "mri": True, "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["mri"] == FRAMES  # unchanged, unfiltered


def test_openai_nonstream_on_remote_ignores_flag(monkeypatch):
    """A remote model with mri:true is byte-identical to off: no telemetry, no mri key."""
    r = _client(monkeypatch, kind="remote").post(
        "/v1/chat/completions",
        json={"model": "m", "mri": True, "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert "mri" not in body


# --- /v1/chat/completions streaming --------------------------------------------------

def test_openai_stream_off_has_no_mri_chunks(monkeypatch):
    """Flag off: every streamed chunk is a plain text chunk; none carries an mri key."""
    _stub_local(monkeypatch)
    r = _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    chunks = _chunks(r)
    assert all("mri" not in c for c in chunks)
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_openai_stream_on_interleaves_mri_frames(monkeypatch):
    """Flag on: mri frames ride as chunk-extension frames (empty delta + mri key), unfiltered."""
    _stub_local(monkeypatch)
    r = _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "mri": True,
              "messages": [{"role": "user", "content": "hi"}]})
    chunks = _chunks(r)
    mri_chunks = [c for c in chunks if "mri" in c]
    assert [c["mri"]["kind"] for c in mri_chunks] == ["meta"] + ["token"] * len(ANSWER) + ["stop"]
    token_frame = next(c["mri"] for c in mri_chunks if c["mri"]["kind"] == "token")
    assert token_frame["ffn_full"] == [[ord("h")]]          # telemetry field survives
    for c in mri_chunks:
        assert c["object"] == "chat.completion.chunk"
        assert c["choices"][0]["delta"] == {}               # empty delta: plain clients ignore it


def test_openai_stream_on_offtheshelf_client_still_reassembles(monkeypatch):
    """An OpenAI reader that only reads choices[].delta.content reassembles the answer with mri on."""
    _stub_local(monkeypatch)
    r = _client(monkeypatch).post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "mri": True,
              "messages": [{"role": "user", "content": "hi"}]})
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in _chunks(r))
    assert text == ANSWER
