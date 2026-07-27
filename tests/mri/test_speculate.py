# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - speculative prefetch: job supersede/stand-down/take semantics, and the
#   _c_engine_stream flush that replays prefetched frames before the engine runs.
# - a fake subprocess stands in for the c engine (rule 33, no model loads). it
#   answers from a fixed reply, resuming at whatever the caller already has, and a
#   fake frame builder stands in for MRI assembly.
# tests/mri/test_speculate.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

from inference import speculate
from routes import backends_routes

# ------------------------------------------------------------------------------------
# Constants

BASE_PROMPT = "who are you?"
REPLY       = b"a small model."
PARAMS      = backends_routes._decode_params(0.7, 40)
BUDGET      = 64
CHUNK       = 4
IDLE_TIMEOUT_S = 5.0
IDLE_POLL_S    = 0.01
SHAPE = {"layers": 1, "hidden": 8, "ffn": 16, "heads": 1, "seq": 64, "vocab": 256}

# ------------------------------------------------------------------------------------
# Functions

class FakeSub:
    """Yields the next slice of REPLY given how much of it is already in the prompt."""

    def __init__(self, base=BASE_PROMPT, reply=REPLY):
        self.base   = base
        self.reply  = reply
        self.calls  = []
        self.shape  = dict(SHAPE)
        self.lock   = threading.Lock()

    def stream(self, prompt, temperature, top_k, max_new, **kwargs):
        self.calls.append((prompt, max_new))
        done = len(prompt) - len(self.base)
        for b in self.reply[done:done + max_new]:
            yield {"fast": True, "byte": b, "argmax_byte": b, "real_len": len(prompt)}


def _wait_idle():
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    while speculate.status()["speculating"] and time.monotonic() < deadline:
        time.sleep(IDLE_POLL_S)


def _frame(raw, fwd_ms):
    return {"kind": "token", "byte": int(raw["byte"]), "fwd_ms": fwd_ms}


def _start(sub, prompt=BASE_PROMPT):
    draft_id = speculate.start(sub, prompt, PARAMS, BUDGET, CHUNK, (), _frame)["draft_id"]
    _wait_idle()
    return draft_id


def _reply(frames):
    return bytes(f["byte"] for f in frames)


def setup_function(_fn):
    speculate.stand_down()
    for k in speculate._STATS:
        speculate._STATS[k] = 0


def test_take_returns_the_speculated_reply_for_the_same_request():
    """A draft speculated to completion hands its bytes to a take() with the same prompt."""
    sub = FakeSub()
    draft_id = _start(sub)
    assert _reply(speculate.take(sub, BASE_PROMPT, draft_id)) == REPLY


def test_speculation_runs_in_chunks_of_the_configured_size():
    """The runner issues one engine turn per chunk, resuming from the bytes it has."""
    sub = FakeSub()
    _start(sub)
    assert [max_new for _prompt, max_new in sub.calls] == [CHUNK] * (len(REPLY) // CHUNK + 1)
    assert sub.calls[1][0] == BASE_PROMPT + REPLY[:CHUNK].decode()


def test_take_with_a_different_prompt_is_a_miss():
    """An edited prompt never flushes the buffer speculated for the old draft."""
    sub = FakeSub()
    draft_id = _start(sub)
    assert speculate.take(sub, BASE_PROMPT + " really", draft_id) == []


def test_take_without_the_draft_id_is_a_miss():
    """A request that does not claim the draft generates normally."""
    sub = FakeSub()
    _start(sub)
    assert speculate.take(sub, BASE_PROMPT, 0) == []


def test_a_miss_still_cancels_the_job_so_the_request_is_never_starved():
    """Every take() ends the live draft: a running job would hold the engine lock."""
    sub = FakeSub()
    _start(sub)
    speculate.take(sub, BASE_PROMPT, 0)
    assert speculate.status()["speculating"] is False


def test_take_with_a_superseded_draft_id_is_a_miss():
    """The id of a draft that was replaced never claims the newer buffer."""
    sub = FakeSub()
    stale = _start(sub)
    _start(sub, BASE_PROMPT + " now.")
    assert speculate.take(sub, BASE_PROMPT + " now.", stale) == []


def test_stand_down_drops_the_buffer():
    """Typing again (empty draft) discards the speculated reply."""
    sub = FakeSub()
    draft_id = _start(sub)
    speculate.stand_down()
    assert speculate.take(sub, BASE_PROMPT, draft_id) == []


def test_take_consumes_the_job_so_a_buffer_is_flushed_once():
    """A second take() for the same prompt gets nothing: buffers are one-shot."""
    sub = FakeSub()
    draft_id = _start(sub)
    speculate.take(sub, BASE_PROMPT, draft_id)
    assert speculate.take(sub, BASE_PROMPT, draft_id) == []


def test_a_real_generation_in_flight_blocks_speculation():
    """No speculative turn starts while a real request holds the subprocess lock."""
    sub = FakeSub()
    with sub.lock:
        _start(sub)
    assert sub.calls == []


def test_stats_report_served_against_spent_bytes():
    """The counters expose the trade: bytes handed to a request vs bytes speculated."""
    sub = FakeSub()
    draft_id = _start(sub)
    speculate.take(sub, BASE_PROMPT, draft_id)
    stats = speculate.status()["stats"]
    assert (stats["served"], stats["served_bytes"], stats["spent_bytes"]) == (1, len(REPLY), len(REPLY))


def _frames_for(reply):
    return [{"kind": "token", "byte": b, "fwd_ms": 0.0} for b in reply]


def _stream_events(monkeypatch, prefetched, max_new):
    monkeypatch.setattr(backends_routes.settings_mod, "get", dict)
    sub = FakeSub()
    cfg = {"C_SUBPROCESS": sub, "C_MODEL": "/models/m/veritate.bin", "C_EXE": "/bin/veritate"}
    events = list(backends_routes._c_engine_stream(cfg, BASE_PROMPT, max_new, trace=False,
                                                   prefetched=prefetched))
    return sub, events


def test_prefetched_frames_replay_before_the_engine_runs(monkeypatch):
    """A flushed prefetch emits a prefetch frame then the buffered token frames."""
    _sub, events = _stream_events(monkeypatch, _frames_for(REPLY[:4]), len(REPLY))
    kinds = [e["kind"] for e in events]
    assert kinds[:2] == ["meta", "prefetch"]
    assert bytes(e["byte"] for e in events[2:6]) == REPLY[:4]


def test_flushed_frames_carry_their_telemetry(monkeypatch):
    """A prefetched byte reaches the dashboard as a token frame, not a bare byte."""
    _sub, events = _stream_events(monkeypatch, _frames_for(REPLY[:4]), len(REPLY))
    assert [e["kind"] for e in events[2:6]] == ["token"] * 4


def test_the_engine_continues_from_the_prefetched_bytes(monkeypatch):
    """The continuation turn asks for the remaining bytes with the reply in its prompt."""
    sub, _events = _stream_events(monkeypatch, _frames_for(REPLY[:4]), len(REPLY))
    assert sub.calls == [(BASE_PROMPT + REPLY[:4].decode(), len(REPLY) - 4)]


def test_a_complete_prefetch_never_touches_the_engine(monkeypatch):
    """When the buffer covers max_new the request is served without an engine turn."""
    sub, _events = _stream_events(monkeypatch, _frames_for(REPLY), len(REPLY))
    assert sub.calls == []


def _prefetch_client(monkeypatch, sub):
    from flask import Flask
    # Real MRI assembly needs a real engine frame; this exercises the prefetch round
    # trip, so the builder is stubbed and frame contents are covered by their own tests.
    monkeypatch.setattr(backends_routes, "_frame_builder", lambda _shape: _frame)
    monkeypatch.setattr(backends_routes.settings_mod, "get",
                        lambda: {"speculative_enabled": True, "speculative_bytes": BUDGET,
                                 "speculative_chunk_bytes": CHUNK})
    app = Flask(__name__)
    app.config["C_SUBPROCESS"] = sub
    app.config["C_MODEL"] = "/models/m/veritate.bin"
    app.config["C_EXE"]   = "/bin/veritate"
    backends_routes.register(app)
    return app.test_client()


def test_a_draft_posted_to_prefetch_is_flushed_by_the_matching_generate(monkeypatch):
    """The round trip the dashboard drives: POST the draft, then GET /generate serves it."""
    sub = FakeSub()
    client = _prefetch_client(monkeypatch, sub)
    draft = client.post("/prefetch", json={"prompt": BASE_PROMPT, "temperature": 0.7,
                                           "top_k": 40}).get_json()
    _wait_idle()
    body = client.get(f"/generate?prompt={BASE_PROMPT}&temperature=0.7&top_k=40&max_new=200"
                      f"&backend=c&prefetch_id={draft['draft_id']}").get_data(as_text=True)
    assert '"kind": "prefetch"' in body


def test_generate_without_the_draft_id_leaves_the_buffer_unclaimed(monkeypatch):
    """A submit that does not echo the id generates normally, never a stale reply."""
    sub = FakeSub()
    client = _prefetch_client(monkeypatch, sub)
    client.post("/prefetch", json={"prompt": BASE_PROMPT, "temperature": 0.7, "top_k": 40})
    _wait_idle()
    body = client.get(f"/generate?prompt={BASE_PROMPT}&temperature=0.7&top_k=40&max_new=200"
                      "&backend=c").get_data(as_text=True)
    assert '"kind": "prefetch"' not in body
