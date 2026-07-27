# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read-ahead: the engine reads a prompt while it is still being typed so the request
#   carrying it skips the prefill. A fake subprocess stands in for the C engine, so
#   these assert the lock discipline and supersede rules, not engine timing.
# tests/mri/test_read_ahead.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

import pytest
from inference import speculate

# ------------------------------------------------------------------------------------
# Constants

PARAMS = {"temperature": 0.7, "top_k": 40, "ablate_layer": -1, "ablate_neuron": -1,
          "addons_csv": "", "rep_window": 0, "rep_penalty": 0.0, "no_repeat_ngram": 0}
PREFIX  = "<|im_start|>user\nwhat is the cap"
LONGER  = "<|im_start|>user\nwhat is the capital of France"
SETTLE_S = 0.5
POLL_S   = 0.01


class FakeSub:
    """Stands in for CTracedSubprocess: records what it was asked to read and holds the
    lock for the duration, exactly as the real one does."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reads = []
        self.shape = {"layers": 2, "hidden": 8, "ffn": 16, "heads": 2, "seq": 64, "vocab": 256}

    def stream(self, prompt, temperature, top_k, max_new, **kw):
        with self.lock:
            self.reads.append((prompt, max_new, kw.get("do_trace")))
            yield {"byte": 65, "pos": 0, "real_len": len(prompt), "fast": True}


def _settle(pred):
    end = time.time() + SETTLE_S
    while time.time() < end and not pred():
        time.sleep(POLL_S)
    return pred()


@pytest.fixture(autouse=True)
def clean():
    """Every test starts with no read-ahead in flight."""
    speculate.read_stand_down()
    yield
    speculate.read_stand_down()


# ------------------------------------------------------------------------------------
# Functions

def test_read_ahead_reads_the_prefix_into_the_engine():
    """The posted prefix is handed to the engine so its state is cached."""
    sub = FakeSub()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    assert _settle(lambda: [r[0] for r in sub.reads] == [PREFIX])


def test_read_ahead_generates_one_byte_untraced():
    """Reading costs one untraced byte: the engine caches its state at step 0."""
    sub = FakeSub()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    _settle(lambda: sub.reads)
    assert sub.reads[0][1:] == (speculate.READ_MAX_NEW, False)


def test_reposting_the_same_prefix_does_not_read_it_twice():
    """A client may post freely; an unchanged prefix is already being read."""
    sub = FakeSub()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    _settle(lambda: sub.reads)
    speculate.read_ahead(sub, PREFIX, PARAMS)
    time.sleep(POLL_S * 5)
    assert len(sub.reads) == 1


def test_a_longer_prefix_supersedes_the_one_in_flight():
    """Typing more replaces the read rather than queueing behind it."""
    sub = FakeSub()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    _settle(lambda: sub.reads)
    speculate.read_ahead(sub, LONGER, PARAMS)
    assert _settle(lambda: [r[0] for r in sub.reads] == [PREFIX, LONGER])


def test_read_ahead_yields_when_a_real_request_holds_the_engine():
    """A held lock means a request owns the engine, so nothing is read behind it."""
    sub = FakeSub()
    sub.lock.acquire()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    time.sleep(POLL_S * 10)
    sub.lock.release()
    assert sub.reads == []


def test_stand_down_reports_nothing_in_flight():
    """After standing down the status carries no live read."""
    sub = FakeSub()
    speculate.read_ahead(sub, PREFIX, PARAMS)
    _settle(lambda: sub.reads)
    assert speculate.read_stand_down()["reading"] is False


def test_status_counts_the_bytes_read_ahead():
    """Stats report how much prompt was read ahead, for the energy trade."""
    sub = FakeSub()
    before = speculate.read_status()["stats"]["bytes"]
    speculate.read_ahead(sub, PREFIX, PARAMS)
    assert _settle(lambda: speculate.read_status()["stats"]["bytes"] == before + len(PREFIX))


def test_dashboard_read_ahead_follows_the_dashboard_setting(monkeypatch):
    """A dashboard request (no bearer token) is gated by read_ahead_enabled."""
    from routes import backends_routes
    monkeypatch.setattr(backends_routes, "_is_api_caller", lambda: False)
    monkeypatch.setattr(backends_routes.settings_mod, "get",
                        lambda: {"read_ahead_enabled": False, "api_read_ahead_enabled": True})
    assert backends_routes._ahead_allowed(read=True) is False


def test_api_read_ahead_follows_the_api_setting(monkeypatch):
    """A caller presenting a bearer token is gated by api_read_ahead_enabled."""
    from routes import backends_routes
    monkeypatch.setattr(backends_routes, "_is_api_caller", lambda: True)
    monkeypatch.setattr(backends_routes.settings_mod, "get",
                        lambda: {"read_ahead_enabled": False, "api_read_ahead_enabled": True})
    assert backends_routes._ahead_allowed(read=True) is True


def test_api_generate_ahead_is_gated_separately(monkeypatch):
    """Generate-ahead can be denied to API callers while read-ahead stays allowed."""
    from routes import backends_routes
    monkeypatch.setattr(backends_routes, "_is_api_caller", lambda: True)
    monkeypatch.setattr(backends_routes.settings_mod, "get",
                        lambda: {"api_read_ahead_enabled": True,
                                 "api_generate_ahead_enabled": False})
    assert backends_routes._ahead_allowed(read=False) is False


def test_prefill_wraps_a_messages_body_in_the_chat_template():
    """A client posts chat messages and this box renders the prefix, scaffold excluded."""
    from routes import backends_routes, hybrid_routes
    body = {"messages": [{"role": "user", "content": "what is the cap"}]}
    assert backends_routes._chat_prefix_in(body) == \
        hybrid_routes.render_local_open(body["messages"], system="")


def test_prefill_reads_nothing_from_a_body_with_no_messages():
    """A body carrying no messages leaves the raw-prompt path to answer for itself."""
    from routes import backends_routes
    assert backends_routes._chat_prefix_in({}) == ""
