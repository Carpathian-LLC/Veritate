# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - concurrency tests for the C-engine persistent subprocess: clean-stream state,
#   the desync-respawn guard, and the wedged-holder reclaim. Uses the real engine
#   + a real bin; skips when either is absent (spawns a real subprocess).
# tests/mri/test_c_engine.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys
import threading
import time

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from readers import paths
from inference.backends import c_engine as ce

# ------------------------------------------------------------------------------------
# Constants

PROMPT = "<|system|>\nYou are Veritate.\n<|user|>\nhi\n<|assistant|>\n"

# ------------------------------------------------------------------------------------
# Functions

def _sub_or_skip():
    exe = paths.engine_binary_path()
    if not exe or not os.path.isfile(exe):
        pytest.skip("engine binary not built")
    binp = paths.bin_path("chat200m")
    if not binp or not os.path.isfile(binp):
        pytest.skip("chat200m bin missing")
    return ce.CTracedSubprocess(exe, binp)


def _gen(sub, n=6):
    out = []
    for fr in sub.stream(PROMPT, 0.2, 1, n):
        b = fr.get("byte")
        if isinstance(b, int):
            out.append(b)
        if len(out) >= n:
            break
    return out


def test_fresh_stream_produces_bytes_and_ends_clean():
    """A fresh subprocess streams bytes and marks the pipe clean after TEND."""
    sub = _sub_or_skip()
    try:
        assert len(_gen(sub)) >= 3
        assert sub._last_clean is True
    finally:
        sub.close()


def test_unclean_subprocess_respawns_on_next_request():
    """A subprocess left unclean (no TEND) is respawned on the next stream."""
    sub = _sub_or_skip()
    try:
        _gen(sub)
        old_pid = sub.proc.pid
        sub._last_clean = False
        assert len(_gen(sub)) >= 3
        assert sub.proc.pid != old_pid
    finally:
        sub.close()


def test_wedged_holder_is_reclaimed(monkeypatch):
    """A holder with a stale heartbeat is reclaimed (epoch bumps) so a new request proceeds."""
    monkeypatch.setattr(ce, "STREAM_LOCK_TIMEOUT_S", 2.0)
    monkeypatch.setattr(ce, "RECLAIM_POLL_S", 0.5)
    sub = _sub_or_skip()
    try:
        def _wedge():
            sub.lock.acquire()
            sub._last_frame_time = time.monotonic() - 100.0
            time.sleep(30)
        threading.Thread(target=_wedge, daemon=True).start()
        time.sleep(0.4)
        epoch0 = sub._epoch
        assert len(_gen(sub)) >= 3
        assert sub._epoch > epoch0
    finally:
        sub.close()


def test_healthy_slow_holder_is_not_reclaimed(monkeypatch):
    """A holder that keeps its heartbeat fresh is NOT killed; the waiter proceeds after it."""
    monkeypatch.setattr(ce, "STREAM_LOCK_TIMEOUT_S", 2.0)
    monkeypatch.setattr(ce, "RECLAIM_POLL_S", 0.5)
    sub = _sub_or_skip()
    try:
        stop = {"v": False}
        def _hold():
            sub.lock.acquire()
            for _ in range(8):
                if stop["v"]:
                    break
                sub._last_frame_time = time.monotonic()
                time.sleep(0.5)
            sub.lock.release()
        threading.Thread(target=_hold, daemon=True).start()
        time.sleep(0.4)
        epoch0 = sub._epoch
        assert len(_gen(sub)) >= 3
        assert sub._epoch == epoch0  # healthy holder never reclaimed
    finally:
        stop["v"] = True
        sub.close()
