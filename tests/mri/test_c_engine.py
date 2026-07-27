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
import threading
import time

import pytest
from inference.backends import c_engine as ce
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

PROMPT = "<|system|>\nYou are Veritate.\n<|user|>\nhi\n<|assistant|>\n"

LOCK_TIMEOUT_S      = 2.0    # monkeypatched STREAM_LOCK_TIMEOUT_S
RECLAIM_POLL_S      = 0.5
HANDSHAKE_S         = 10.0   # bound on a thread handshake; never a timing assertion
STALE_HEARTBEAT_S   = 100.0  # heartbeat age that is unambiguously past LOCK_TIMEOUT_S
HEARTBEAT_PERIOD_S  = 0.2    # well under LOCK_TIMEOUT_S, so the holder reads as healthy
HEARTBEAT_TICKS     = 5

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


def _run_turn(sub, prompt, max_new=2):
    """Fully consume one turn (to TEND); return (first frame's real_len, frame count)."""
    first_real_len = None
    frames = 0
    for fr in sub.stream(prompt, 0.2, 1, max_new):
        if first_real_len is None:
            first_real_len = fr.get("real_len")
        frames += 1
    return first_real_len, frames


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


@pytest.mark.slow
def test_wedged_holder_is_reclaimed(monkeypatch):
    """A holder with a stale heartbeat is reclaimed so a waiting request bumps the epoch."""
    monkeypatch.setattr(ce, "STREAM_LOCK_TIMEOUT_S", LOCK_TIMEOUT_S)
    monkeypatch.setattr(ce, "RECLAIM_POLL_S", RECLAIM_POLL_S)
    sub = _sub_or_skip()
    held, release = threading.Event(), threading.Event()

    def _wedge():
        sub.lock.acquire()
        sub._last_frame_time = time.monotonic() - STALE_HEARTBEAT_S
        held.set()
        release.wait(HANDSHAKE_S)

    holder = threading.Thread(target=_wedge, daemon=True)
    holder.start()
    try:
        assert held.wait(HANDSHAKE_S), "holder never took the stream lock"
        epoch0 = sub._epoch
        _gen(sub)
        assert sub._epoch > epoch0
    finally:
        release.set()
        holder.join(HANDSHAKE_S)
        sub.close()


@pytest.mark.slow
def test_healthy_slow_holder_is_not_reclaimed(monkeypatch):
    """A holder that keeps its heartbeat fresh is never reclaimed: the epoch is unchanged."""
    monkeypatch.setattr(ce, "STREAM_LOCK_TIMEOUT_S", LOCK_TIMEOUT_S)
    monkeypatch.setattr(ce, "RECLAIM_POLL_S", RECLAIM_POLL_S)
    sub = _sub_or_skip()
    held = threading.Event()

    def _hold():
        sub.lock.acquire()
        held.set()
        for _ in range(HEARTBEAT_TICKS):
            sub._last_frame_time = time.monotonic()
            time.sleep(HEARTBEAT_PERIOD_S)
        sub.lock.release()

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    try:
        assert held.wait(HANDSHAKE_S), "holder never took the stream lock"
        epoch0 = sub._epoch
        _gen(sub)
        assert sub._epoch == epoch0
    finally:
        holder.join(HANDSHAKE_S)
        sub.close()


def test_overlong_prompt_keeps_following_turn_in_sync():
    """An over-long prompt is tail-clamped so the following short turn reports its true real_len."""
    sub = _sub_or_skip()
    try:
        seq = sub.shape["seq"]
        max_new = 8
        rl_long, frames_long = _run_turn(sub, "x" * (seq * 2), max_new=max_new)
        assert rl_long == seq - min(max_new, seq // 2)  # clamp reserves reply room
        assert frames_long > 0                           # long turn actually generates
        rl_short, _ = _run_turn(sub, "hi")
        assert rl_short == 2                             # following 2-byte turn in sync
    finally:
        sub.close()


def test_overlong_prompt_does_not_respawn_backend():
    """An over-long prompt is handled in-band: the subprocess is not killed/respawned as a desync."""
    sub = _sub_or_skip()
    try:
        pid0 = sub.proc.pid
        _run_turn(sub, "x" * (sub.shape["seq"] * 2))
        rl, frames = _run_turn(sub, "hi")
        assert (rl, frames > 0) == (2, True)
        assert sub.proc.pid == pid0
        assert sub._last_clean is True
    finally:
        sub.close()
