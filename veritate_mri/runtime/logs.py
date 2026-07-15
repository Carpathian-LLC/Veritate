# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - in-memory ring buffer for status and error messages. no file IO.
# - dashboard Logs tab streams from this via the /logs SSE endpoint.
# - any module pushes via emit(level, source, msg).
# veritate_mri/runtime/logs.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time
from collections import deque

# ------------------------------------------------------------------------------------
# Constants

RING_CAPACITY = 1000

LEVELS = ("info", "warn", "error", "ok")

_LOCK = threading.Lock()
_BUFFER = deque(maxlen=RING_CAPACITY)
_SEQ = 0
_SUBSCRIBERS = set()
_ERROR_HOOK = None

# Per-key throttle state for emit_throttled: key -> {"last_emit", "suppressed"}.
_THROTTLE_LOCK = threading.Lock()
_THROTTLE = {}

# ------------------------------------------------------------------------------------
# Functions

def emit(level, source, msg):
    global _SEQ
    if level not in LEVELS:
        level = "info"
    entry = {
        "seq":    0,
        "ts":     time.time(),
        "level":  level,
        "source": str(source),
        "msg":    str(msg),
    }
    with _LOCK:
        _SEQ += 1
        entry["seq"] = _SEQ
        _BUFFER.append(entry)
        targets = list(_SUBSCRIBERS)
    if level == "error" and _ERROR_HOOK is not None:
        try:
            _ERROR_HOOK(entry["source"], entry["msg"])
        except Exception:
            pass
    for q in targets:
        try:
            q.put_nowait(entry)
        except Exception:
            pass
    return entry


def set_error_hook(fn):
    global _ERROR_HOOK
    _ERROR_HOOK = fn


def info(source, msg):  return emit("info",  source, msg)
def warn(source, msg):  return emit("warn",  source, msg)
def error(source, msg): return emit("error", source, msg)
def ok(source, msg):    return emit("ok",    source, msg)


def emit_throttled(level, source, msg, key, interval_secs=1800):
    """Coalesce repeated, expected-to-recur log lines (e.g. network failures on
    an offline machine) under a stable `key`. The first call emits immediately;
    further calls with the same key inside `interval_secs` are dropped and
    counted. When one finally emits after suppressions, it appends
    "(+N suppressed …)". Returns the emitted entry, or None if this call was
    suppressed. Call clear_throttle(key) on the success/recovery edge so the
    next failure logs at once again."""
    now = time.time()
    with _THROTTLE_LOCK:
        st = _THROTTLE.get(key)
        if st is not None and (now - st["last_emit"]) < interval_secs:
            st["suppressed"] += 1
            return None
        suppressed = st["suppressed"] if st else 0
        _THROTTLE[key] = {"last_emit": now, "suppressed": 0}
    if suppressed:
        msg = f"{msg} (+{suppressed} suppressed since last shown)"
    return emit(level, source, msg)


def clear_throttle(key):
    """Reset a throttle key so the next emit_throttled(key) fires immediately.
    Used on a recovery edge (a send that succeeds after failures) so the first
    failure of the next outage is always shown."""
    with _THROTTLE_LOCK:
        return _THROTTLE.pop(key, None) is not None


def snapshot(after_seq=0, limit=None):
    with _LOCK:
        rows = [e for e in _BUFFER if e["seq"] > after_seq]
    if limit is not None:
        rows = rows[-int(limit):]
    return rows


def latest_seq():
    with _LOCK:
        return _SEQ


def subscribe():
    """Returns a Queue that receives every new entry. Caller must call unsubscribe()."""
    import queue
    q = queue.Queue(maxsize=RING_CAPACITY)
    with _LOCK:
        _SUBSCRIBERS.add(q)
        backlog = list(_BUFFER)
    for e in backlog:
        try:
            q.put_nowait(e)
        except Exception:
            break
    return q


def unsubscribe(q):
    with _LOCK:
        _SUBSCRIBERS.discard(q)
