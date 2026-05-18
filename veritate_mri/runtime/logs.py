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
