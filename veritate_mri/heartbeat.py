# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - phones home to carpathian.ai every HEARTBEAT_INTERVAL_SECS with a tiny json
#   payload: machine_id, version, uptime, total_runtime, restarts, errors,
#   models hash, optional active_training. opt-out via settings.heartbeat_enabled.
# - state persisted at data/heartbeat_state.json. one daemon thread, one socket
#   connection per ping, no deps beyond stdlib + readers.
# veritate_mri/heartbeat.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import platform
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid

import logs as logmod
import settings as settings_mod
from readers import paths, models as models_reader

# ------------------------------------------------------------------------------------
# Constants

HEARTBEAT_URL          = "https://api.carpathian.ai/webhook/veritate-heartbeat"
HEARTBEAT_INTERVAL_SECS = 6 * 60 * 60
HEARTBEAT_FIRST_DELAY   = 5 * 60
HEARTBEAT_TIMEOUT_SECS  = 8.0
HEARTBEAT_USER_AGENT    = "veritate-heartbeat/1"

STATE_PATH    = os.path.join(paths.REPO_ROOT, "data", "heartbeat_state.json")
MACHINE_ID_LEN = 16
MODELS_HASH_LEN = 12

PROTOCOL_VERSION = 1

_LOCK        = threading.Lock()
_PROCESS_START = time.monotonic()
_STATE_CACHE = None
_THREAD      = None
_TRAINING_FN = None

# ------------------------------------------------------------------------------------
# Functions

def _read_state():
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(data):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STATE_PATH)


def _state():
    global _STATE_CACHE
    with _LOCK:
        if _STATE_CACHE is None:
            _STATE_CACHE = {**_read_state()}
        return dict(_STATE_CACHE)


def _update_state(patch):
    global _STATE_CACHE
    with _LOCK:
        cur = {**_read_state(), **(patch or {})}
        _write_state(cur)
        _STATE_CACHE = cur
        return dict(cur)


def _machine_id():
    s = _state()
    mid = s.get("machine_id")
    if isinstance(mid, str) and len(mid) == MACHINE_ID_LEN:
        return mid
    parts = [
        platform.node() or "",
        platform.system() or "",
        platform.machine() or "",
        platform.processor() or "",
        str(uuid.getnode()),
    ]
    seed = "|".join(parts).encode("utf-8")
    mid = hashlib.sha256(seed).hexdigest()[:MACHINE_ID_LEN]
    _update_state({"machine_id": mid})
    return mid


def _models_hash():
    names = models_reader.list_models()
    if not names:
        return "", 0
    blob = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:MODELS_HASH_LEN], len(names)


def record_restart():
    s = _state()
    n = int(s.get("restarts") or 0) + 1
    _update_state({"restarts": n, "last_start_ts": time.time()})
    return n


def record_error_tick():
    s = _state()
    n = int(s.get("errors_pending") or 0) + 1
    _update_state({"errors_pending": n})


def _consume_errors():
    s = _state()
    n = int(s.get("errors_pending") or 0)
    if n:
        _update_state({"errors_pending": 0})
    return n


def _accumulate_runtime(uptime_secs):
    s = _state()
    total = float(s.get("total_runtime_secs") or 0.0) + float(uptime_secs)
    _update_state({"total_runtime_secs": total})
    return total


def set_training_provider(fn):
    """Optional callable returning a small dict describing in-progress training,
    or None when nothing is running. Output is sent verbatim if not None."""
    global _TRAINING_FN
    _TRAINING_FN = fn


def _build_payload():
    uptime = max(0.0, time.monotonic() - _PROCESS_START)
    s = _state()
    mh, n_models = _models_hash()
    payload = {
        "v":           PROTOCOL_VERSION,
        "machine_id":  _machine_id(),
        "ts":          int(time.time()),
        "host":        platform.node() or "",
        "os":          platform.system() or "",
        "arch":        platform.machine() or "",
        "uptime_secs": int(uptime),
        "total_runtime_secs": int(float(s.get("total_runtime_secs") or 0.0) + uptime),
        "restarts":    int(s.get("restarts") or 0),
        "errors":      _consume_errors(),
        "n_models":    n_models,
        "models_hash": mh,
    }
    fn = _TRAINING_FN
    if fn is not None:
        try:
            t = fn()
            if isinstance(t, dict) and t:
                payload["training"] = t
        except Exception:
            pass
    return payload


def _post(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        HEARTBEAT_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type":  "application/json",
            "User-Agent":    HEARTBEAT_USER_AGENT,
            "X-Machine-Id":  payload.get("machine_id") or "",
        },
    )
    with urllib.request.urlopen(req, timeout=HEARTBEAT_TIMEOUT_SECS) as resp:
        return resp.status


def _send_once():
    payload = _build_payload()
    try:
        status = _post(payload)
        _update_state({
            "last_send_ts":     int(time.time()),
            "last_send_status": int(status),
            "last_send_error":  None,
        })
        return True
    except urllib.error.HTTPError as e:
        body_excerpt = ""
        try:
            body_excerpt = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        reason = f"http {e.code}: {body_excerpt or e.reason}"
        logmod.warn("heartbeat", f"send failed: {reason}")
        _update_state({
            "last_send_ts":     int(time.time()),
            "last_send_status": int(e.code),
            "last_send_error":  reason,
        })
        return False
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        reason = f"{type(e).__name__}: {e}"
        logmod.warn("heartbeat", f"send failed: {reason}")
        _update_state({
            "last_send_ts":     int(time.time()),
            "last_send_status": 0,
            "last_send_error":  reason,
        })
        return False


def _enabled():
    s = settings_mod.get()
    return bool(s.get("heartbeat_enabled"))


def _loop():
    last_persist = time.monotonic()
    next_send    = time.monotonic() + HEARTBEAT_FIRST_DELAY
    while True:
        time.sleep(60)
        now = time.monotonic()
        if now - last_persist >= 600:
            _accumulate_runtime(now - last_persist)
            last_persist = now
            _update_state({"last_start_ts": time.time()})
        if not _enabled():
            next_send = now + HEARTBEAT_INTERVAL_SECS
            continue
        if now >= next_send:
            _send_once()
            next_send = now + HEARTBEAT_INTERVAL_SECS


def start():
    global _THREAD
    if _THREAD is not None:
        return
    record_restart()
    _machine_id()
    logmod.set_error_hook(record_error_tick)
    t = threading.Thread(target=_loop, name="heartbeat", daemon=True)
    t.start()
    _THREAD = t
    logmod.info("heartbeat", f"enabled={_enabled()} machine_id={_machine_id()}")


def status():
    s = _state()
    return {
        "machine_id":         _machine_id(),
        "enabled":            _enabled(),
        "interval_secs":      HEARTBEAT_INTERVAL_SECS,
        "url":                HEARTBEAT_URL,
        "restarts":           int(s.get("restarts") or 0),
        "total_runtime_secs": int(float(s.get("total_runtime_secs") or 0.0)),
        "last_send_ts":       s.get("last_send_ts"),
        "last_send_status":   s.get("last_send_status"),
        "last_send_error":    s.get("last_send_error"),
        "errors_pending":     int(s.get("errors_pending") or 0),
    }


def send_now():
    return _send_once()
