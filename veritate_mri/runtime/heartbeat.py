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
# veritate_mri/runtime/heartbeat.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

from readers import paths, models as models_reader
from . import logs as logmod
from . import settings as settings_mod
from . import sys_metrics

# ------------------------------------------------------------------------------------
# Constants

HEARTBEAT_URL          = "https://api.carpathian.ai/webhook/veritate-heartbeat"
HEARTBEAT_INTERVAL_SECS = 6 * 60 * 60
# +/- jitter applied to every scheduled send. Spreads thundering herds when
# many clients restart at once (release rollout, regional power blip, etc).
HEARTBEAT_JITTER_FRAC   = 0.10
HEARTBEAT_FIRST_DELAY   = 5 * 60
HEARTBEAT_TIMEOUT_SECS  = 8.0
HEARTBEAT_USER_AGENT    = "veritate-heartbeat/2"

STATE_PATH    = os.path.join(paths.REPO_ROOT, "data", "heartbeat_state.json")
MACHINE_ID_LEN = 16
MODELS_HASH_LEN = 12
DEVICE_ID_DEFAULT_LEN = 8
HOST_TOKEN_LEN = 12

PROTOCOL_VERSION = 2
TRAINING_EVENTS_PER_PING_MAX = 32
ERRORS_PER_PING_MAX = 20
ERROR_SOURCE_MAX  = 32
ERROR_MESSAGE_MAX = 240

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


def _host_token():
    """Random per-install token persisted in state. Replaces the OS hostname
    in the payload so we never ship platform.node() (which on macOS reveals
    the user's name, e.g. 'Sams-MacBook-Pro.local')."""
    s = _state()
    h = s.get("host_token")
    if isinstance(h, str) and len(h) == HOST_TOKEN_LEN:
        return h
    h = uuid.uuid4().hex[:HOST_TOKEN_LEN]
    _update_state({"host_token": h})
    return h


def _default_device_id():
    return _machine_id()[:DEVICE_ID_DEFAULT_LEN]


def _effective_device_id():
    name = settings_mod.get().get("device_name") or ""
    name = name.strip() if isinstance(name, str) else ""
    max_len = getattr(settings_mod, "DEVICE_NAME_MAX_LEN", 15)
    if name:
        return name[:max_len]
    return _default_device_id()


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


def record_error_tick(source="", msg=""):
    """Logger hook. Always increments the count; appends a {ts, source, msg}
    record to a bounded ring so the next heartbeat can ship error detail when
    heartbeat_send_errors is enabled. Truncates fields so a runaway logger
    can't blow up state."""
    s = _state()
    n = int(s.get("errors_pending") or 0) + 1
    patch = {"errors_pending": n}
    buf = s.get("errors_pending_detail") or []
    if not isinstance(buf, list):
        buf = []
    buf.append({
        "ts":     int(time.time()),
        "source": str(source or "")[:ERROR_SOURCE_MAX],
        "msg":    str(msg or "")[:ERROR_MESSAGE_MAX],
    })
    if len(buf) > ERRORS_PER_PING_MAX * 4:
        buf = buf[-ERRORS_PER_PING_MAX * 4:]
    patch["errors_pending_detail"] = buf
    _update_state(patch)


def record_training_event(model_name, arch, started_at=None):
    """Queue a training-start event. Drained on next heartbeat when advanced
    telemetry is enabled."""
    if not isinstance(model_name, str) or not model_name:
        return
    s = _state()
    pend = s.get("pending_training_events") or []
    if not isinstance(pend, list):
        pend = []
    pend.append({
        "model":      model_name[:128],
        "arch":       (arch or "")[:64],
        "started_at": int(started_at if started_at is not None else time.time()),
    })
    if len(pend) > TRAINING_EVENTS_PER_PING_MAX * 4:
        pend = pend[-TRAINING_EVENTS_PER_PING_MAX * 4:]
    _update_state({"pending_training_events": pend})


def _consume_errors(with_detail):
    """Drains the pending error counter and the detail ring. Returns
    (count, detail_list_or_None). detail_list is None when the user has
    opted out of error-detail telemetry; the count itself is always part of
    the minimal payload."""
    s = _state()
    n = int(s.get("errors_pending") or 0)
    detail = None
    patch = {}
    if n:
        patch["errors_pending"] = 0
    if with_detail:
        buf = s.get("errors_pending_detail") or []
        detail = buf[-ERRORS_PER_PING_MAX:] if isinstance(buf, list) and buf else []
        if buf:
            patch["errors_pending_detail"] = []
    else:
        # Opted out: drop the buffer so nothing carries over if the user
        # toggles the checkbox on later. Their consent applies forward.
        if s.get("errors_pending_detail"):
            patch["errors_pending_detail"] = []
    if patch:
        _update_state(patch)
    return n, detail


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


def _hw_block():
    snap = sys_metrics.snapshot()
    if not snap or not snap.get("available"):
        return None
    gpus_out = []
    for g in (snap.get("gpus") or [])[:8]:
        gpus_out.append({
            "vendor":     g.get("vendor") or "",
            "name":       (g.get("name") or "")[:96],
            "integrated": bool(g.get("integrated")),
            "vram_total": int(g.get("vram_total") or 0) or None,
        })
    return {
        "cpu_count":       int(snap.get("cpu_count") or 0) or None,
        "ram_total_bytes": int(snap.get("sys_mem_total") or 0) or None,
        "gpus":            gpus_out,
    }


def _build_payload():
    """Tiered payload:
      minimal (always sent if heartbeat enabled): machine_id, ts, uptime,
        restarts, error count, presence of training (no detail).
      errors tier (heartbeat_send_errors): full {ts, source, msg} list.
      analytics tier (analytics_advanced_enabled): host/os/arch, total
        runtime, n_models, models_hash, hw block (once), training detail
        with model name + shape + n_params, trainings event buffer."""
    uptime = max(0.0, time.monotonic() - _PROCESS_START)
    s    = _state()
    cfg  = settings_mod.get()
    send_errors    = bool(cfg.get("heartbeat_send_errors"))
    send_analytics = bool(cfg.get("analytics_advanced_enabled"))
    err_count, err_detail = _consume_errors(with_detail=send_errors)
    payload = {
        "v":           PROTOCOL_VERSION,
        "machine_id":  _machine_id(),
        "device_id":   _effective_device_id(),
        "ts":          int(time.time()),
        "uptime_secs": int(uptime),
        "restarts":    int(s.get("restarts") or 0),
        "errors":      err_count,
    }
    if send_errors and err_detail:
        payload["errors_detail"] = err_detail
    fn = _TRAINING_FN
    train_payload = None
    if fn is not None:
        try:
            t = fn()
            if isinstance(t, dict) and t:
                train_payload = t
        except Exception:
            train_payload = None
    if train_payload is not None:
        if send_analytics:
            payload["training"] = train_payload
        else:
            # Minimal tier still reports whether a run is active so the
            # server can flip online/active, just without identifying it.
            payload["training_active"] = True
    if send_analytics:
        mh, n_models = _models_hash()
        payload["host"]               = _host_token()
        payload["os"]                 = platform.system() or ""
        payload["arch"]               = platform.machine() or ""
        payload["total_runtime_secs"] = int(float(s.get("total_runtime_secs") or 0.0) + uptime)
        payload["n_models"]           = n_models
        payload["models_hash"]        = mh
        if not bool(s.get("hw_dump_sent")):
            hw = _hw_block()
            if hw is not None:
                payload["hw"] = hw
        pend = s.get("pending_training_events") or []
        if isinstance(pend, list) and pend:
            payload["trainings"] = pend[:TRAINING_EVENTS_PER_PING_MAX]
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
    with urllib.request.urlopen(req, timeout=HEARTBEAT_TIMEOUT_SECS, context=_SSL_CTX) as resp:
        return resp.status


def _send_once():
    payload = _build_payload()
    sent_trainings = payload.get("trainings") or []
    sent_hw        = payload.get("hw") is not None
    try:
        status = _post(payload)
        patch = {
            "last_send_ts":     int(time.time()),
            "last_send_status": int(status),
            "last_send_error":  None,
        }
        if sent_hw:
            patch["hw_dump_sent"] = True
        if sent_trainings:
            remaining = (_state().get("pending_training_events") or [])[len(sent_trainings):]
            patch["pending_training_events"] = remaining
        _update_state(patch)
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


def _jittered_interval():
    spread = HEARTBEAT_INTERVAL_SECS * HEARTBEAT_JITTER_FRAC
    return HEARTBEAT_INTERVAL_SECS + random.uniform(-spread, spread)


def _training_signature():
    fn = _TRAINING_FN
    if fn is None:
        return None
    try:
        t = fn()
    except Exception:
        return None
    if not isinstance(t, dict) or not t:
        return None
    return (t.get("plugin_id"), t.get("started_at"))


def _loop():
    last_persist = time.monotonic()
    next_send    = time.monotonic() + HEARTBEAT_FIRST_DELAY
    last_train   = _training_signature()
    while True:
        time.sleep(60)
        now = time.monotonic()
        if now - last_persist >= 600:
            _accumulate_runtime(now - last_persist)
            last_persist = now
            _update_state({"last_start_ts": time.time()})
        if not _enabled():
            next_send = now + _jittered_interval()
            last_train = _training_signature()
            continue
        # Edge-triggered ping on training start/stop. Lets the server flip a
        # client between "online" and "active" within a minute instead of
        # waiting up to 6h for the next steady ping. Coalesced with the
        # steady cadence: one transition = one extra ping, not a flood.
        cur_train = _training_signature()
        if cur_train != last_train:
            last_train = cur_train
            _send_once()
            next_send = now + _jittered_interval()
            continue
        if now >= next_send:
            _send_once()
            next_send = now + _jittered_interval()


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
        "device_id":          _effective_device_id(),
        "device_id_default":  _default_device_id(),
        "device_name":        (settings_mod.get().get("device_name") or ""),
        "device_name_max":    getattr(settings_mod, "DEVICE_NAME_MAX_LEN", 15),
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
