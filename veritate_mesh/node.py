# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mesh node: registers with hub, heartbeats, long-polls jobs, executes locally.
# - one daemon worker thread. handlers are stubs in this build.
# - hub address + auth token come from runtime/settings. role gating: thread is
#   a no-op unless mesh_role is node or both and address/token are set.
# veritate_mesh/node.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

from flask import jsonify

from runtime import logs as logmod
from runtime import settings as settings_mod

from .capabilities import detect
from .client import HubClient, HubClientError
from .protocol import (
    ROLE_NODE,
    ROLE_BOTH,
)

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE         = "mesh.node"
WORKER_THREAD_NAME = "mesh-node-worker"

HEARTBEAT_INTERVAL  = 30.0
REGISTER_BACKOFF    = 10.0
POLL_LONG_SECS      = 25.0
LOOP_ERROR_BACKOFF  = 5.0
ERRORS_RING_MAX     = 10
STUB_WORK_SECS      = 2.0

ACTIVE_ROLES        = (ROLE_NODE, ROLE_BOTH)

SETTING_ROLE        = "mesh_role"
SETTING_HUB_ADDR    = "mesh_hub_address"
SETTING_AUTH_TOKEN  = "mesh_auth_token"

HTTP_UNKNOWN_NODE   = "404"

# ------------------------------------------------------------------------------------
# Module state

_STATE_LOCK = threading.Lock()
_STATE = {
    "registered":     False,
    "last_heartbeat": 0.0,
    "last_job_id":    None,
    "current_job":    None,
    "errors":         [],
}

_THREAD = None
_THREAD_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def _settings_triple():
    s = settings_mod.get()
    return (
        s.get(SETTING_ROLE)       or "",
        s.get(SETTING_HUB_ADDR)   or "",
        s.get(SETTING_AUTH_TOKEN) or "",
    )


def _node_enabled():
    role, addr, token = _settings_triple()
    return role in ACTIVE_ROLES and bool(addr) and bool(token)


def _state_set(**patch):
    with _STATE_LOCK:
        _STATE.update(patch)


def _state_snapshot():
    with _STATE_LOCK:
        return {
            "registered":     _STATE["registered"],
            "last_heartbeat": _STATE["last_heartbeat"],
            "last_job_id":    _STATE["last_job_id"],
            "current_job":    dict(_STATE["current_job"]) if _STATE["current_job"] else None,
            "errors":         list(_STATE["errors"]),
        }


def _push_error(msg):
    with _STATE_LOCK:
        errs = _STATE["errors"]
        errs.append({"ts": time.time(), "msg": str(msg)})
        while len(errs) > ERRORS_RING_MAX:
            errs.pop(0)


def _load_dict():
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        return {}
    with _STATE_LOCK:
        cur = _STATE["current_job"]
        cur_id = cur["job_id"] if cur else None
    return {"current_job": cur_id, "cpu_pct": cpu_pct, "ram_pct": ram_pct}


# ------------------------------------------------------------------------------------
# Job handlers (stubs)

def _stub_result(job):
    time.sleep(STUB_WORK_SECS)
    return {
        "status":       "stub",
        "kind":         job.kind,
        "payload_keys": list((job.payload or {}).keys()),
    }


# ------------------------------------------------------------------------------------
# Worker loop

def _build_client():
    _, addr, token = _settings_triple()
    return HubClient(addr, token)


def _try_register(client, caps):
    try:
        client.register(caps)
        _state_set(registered=True)
        logmod.info(LOG_SOURCE, f"registered with hub as {caps.node_id}")
        return True
    except HubClientError as e:
        logmod.warn(LOG_SOURCE, f"register failed: {e}")
        _push_error(f"register: {e}")
        return False


def _try_heartbeat(client, caps):
    try:
        client.heartbeat(caps.node_id, _load_dict())
        _state_set(last_heartbeat=time.time())
        return True
    except HubClientError as e:
        if HTTP_UNKNOWN_NODE in str(e).split(":", 1)[0]:
            logmod.warn(LOG_SOURCE, "hub does not know this node; re-registering")
            _state_set(registered=False)
        else:
            logmod.warn(LOG_SOURCE, f"heartbeat failed: {e}")
            _push_error(f"heartbeat: {e}")
        return False


def _try_poll(client, caps):
    try:
        return client.poll_job(caps, long_poll_secs=POLL_LONG_SECS)
    except HubClientError as e:
        logmod.warn(LOG_SOURCE, f"poll failed: {e}")
        _push_error(f"poll: {e}")
        return None


def _run_job(client, job):
    _state_set(current_job={
        "job_id":     job.job_id,
        "kind":       job.kind,
        "started_at": time.time(),
    })
    try:
        result = _stub_result(job)
        try:
            client.report_result(job.job_id, result)
        except HubClientError as e:
            logmod.warn(LOG_SOURCE, f"report_result failed: {e}")
            _push_error(f"report_result: {e}")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logmod.error(LOG_SOURCE, f"job {job.job_id} failed: {msg}")
        _push_error(f"job {job.job_id}: {msg}")
        try:
            client.report_failure(job.job_id, msg)
        except HubClientError as ee:
            logmod.warn(LOG_SOURCE, f"report_failure failed: {ee}")
            _push_error(f"report_failure: {ee}")
    finally:
        _state_set(current_job=None, last_job_id=job.job_id)


def _worker_loop():
    caps = detect()
    next_heartbeat = 0.0
    while True:
        try:
            if not _node_enabled():
                time.sleep(LOOP_ERROR_BACKOFF)
                continue

            with _STATE_LOCK:
                registered = _STATE["registered"]

            if not registered:
                caps = detect()
                if not _try_register(_build_client(), caps):
                    time.sleep(REGISTER_BACKOFF)
                    continue
                next_heartbeat = 0.0

            client = _build_client()
            now = time.time()
            if now >= next_heartbeat:
                _try_heartbeat(client, caps)
                next_heartbeat = time.time() + HEARTBEAT_INTERVAL

            with _STATE_LOCK:
                still_registered = _STATE["registered"]
            if not still_registered:
                continue

            job = _try_poll(client, caps)
            if job is None:
                continue
            _run_job(client, job)
        except Exception as e:
            logmod.error(LOG_SOURCE, f"worker loop error: {type(e).__name__}: {e}")
            _push_error(f"loop: {type(e).__name__}: {e}")
            time.sleep(LOOP_ERROR_BACKOFF)


# ------------------------------------------------------------------------------------
# Public API

def register(app):
    """register node-side flask routes onto the existing app."""
    @app.route("/mesh/node/status")
    def mesh_node_status():
        role, _, _ = _settings_triple()
        snap = _state_snapshot()
        snap["role"] = role
        return jsonify(snap)


def start_workers():
    """spawn the worker daemon thread. idempotent. no-op when role is off or
    hub address/token are unset."""
    global _THREAD
    if not _node_enabled():
        return
    with _THREAD_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        t = threading.Thread(target=_worker_loop, name=WORKER_THREAD_NAME, daemon=True)
        t.start()
        _THREAD = t
        logmod.info(LOG_SOURCE, "node worker started")
