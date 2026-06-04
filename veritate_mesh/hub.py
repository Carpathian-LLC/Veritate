# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mesh hub: coordinator routes + background sweeper. lazy singleton registry + queue.
# - auth: bearer token from runtime/settings.py::mesh_auth_token.
# - long-poll /mesh/job/next holds open up to JOB_NEXT_LONG_POLL_MAX seconds.
# veritate_mesh/hub.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time
import uuid

from dataclasses import asdict

from flask import Response, request

from runtime import logs as logmod
from runtime import settings as settings_mod

from .protocol import (
    Capabilities,
    Job,
    JobRequirements,
    JOB_STATUS_PENDING,
    VALID_JOB_KINDS,
    PROTOCOL_VERSION,
)
from .registry import NodeRegistry, JobQueue

# ------------------------------------------------------------------------------------
# Constants

JOB_NEXT_LONG_POLL_MAX = 30.0
JOB_NEXT_POLL_INTERVAL = 0.5
NODE_STALE_TTL         = 90.0
SWEEPER_INTERVAL       = 15.0

AUTH_HEADER       = "Authorization"
AUTH_SCHEME       = "Bearer "
AUTH_TOKEN_KEY    = "mesh_auth_token"

LOG_SOURCE = "mesh.hub"

ERR_UNAUTHORIZED  = {"ok": False, "error": "unauthorized"}
ERR_NO_TOKEN_CFG  = {"ok": False, "error": "hub token not configured"}
ERR_UNKNOWN_NODE  = {"ok": False, "error": "unknown node"}
ERR_UNKNOWN_JOB   = {"ok": False, "error": "unknown job"}
ERR_BAD_BODY      = {"ok": False, "error": "bad body"}
ERR_BAD_KIND      = {"ok": False, "error": "invalid job kind"}

# ------------------------------------------------------------------------------------
# Module state

_REGISTRY: NodeRegistry | None = None
_QUEUE:    JobQueue     | None = None

_singleton_lock = threading.Lock()
_workers_started = False
_workers_lock = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def get_registry() -> NodeRegistry:
    """lazy singleton."""
    global _REGISTRY
    with _singleton_lock:
        if _REGISTRY is None:
            _REGISTRY = NodeRegistry()
        return _REGISTRY


def get_queue() -> JobQueue:
    """lazy singleton."""
    global _QUEUE
    with _singleton_lock:
        if _QUEUE is None:
            _QUEUE = JobQueue()
        return _QUEUE


def _configured_token() -> str:
    return (settings_mod.get().get(AUTH_TOKEN_KEY) or "").strip()


def _require_auth():
    token = _configured_token()
    if not token:
        return (ERR_NO_TOKEN_CFG, 503)
    header = request.headers.get(AUTH_HEADER, "")
    if not header.startswith(AUTH_SCHEME):
        return (ERR_UNAUTHORIZED, 401)
    presented = header[len(AUTH_SCHEME):].strip()
    if presented != token:
        return (ERR_UNAUTHORIZED, 401)
    return None


def _clamp_long_poll(raw) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > JOB_NEXT_LONG_POLL_MAX:
        return JOB_NEXT_LONG_POLL_MAX
    return val


def register(app):
    """register hub-side flask routes onto the existing app."""

    @app.route("/mesh/register", methods=["POST"])
    def mesh_register_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return (ERR_BAD_BODY, 400)
        try:
            caps = Capabilities.from_dict(body)
        except Exception as e:
            logmod.warn(LOG_SOURCE, f"register parse failed: {e}")
            return (ERR_BAD_BODY, 400)
        get_registry().register(caps)
        logmod.info(LOG_SOURCE, f"node registered: {caps.node_id} ({caps.hostname})")
        return {"ok": True, "node_id": caps.node_id, "protocol_version": PROTOCOL_VERSION}

    @app.route("/mesh/heartbeat", methods=["POST"])
    def mesh_heartbeat_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        body = request.get_json(silent=True) or {}
        node_id = body.get("node_id")
        load = body.get("load") or {}
        if not node_id:
            return (ERR_BAD_BODY, 400)
        ok = get_registry().heartbeat(node_id, load)
        if not ok:
            return (ERR_UNKNOWN_NODE, 404)
        return {"ok": True, "directives": {}}

    @app.route("/mesh/job/next", methods=["GET"])
    def mesh_job_next_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        node_id = request.args.get("node_id", "")
        if not node_id:
            return (ERR_BAD_BODY, 400)
        entry = get_registry().get(node_id)
        if entry is None:
            return (ERR_UNKNOWN_NODE, 404)
        caps = entry["caps"]
        budget = _clamp_long_poll(request.args.get("long_poll", 0.0))
        deadline = time.time() + budget
        queue = get_queue()
        while True:
            job = queue.claim_for(caps)
            if job is not None:
                logmod.info(LOG_SOURCE, f"job {job.job_id} -> {node_id}")
                return asdict(job)
            if time.time() >= deadline:
                return Response("", status=204)
            time.sleep(JOB_NEXT_POLL_INTERVAL)

    @app.route("/mesh/job/<job_id>/progress", methods=["POST"])
    def mesh_job_progress_route(job_id):
        auth = _require_auth()
        if auth is not None:
            return auth
        body = request.get_json(silent=True) or {}
        progress = body.get("progress") or {}
        queue = get_queue()
        if queue.get(job_id) is None:
            return (ERR_UNKNOWN_JOB, 404)
        queue.mark_running(job_id)
        queue.update_progress(job_id, progress)
        return {"ok": True}

    @app.route("/mesh/job/<job_id>/result", methods=["POST"])
    def mesh_job_result_route(job_id):
        auth = _require_auth()
        if auth is not None:
            return auth
        body = request.get_json(silent=True) or {}
        queue = get_queue()
        if queue.get(job_id) is None:
            return (ERR_UNKNOWN_JOB, 404)
        if "error" in body and body.get("error"):
            queue.mark_failed(job_id, str(body.get("error")))
            logmod.warn(LOG_SOURCE, f"job {job_id} failed")
        else:
            queue.mark_done(job_id, body.get("result") or {})
            logmod.info(LOG_SOURCE, f"job {job_id} done")
        return {"ok": True}

    @app.route("/mesh/hub/nodes", methods=["GET"])
    def mesh_hub_nodes_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        nodes = [
            {"caps": asdict(e["caps"]), "last_seen": e["last_seen"], "load": e["load"]}
            for e in get_registry().list_all()
        ]
        return {"ok": True, "nodes": nodes}

    @app.route("/mesh/hub/jobs", methods=["GET"])
    def mesh_hub_jobs_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        return {"ok": True, "jobs": [asdict(j) for j in get_queue().list_all()]}

    @app.route("/mesh/hub/submit", methods=["POST"])
    def mesh_hub_submit_route():
        auth = _require_auth()
        if auth is not None:
            return auth
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return (ERR_BAD_BODY, 400)
        kind = body.get("kind")
        if kind not in VALID_JOB_KINDS:
            return (ERR_BAD_KIND, 400)
        req_raw = body.get("requirements") or {}
        if not isinstance(req_raw, dict):
            return (ERR_BAD_BODY, 400)
        allowed = set(JobRequirements.__dataclass_fields__.keys())
        req_kwargs = {k: v for k, v in req_raw.items() if k in allowed}
        try:
            requirements = JobRequirements(**req_kwargs)
        except TypeError as e:
            logmod.warn(LOG_SOURCE, f"submit requirements bad: {e}")
            return (ERR_BAD_BODY, 400)
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            return (ERR_BAD_BODY, 400)
        job_id = str(uuid.uuid4())
        job = Job(
            job_id       = job_id,
            kind         = kind,
            payload      = payload,
            requirements = requirements,
            status       = JOB_STATUS_PENDING,
        )
        get_queue().submit(job)
        logmod.info(LOG_SOURCE, f"job submitted: {job_id} kind={kind}")
        return {"ok": True, "job_id": job_id}


def _sweeper_loop():
    while True:
        try:
            registry = get_registry()
            queue = get_queue()
            dropped = registry.expire_stale(NODE_STALE_TTL)
            if dropped:
                logmod.warn(LOG_SOURCE, f"stale nodes dropped: {dropped}")
                known = {e["caps"].node_id for e in registry.list_all()}
                requeued = queue.requeue_orphaned(known)
                if requeued:
                    logmod.warn(LOG_SOURCE, f"orphan jobs requeued: {requeued}")
        except Exception as e:
            logmod.error(LOG_SOURCE, f"sweeper tick failed: {e}")
        time.sleep(SWEEPER_INTERVAL)


def start_workers():
    """spawn background daemon thread for sweeper. idempotent."""
    global _workers_started
    with _workers_lock:
        if _workers_started:
            return
        t = threading.Thread(target=_sweeper_loop, name="mesh-hub-sweeper", daemon=True)
        t.start()
        _workers_started = True
        logmod.info(LOG_SOURCE, "sweeper started")
