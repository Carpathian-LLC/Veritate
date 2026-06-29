# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - hub-side in-memory registry: nodes + heartbeats and a job queue.
# - thread-safe via one lock per class. no fine-grained locking, no persistence.
# - claim_for picks oldest pending whose requirements match caller capabilities.
# veritate_mesh/registry.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

from .protocol import (
    Capabilities,
    Job,
    JobRequirements,
    capabilities_satisfy,
    JOB_STATUS_PENDING,
    JOB_STATUS_ASSIGNED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_DONE,
    JOB_STATUS_FAILED,
)

# ------------------------------------------------------------------------------------
# Constants

ACTIVE_JOB_STATUSES = (JOB_STATUS_ASSIGNED, JOB_STATUS_RUNNING)

# ------------------------------------------------------------------------------------
# Node registry

def _entry_view(entry: dict) -> dict:
    return {
        "caps":      entry["caps"],
        "last_seen": entry["last_seen"],
        "load":      dict(entry["load"]),
    }


class NodeRegistry:
    """thread-safe in-memory node + heartbeat state. hub-side only."""

    def __init__(self):
        self._lock = threading.Lock()
        self._nodes = {}  # node_id -> {"caps", "last_seen", "load"}

    def register(self, caps: Capabilities) -> None:
        now = time.time()
        with self._lock:
            prev = self._nodes.get(caps.node_id)
            load = prev["load"] if prev else {}
            self._nodes[caps.node_id] = {
                "caps":      caps,
                "last_seen": now,
                "load":      load,
            }

    def heartbeat(self, node_id: str, load: dict) -> bool:
        """update last_seen + live load. returns False if node_id is unknown."""
        now = time.time()
        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                return False
            entry["last_seen"] = now
            entry["load"] = dict(load) if load else {}
            return True

    def expire_stale(self, ttl_secs: float) -> list:
        """drop nodes whose last_seen is older than ttl. returns dropped node_ids."""
        cutoff = time.time() - ttl_secs
        dropped = []
        with self._lock:
            for node_id, entry in list(self._nodes.items()):
                if entry["last_seen"] < cutoff:
                    del self._nodes[node_id]
                    dropped.append(node_id)
        return dropped

    def get(self, node_id: str) -> dict | None:
        """returns {"caps", "last_seen", "load"} or None."""
        with self._lock:
            entry = self._nodes.get(node_id)
            return None if entry is None else _entry_view(entry)

    def list_all(self) -> list:
        """returns list of {"caps", "last_seen", "load"} dicts."""
        with self._lock:
            return [_entry_view(entry) for entry in self._nodes.values()]


# ------------------------------------------------------------------------------------
# Job queue

class JobQueue:
    """thread-safe in-memory job queue. hub-side only."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}  # job_id -> Job

    def submit(self, job: Job) -> str:
        """enqueue. returns job_id."""
        with self._lock:
            self._jobs[job.job_id] = job
            return job.job_id

    def claim_for(self, caps: Capabilities) -> Job | None:
        """oldest pending whose requirements match caps. marks ASSIGNED."""
        with self._lock:
            pending = [j for j in self._jobs.values() if j.status == JOB_STATUS_PENDING]
            pending.sort(key=lambda j: j.created_at)
            for job in pending:
                if capabilities_satisfy(job.requirements, caps):
                    job.status = JOB_STATUS_ASSIGNED
                    job.assigned_to = caps.node_id
                    return job
            return None

    def mark_running(self, job_id: str) -> bool:
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.status = JOB_STATUS_RUNNING
            if job.started_at is None:
                job.started_at = now
            return True

    def update_progress(self, job_id: str, progress: dict) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.progress = dict(progress) if progress else {}
            return True

    def mark_done(self, job_id: str, result: dict) -> bool:
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.status = JOB_STATUS_DONE
            job.result = dict(result) if result else {}
            job.finished_at = now
            return True

    def mark_failed(self, job_id: str, error: str) -> bool:
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.status = JOB_STATUS_FAILED
            job.error = error or ""
            job.finished_at = now
            return True

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self, status_in: list | None = None) -> list:
        with self._lock:
            if status_in is None:
                return list(self._jobs.values())
            allowed = set(status_in)
            return [j for j in self._jobs.values() if j.status in allowed]

    def requeue_orphaned(self, known_node_ids: set) -> int:
        """reset ASSIGNED/RUNNING jobs whose node vanished. returns count requeued."""
        n = 0
        with self._lock:
            for job in self._jobs.values():
                if job.status in ACTIVE_JOB_STATUSES and job.assigned_to not in known_node_ids:
                    job.status = JOB_STATUS_PENDING
                    job.assigned_to = None
                    n += 1
        return n
