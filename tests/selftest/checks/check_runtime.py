# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_mri/runtime/* import + the cold entry points (sys_metrics snapshot,
#   logs snapshot, settings load, heartbeat status, lifecycle import).
# tests/selftest/checks/check_runtime.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA           = "platform"
RUNTIME_MODS   = ("sys_metrics", "logs", "settings", "heartbeat", "lifecycle", "ai_assist")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """runtime modules import; sys_metrics.snapshot, logs.snapshot, settings,
    heartbeat status all return without raising."""
    miss = []
    for name in RUNTIME_MODS:
        try:
            __import__("runtime." + name)
        except Exception as exc:
            miss.append(f"{name}: {exc}")
    if miss:
        return _status.fail("runtime", miss[0], {"errors": miss})

    from runtime import sys_metrics, logs as logmod, heartbeat
    snap = sys_metrics.snapshot()
    if not isinstance(snap, dict):
        return _status.fail("runtime", f"sys_metrics.snapshot returned {type(snap).__name__}")
    log_snap = logmod.snapshot() if hasattr(logmod, "snapshot") else None
    if log_snap is None:
        return _status.fail("runtime", "logs.snapshot missing")
    if hasattr(heartbeat, "status"):
        heartbeat.status()
    return _status.ok("runtime", f"{len(RUNTIME_MODS)} runtime modules ok",
                      {"sys_metric_keys": list(snap)[:6]})
