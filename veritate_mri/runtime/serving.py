# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - serving-activity beacon: the one live signal saying a generation is in flight
#   right now. The experience log wraps every serving stream on both backends and
#   both routes, so marking it there covers the whole surface with one owner.
# - background work reads this to get out of the way. The sleep controller
#   suspends its trainer child while `active()` holds and resumes once `idle_s()
#   clears its quiet window: on a weak box the sleep child takes every core, and
#   an unyielding consolidation run costs a served request 2.5-3x throughput and
#   ~200x first-byte latency (measured, i7-9700T 8c @ 800 MHz).
# - counts concurrent streams rather than a bool: two overlapping requests must
#   not have the first one to finish declare the box quiet.
# veritate_mri/runtime/serving.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

# ------------------------------------------------------------------------------------
# Constants

_LOCK = threading.Lock()
_STATE = {"active": 0, "last_end": 0.0}
# callbacks fired when a generation starts. Background work registers here so the
# dependency runs runtime -> consumer, never inference -> training.
_HOOKS = []

# ------------------------------------------------------------------------------------
# Functions


def on_began(callback):
    """Register a callback fired when a generation starts. Used by the sleep
    controller to park its trainer child for the duration of the request."""
    with _LOCK:
        if callback not in _HOOKS:
            _HOOKS.append(callback)


def began():
    """Mark one generation as started and notify anything that must yield."""
    with _LOCK:
        _STATE["active"] += 1
        hooks = list(_HOOKS)
    for cb in hooks:
        try:
            cb()
        except Exception:
            pass   # serving must not fail because a background yield did


def ended():
    """Mark one generation as finished."""
    with _LOCK:
        _STATE["active"] = max(0, _STATE["active"] - 1)
        if _STATE["active"] == 0:
            _STATE["last_end"] = time.monotonic()


def active():
    """True while at least one generation is streaming."""
    with _LOCK:
        return _STATE["active"] > 0


def idle_s():
    """Seconds since the last generation ended; None while one is streaming or
    when nothing has been served yet."""
    with _LOCK:
        if _STATE["active"] > 0 or not _STATE["last_end"]:
            return None
        return time.monotonic() - _STATE["last_end"]


def reset():
    """Drop all activity state and hooks. Tests only."""
    with _LOCK:
        _STATE["active"] = 0
        _STATE["last_end"] = 0.0
        _HOOKS.clear()
