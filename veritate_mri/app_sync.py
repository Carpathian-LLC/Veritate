# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sync the parent Veritate repo against its remote git origin. one of three
#   release channels selects which branch tracks the upstream:
#       stable        -> main
#       experimental  -> experimental
#       development   -> dev
# - status() reports current branch, channel, head, and behind count vs the
#   channel's branch on origin. check() runs git fetch and refreshes behind.
#   pull() does ff-only pull on the channel branch. switch_channel() runs a
#   safe git checkout (refuses if working tree is dirty).
# - poll_loop() runs in the background and writes the latest behind count to
#   data/app_sync_state.json. the dashboard polls /app/update_status to read
#   it and (optionally) auto-triggers a soft reload when an update lands.
# veritate_mri/app_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import threading
import time

from readers import paths

import logs as logmod
import settings as settings_mod
from git_runner import run_git as _git

# ------------------------------------------------------------------------------------
# Constants

REPO_DIR    = paths.REPO_ROOT
STATE_PATH  = os.path.join(REPO_DIR, "data", "app_sync_state.json")
GIT_TIMEOUT_SECS = 60

CHANNEL_STABLE       = "stable"
CHANNEL_EXPERIMENTAL = "experimental"
CHANNEL_DEVELOPMENT  = "development"

CHANNEL_BRANCHES = {
    CHANNEL_STABLE:       "main",
    CHANNEL_EXPERIMENTAL: "experimental",
    CHANNEL_DEVELOPMENT:  "dev",
}
BRANCH_TO_CHANNEL = {v: k for k, v in CHANNEL_BRANCHES.items()}
ALL_CHANNELS = (CHANNEL_STABLE, CHANNEL_EXPERIMENTAL, CHANNEL_DEVELOPMENT)

POLL_INTERVAL_SECS = 30 * 60
POLL_FIRST_DELAY   = 60

_LOCK   = threading.RLock()
_STATE_CACHE = None
_THREAD = None
_RELOAD_HOOK = None

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


def _run_git(args, timeout=GIT_TIMEOUT_SECS):
    return _git(args, REPO_DIR, timeout=timeout)


def _is_repo():
    return os.path.isdir(os.path.join(REPO_DIR, ".git"))


def _channel():
    s = settings_mod.get()
    ch = s.get("update_channel")
    return ch if ch in ALL_CHANNELS else CHANNEL_STABLE


def _channel_branch():
    return CHANNEL_BRANCHES[_channel()]


def _is_dirty():
    code, so, _ = _run_git(["status", "--porcelain"], timeout=15)
    if code != 0:
        return None
    return bool(so.strip())


def _current_branch():
    code, so, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
    return so if code == 0 else None


def _head_short():
    code, so, _ = _run_git(["rev-parse", "--short", "HEAD"], timeout=10)
    return so if code == 0 else None


def _remote_url():
    code, so, _ = _run_git(["remote", "get-url", "origin"], timeout=10)
    return so if code == 0 else None


def _ahead_behind(branch):
    code, so, _ = _run_git(
        ["rev-list", "--left-right", "--count", f"HEAD...origin/{branch}"],
        timeout=15,
    )
    if code != 0 or not so:
        return None, None
    parts = so.split()
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def status():
    out = {
        "is_repo":         _is_repo(),
        "channel":         _channel(),
        "channel_branch":  _channel_branch(),
        "channels":        list(ALL_CHANNELS),
        "channel_map":     dict(CHANNEL_BRANCHES),
        "branch":          None,
        "head_short":      None,
        "remote_url":      None,
        "behind":          None,
        "ahead":           None,
        "dirty":           None,
        "update_available": False,
        "last": _state(),
    }
    if not out["is_repo"]:
        return out
    out["branch"]     = _current_branch()
    out["head_short"] = _head_short()
    out["remote_url"] = _remote_url()
    out["dirty"]      = _is_dirty()
    if out["branch"]:
        a, b = _ahead_behind(out["branch"])
        out["ahead"]  = a
        out["behind"] = b
        out["update_available"] = bool(b and b > 0)
    return out


def check():
    if not _is_repo():
        msg = "veritate repo .git not found"
        logmod.warn("app-sync", msg)
        _update_state({"last_check_ts": time.time(), "last_check_ok": False, "last_check_msg": msg})
        return {"ok": False, "error": msg, "status": status()}
    code, so, se = _run_git(["fetch", "origin", "--prune"], timeout=GIT_TIMEOUT_SECS)
    if code != 0:
        msg = se or so or f"git fetch exit {code}"
        logmod.error("app-sync", f"fetch failed: {msg}")
        _update_state({"last_check_ts": time.time(), "last_check_ok": False, "last_check_msg": msg})
        return {"ok": False, "error": msg, "status": status()}
    correction_msg = ""
    cur = _current_branch()
    if cur and cur in BRANCH_TO_CHANNEL:
        mapped = BRANCH_TO_CHANNEL[cur]
        if mapped != _channel():
            settings_mod.update({"update_channel": mapped})
            correction_msg = f"channel auto-switched to {mapped} (matches branch {cur!r})"
            logmod.ok("app-sync", correction_msg)
    st = status()
    _update_state({
        "last_check_ts":  time.time(),
        "last_check_ok":  True,
        "last_check_msg": correction_msg,
        "behind":         st.get("behind"),
        "remote_branch":  st.get("channel_branch"),
    })
    return {"ok": True, "status": status()}


def pull():
    if not _is_repo():
        return {"ok": False, "error": "not a git repo"}
    cur = _current_branch()
    target = _channel_branch()
    if cur != target:
        msg = (f"current branch is {cur!r}; channel expects {target!r}. "
               f"switch channel first.")
        logmod.warn("app-sync", msg)
        return {"ok": False, "error": msg}
    if _is_dirty():
        msg = "working tree has uncommitted changes; refusing to pull. commit or stash first."
        logmod.warn("app-sync", msg)
        return {"ok": False, "error": msg}
    code, so, se = _run_git(["pull", "--ff-only", "origin", target])
    if code != 0:
        msg = se or so or f"git pull --ff-only exit {code}"
        logmod.error("app-sync", f"pull failed: {msg}")
        _update_state({"last_pull_ts": time.time(), "last_pull_ok": False, "last_pull_msg": msg})
        return {"ok": False, "error": msg}
    logmod.ok("app-sync", f"pulled origin/{target}: {so or 'already up to date'}")
    _update_state({"last_pull_ts": time.time(), "last_pull_ok": True, "last_pull_msg": so or "ok"})
    return {"ok": True, "status": status()}


def switch_channel(channel):
    if channel not in ALL_CHANNELS:
        return {"ok": False, "error": f"unknown channel: {channel}"}
    target = CHANNEL_BRANCHES[channel]
    if not _is_repo():
        settings_mod.update({"update_channel": channel})
        return {"ok": True, "status": status()}
    cur = _current_branch()
    if cur == target:
        settings_mod.update({"update_channel": channel})
        return {"ok": True, "status": status()}
    if _is_dirty():
        msg = "working tree has uncommitted changes; refusing to switch branch. commit or stash first."
        logmod.warn("app-sync", msg)
        return {"ok": False, "error": msg}
    code, so, se = _run_git(["fetch", "origin", target], timeout=GIT_TIMEOUT_SECS)
    if code != 0:
        msg = se or so or f"git fetch exit {code}"
        logmod.error("app-sync", f"channel fetch failed: {msg}")
        return {"ok": False, "error": msg}
    code, so, se = _run_git(["checkout", target])
    if code != 0:
        msg = se or so or f"git checkout exit {code}"
        logmod.error("app-sync", f"channel checkout failed: {msg}")
        return {"ok": False, "error": msg}
    settings_mod.update({"update_channel": channel})
    logmod.ok("app-sync", f"switched channel to {channel} (branch {target})")
    return {"ok": True, "status": status()}


def set_reload_hook(fn):
    """Caller registers a parameterless callable that triggers a soft reload.
    poll_loop calls it when an update lands and auto_reload_on_update is on."""
    global _RELOAD_HOOK
    _RELOAD_HOOK = fn


def _maybe_auto_reload(behind_before, behind_after):
    if behind_after is None or behind_before is None:
        return
    if behind_after >= behind_before:
        return
    s = settings_mod.get()
    if not s.get("auto_reload_on_update"):
        return
    if _RELOAD_HOOK is None:
        return
    logmod.warn("app-sync", "auto-reload triggered after update")
    try:
        _RELOAD_HOOK()
    except Exception as e:
        logmod.error("app-sync", f"auto reload failed: {e}")


def _poll_loop():
    time.sleep(POLL_FIRST_DELAY)
    while True:
        try:
            check()
        except Exception as e:
            logmod.error("app-sync", f"poll error: {e}")
        time.sleep(POLL_INTERVAL_SECS)


def start():
    global _THREAD
    if _THREAD is not None:
        return
    t = threading.Thread(target=_poll_loop, name="app-sync-poll", daemon=True)
    t.start()
    _THREAD = t
    logmod.info("app-sync", f"channel={_channel()} branch={_current_branch()}")
