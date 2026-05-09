# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sync the plugins/ folder against its remote git repo. plugins/ is its own
#   self-contained git repo, separate from the parent Veritate repo.
# - status() reports remote, branch, head, and ahead/behind. local dirty state
#   is not surfaced: pull is fast-forward-only, so git itself rejects pulls
#   that would overwrite tracked edits, and untracked files are expected.
# - sync() does fetch + ff-only pull, or clones if plugins/ is missing.
# - never destructive: refuses to clone over a non-empty non-repo dir, refuses
#   to pull if a plugin is currently running, refuses non-fast-forward merges.
# veritate_mri/plugins_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import subprocess
import threading
import time

from readers import paths

import logs as logmod
import plugin_runner

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_REMOTE_URL = "https://github.com/Carpathian-LLC/Veritate-Plugins.git"
DEFAULT_BRANCH     = "main"
GIT_TIMEOUT_SECS   = 120

PLUGINS_DIR = paths.PLUGINS_ROOT
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

_LOCK = threading.RLock()
_LAST = {
    "ok":         None,
    "message":    "",
    "finished_at": None,
    "action":     None,
}

# ------------------------------------------------------------------------------------
# Functions

def _run_git(args, cwd, timeout=GIT_TIMEOUT_SECS):
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL":  "https",
        "GIT_ASKPASS":         "echo",
    }
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            creationflags=_NO_WINDOW,
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "git executable not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"git {' '.join(args)} timed out after {timeout}s"


def _is_repo(path):
    return os.path.isdir(os.path.join(path, ".git"))


def _dir_is_empty(path):
    if not os.path.isdir(path):
        return True
    for entry in os.listdir(path):
        if entry == ".git":
            continue
        return False
    return True


def status():
    out = {
        "exists":     os.path.isdir(PLUGINS_DIR),
        "is_repo":    False,
        "remote_url": None,
        "branch":     None,
        "head_sha":   None,
        "head_short": None,
        "behind":     None,
        "ahead":      None,
        "default_remote_url": DEFAULT_REMOTE_URL,
        "default_branch":     DEFAULT_BRANCH,
        "last": dict(_LAST),
    }
    if not out["exists"]:
        return out
    if not _is_repo(PLUGINS_DIR):
        return out
    out["is_repo"] = True

    code, so, _ = _run_git(["remote", "get-url", "origin"], PLUGINS_DIR, timeout=10)
    if code == 0:
        out["remote_url"] = so

    code, so, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], PLUGINS_DIR, timeout=10)
    if code == 0:
        out["branch"] = so or None

    code, so, _ = _run_git(["rev-parse", "HEAD"], PLUGINS_DIR, timeout=10)
    if code == 0 and so:
        out["head_sha"]   = so
        out["head_short"] = so[:8]

    if out["branch"] and out["remote_url"]:
        code, so, _ = _run_git(
            ["rev-list", "--left-right", "--count", f"HEAD...origin/{out['branch']}"],
            PLUGINS_DIR, timeout=10,
        )
        if code == 0 and so:
            parts = so.split()
            if len(parts) == 2:
                try:
                    out["ahead"]  = int(parts[0])
                    out["behind"] = int(parts[1])
                except ValueError:
                    pass

    return out


def _record(action, ok, message):
    with _LOCK:
        _LAST.update({
            "ok":          bool(ok),
            "message":     message,
            "finished_at": time.time(),
            "action":      action,
        })


def _clone(remote_url, branch):
    if os.path.isdir(PLUGINS_DIR) and not _dir_is_empty(PLUGINS_DIR):
        msg = f"refuse to clone: {PLUGINS_DIR} is non-empty and not a git repo. move or delete it first."
        logmod.error("plugins-sync", msg)
        _record("clone", False, msg)
        return {"ok": False, "error": msg}
    parent = os.path.dirname(PLUGINS_DIR)
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(PLUGINS_DIR) and _dir_is_empty(PLUGINS_DIR):
        try:
            shutil.rmtree(PLUGINS_DIR)
        except OSError as e:
            msg = f"could not remove empty {PLUGINS_DIR}: {e}"
            logmod.error("plugins-sync", msg)
            _record("clone", False, msg)
            return {"ok": False, "error": msg}
    logmod.info("plugins-sync", f"cloning {remote_url} (branch {branch}) into {PLUGINS_DIR}")
    code, so, se = _run_git(
        ["clone", "--branch", branch, "--single-branch", remote_url, PLUGINS_DIR],
        cwd=parent,
        timeout=GIT_TIMEOUT_SECS,
    )
    if code != 0:
        msg = se or so or f"git clone exit {code}"
        logmod.error("plugins-sync", f"clone failed: {msg}")
        _record("clone", False, msg)
        return {"ok": False, "error": msg}
    logmod.ok("plugins-sync", "clone done")
    _record("clone", True, f"cloned {remote_url}@{branch}")
    return {"ok": True, "action": "clone", "status": status()}


def _pull(branch):
    code, so, se = _run_git(["pull", "--ff-only", "origin", branch], PLUGINS_DIR)
    if code != 0:
        msg = se or so or f"git pull --ff-only exit {code}"
        logmod.error("plugins-sync", f"pull failed: {msg}")
        _record("pull", False, msg)
        return {"ok": False, "error": msg}
    logmod.ok("plugins-sync", f"pulled origin/{branch}: {so or 'already up to date'}")
    _record("pull", True, so or "already up to date")
    return {"ok": True, "action": "pull", "status": status()}


def check():
    """Refresh remote state without pulling. `git fetch origin <branch>` then
    return the updated status. Read-only; does not mutate the working tree."""
    if not _is_repo(PLUGINS_DIR):
        return {"ok": False, "error": "plugins/ is not a git repo. clone via sync first.",
                "status": status()}
    branch = DEFAULT_BRANCH
    code, so, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], PLUGINS_DIR, timeout=10)
    if code == 0 and so:
        branch = so
    code, so, se = _run_git(["fetch", "origin", branch], PLUGINS_DIR)
    if code != 0:
        msg = se or so or f"git fetch exit {code}"
        logmod.error("plugins-sync", f"check failed: {msg}")
        _record("check", False, msg)
        return {"ok": False, "error": msg, "status": status()}
    _record("check", True, f"fetched origin/{branch}")
    return {"ok": True, "action": "check", "status": status()}


def sync():
    remote_url = DEFAULT_REMOTE_URL
    branch     = DEFAULT_BRANCH

    with _LOCK:
        if plugin_runner.is_running():
            msg = "a plugin is currently running. stop it before syncing."
            logmod.warn("plugins-sync", msg)
            return {"ok": False, "error": msg}

        if not _is_repo(PLUGINS_DIR):
            return _clone(remote_url, branch)

        code, so, _ = _run_git(["remote", "get-url", "origin"], PLUGINS_DIR, timeout=10)
        current_remote = so if code == 0 else None
        if current_remote and current_remote != remote_url:
            msg = (f"origin remote is {current_remote!r}, refusing to pull from {remote_url!r}. "
                   f"reconfigure the remote manually if intentional.")
            logmod.error("plugins-sync", msg)
            _record("pull", False, msg)
            return {"ok": False, "error": msg}

        return _pull(branch)
