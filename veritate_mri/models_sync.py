# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sync the models/ folder against its remote git repo. models/ is its own
#   self-contained git repo, separate from the parent Veritate repo. ships
#   barebones manifests only; users add their own checkpoints and hooks on top.
# - status() reports remote, branch, head, and ahead/behind. local dirty state
#   is intentionally not surfaced: users will accumulate untracked checkpoints
#   and hooks, and that is expected.
# - sync() does fetch + hard reset to origin/<branch>, or clones if models/ is
#   missing. tracked files are forced to match upstream (this is a download,
#   not a merge), so any local commits or edits to tracked files are discarded.
#   untracked user content (checkpoints, hooks) is preserved. this avoids the
#   "diverging branches can't be fast-forwarded" failure mode that ff-only
#   pulls hit whenever local history drifts from remote.
# veritate_mri/models_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import threading
import time

from readers import paths

import logs as logmod
from git_runner import run_git as _git

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_REMOTE_URL = "https://github.com/Carpathian-LLC/Veritate-Models.git"
DEFAULT_BRANCH     = "main"
GIT_TIMEOUT_SECS   = 120

MODELS_DIR = paths.MODELS_ROOT

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
    return _git(args, cwd, timeout=timeout)


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
        "exists":     os.path.isdir(MODELS_DIR),
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
    if not _is_repo(MODELS_DIR):
        return out
    out["is_repo"] = True

    code, so, _ = _run_git(["remote", "get-url", "origin"], MODELS_DIR, timeout=10)
    if code == 0:
        out["remote_url"] = so

    code, so, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], MODELS_DIR, timeout=10)
    if code == 0:
        out["branch"] = so or None

    code, so, _ = _run_git(["rev-parse", "HEAD"], MODELS_DIR, timeout=10)
    if code == 0 and so:
        out["head_sha"]   = so
        out["head_short"] = so[:8]

    if out["branch"] and out["remote_url"]:
        code, so, _ = _run_git(
            ["rev-list", "--left-right", "--count", f"HEAD...origin/{out['branch']}"],
            MODELS_DIR, timeout=10,
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
    if os.path.isdir(MODELS_DIR) and not _dir_is_empty(MODELS_DIR):
        msg = f"refuse to clone: {MODELS_DIR} is non-empty and not a git repo. move or delete it first."
        logmod.error("models-sync", msg)
        _record("clone", False, msg)
        return {"ok": False, "error": msg}
    parent = os.path.dirname(MODELS_DIR)
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(MODELS_DIR) and _dir_is_empty(MODELS_DIR):
        try:
            shutil.rmtree(MODELS_DIR)
        except OSError as e:
            msg = f"could not remove empty {MODELS_DIR}: {e}"
            logmod.error("models-sync", msg)
            _record("clone", False, msg)
            return {"ok": False, "error": msg}
    logmod.info("models-sync", f"cloning {remote_url} (branch {branch}) into {MODELS_DIR}")
    code, so, se = _run_git(
        ["clone", "--branch", branch, "--single-branch", remote_url, MODELS_DIR],
        cwd=parent,
        timeout=GIT_TIMEOUT_SECS,
    )
    if code != 0:
        msg = se or so or f"git clone exit {code}"
        logmod.error("models-sync", f"clone failed: {msg}")
        _record("clone", False, msg)
        return {"ok": False, "error": msg}
    logmod.ok("models-sync", "clone done")
    _record("clone", True, f"cloned {remote_url}@{branch}")
    return {"ok": True, "action": "clone", "status": status()}


def _pull(branch):
    code, so, se = _run_git(["fetch", "origin", branch], MODELS_DIR)
    if code != 0:
        msg = se or so or f"git fetch exit {code}"
        logmod.error("models-sync", f"fetch failed: {msg}")
        _record("pull", False, msg)
        return {"ok": False, "error": msg}
    code, so, se = _run_git(["reset", "--hard", f"origin/{branch}"], MODELS_DIR)
    if code != 0:
        msg = se or so or f"git reset --hard exit {code}"
        logmod.error("models-sync", f"reset failed: {msg}")
        _record("pull", False, msg)
        return {"ok": False, "error": msg}
    logmod.ok("models-sync", f"synced to origin/{branch}: {so or 'ok'}")
    _record("pull", True, so or f"synced to origin/{branch}")
    return {"ok": True, "action": "pull", "status": status()}


def check():
    """Refresh remote state without pulling. `git fetch origin <branch>` then
    return the updated status. Read-only; does not mutate the working tree."""
    if not _is_repo(MODELS_DIR):
        return {"ok": False, "error": "models/ is not a git repo. clone via sync first.",
                "status": status()}
    branch = DEFAULT_BRANCH
    code, so, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], MODELS_DIR, timeout=10)
    if code == 0 and so:
        branch = so
    code, so, se = _run_git(["fetch", "origin", branch], MODELS_DIR)
    if code != 0:
        msg = se or so or f"git fetch exit {code}"
        logmod.error("models-sync", f"check failed: {msg}")
        _record("check", False, msg)
        return {"ok": False, "error": msg, "status": status()}
    _record("check", True, f"fetched origin/{branch}")
    return {"ok": True, "action": "check", "status": status()}


def sync():
    remote_url = DEFAULT_REMOTE_URL
    branch     = DEFAULT_BRANCH

    with _LOCK:
        if not _is_repo(MODELS_DIR):
            return _clone(remote_url, branch)

        code, so, _ = _run_git(["remote", "get-url", "origin"], MODELS_DIR, timeout=10)
        current_remote = so if code == 0 else None
        if current_remote and current_remote != remote_url:
            msg = (f"origin remote is {current_remote!r}, refusing to pull from {remote_url!r}. "
                   f"reconfigure the remote manually if intentional.")
            logmod.error("models-sync", msg)
            _record("pull", False, msg)
            return {"ok": False, "error": msg}

        return _pull(branch)
