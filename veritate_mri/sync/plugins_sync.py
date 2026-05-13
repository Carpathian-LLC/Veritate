# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - downloads plugin files from the public Veritate-Plugins repo over plain
#   https. plugins/ is NOT a git repo locally; this is a one-way file pull.
# - three-state per-file sync (see sync_common.py): missing / current /
#   update_available / modified / conflict. only "safe" actions run by default
#   (install missing, update clean-but-outdated). modified and conflict files
#   require an explicit per-file action from the caller (force, adopt, skip).
# - tarball mode: plugin files are small enough to fit one full tarball in
#   memory. avoids per-file http round-trips. for large model files see
#   models_sync.py.
# - .sync_state.json lives at plugins/.sync_state.json and stores the SHA of
#   each file as written by the last successful sync. compared at every
#   check() to detect user edits.
# - refuses to sync while a plugin is running.
# veritate_mri/sync/plugins_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import os
import ssl
import tarfile
import threading
import time
import urllib.error
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

from readers import paths
from runtime import logs as logmod
from training import plugin_runner
from . import sync_common as sc

# ------------------------------------------------------------------------------------
# Constants

REPO_OWNER         = "Carpathian-LLC"
REPO_NAME          = "Veritate-Plugins"
DEFAULT_BRANCH     = "main"
DOWNLOAD_TIMEOUT_S = 120

DEFAULT_REMOTE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

PLUGINS_DIR = paths.PLUGINS_ROOT

_LOCK = threading.RLock()
# Per-process cache of the most recent remote-file map {rel: {"sha":..., "bytes":...}}
# populated by check(). sync() reuses it if fresh, else re-fetches. ttl 5 min.
_REMOTE_CACHE = {"files": {}, "fetched_at": 0.0, "branch": ""}
_REMOTE_CACHE_TTL_S = 300
_LAST = {"ok": None, "message": "", "finished_at": None, "action": None}


# ------------------------------------------------------------------------------------
# HTTP

def _tarball_url(branch):
    return f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/tar.gz/refs/heads/{branch}"


def _download_tarball(branch):
    url = _tarball_url(branch)
    req = urllib.request.Request(url, headers={"User-Agent": "veritate-mri/sync"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S, context=_SSL_CTX) as resp:
        code = getattr(resp, "status", 200)
        if code != 200:
            raise RuntimeError(f"http {code} from {url}")
        return resp.read()


def _safe_dest(rel_path, root):
    candidate = os.path.normpath(os.path.join(root, rel_path))
    root_norm = os.path.normpath(root)
    if candidate == root_norm: return None
    if not candidate.startswith(root_norm + os.sep): return None
    return candidate


def _strip_top_dir(name):
    parts = name.split("/", 1)
    if len(parts) < 2 or not parts[1]: return None
    return parts[1]


def _parse_tarball(data):
    """Return dict {rel_path: (sha256, bytes)} for every file in the tarball.
    Bytes are held in memory — plugin tarballs are small (< 5 MB typically)."""
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile(): continue
            rel = _strip_top_dir(member.name)
            if rel is None: continue
            if _safe_dest(rel, PLUGINS_DIR) is None: continue
            src = tar.extractfile(member)
            if src is None: continue
            content = src.read()
            files[rel] = (sc.sha256_bytes(content), content)
    return files


def _fetch_remote_files(branch, use_cache=True):
    """Returns {rel: (sha, bytes)}. Caches across calls within TTL."""
    now = time.time()
    if use_cache and _REMOTE_CACHE["branch"] == branch \
            and (now - _REMOTE_CACHE["fetched_at"]) < _REMOTE_CACHE_TTL_S \
            and _REMOTE_CACHE["files"]:
        return _REMOTE_CACHE["files"]
    data = _download_tarball(branch)
    files = _parse_tarball(data)
    _REMOTE_CACHE["files"]      = files
    _REMOTE_CACHE["fetched_at"] = now
    _REMOTE_CACHE["branch"]     = branch
    return files


# ------------------------------------------------------------------------------------
# Bookkeeping

def _record(action, ok, message):
    with _LOCK:
        _LAST.update({
            "ok":          bool(ok),
            "message":     message,
            "finished_at": time.time(),
            "action":      action,
        })


def _count_local_files():
    if not os.path.isdir(PLUGINS_DIR): return 0
    n = 0
    for _root, _dirs, files in os.walk(PLUGINS_DIR):
        for fn in files:
            if fn == sc.STATE_FILE_NAME: continue
            n += 1
    return n


def _summarize_states(rows):
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return counts


# ------------------------------------------------------------------------------------
# Public API

def status():
    """Lightweight status — does not hit the network. Returns last known sync info."""
    state = sc.load_state(PLUGINS_DIR)
    return {
        "exists":             os.path.isdir(PLUGINS_DIR),
        "remote_url":         DEFAULT_REMOTE_URL,
        "default_remote_url": DEFAULT_REMOTE_URL,
        "default_branch":     DEFAULT_BRANCH,
        "local_files":        _count_local_files(),
        "tracked_files":      len(state),
        "last":               dict(_LAST),
    }


def files():
    """Network-touching status: fetches remote tree, classifies every file.
    Returns the full per-file table for the dashboard."""
    branch = DEFAULT_BRANCH
    try:
        remote = _fetch_remote_files(branch)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
        msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
        logmod.error("plugins-sync", msg)
        _record("files", False, msg)
        return {"ok": False, "error": msg, "status": status()}

    remote_sha_only = {rel: sha for rel, (sha, _b) in remote.items()}
    state = sc.load_state(PLUGINS_DIR)
    rows  = sc.classify_set(PLUGINS_DIR, remote_sha_only, state)
    counts = _summarize_states(rows)
    msg = (f"remote {len(remote_sha_only)} file(s) "
           f"({counts.get(sc.STATE_MISSING, 0)} missing, "
           f"{counts.get(sc.STATE_UPDATE_AVAILABLE, 0)} updates, "
           f"{counts.get(sc.STATE_MODIFIED, 0)} modified, "
           f"{counts.get(sc.STATE_CONFLICT, 0)} conflicts)")
    logmod.ok("plugins-sync", msg)
    _record("files", True, msg)
    return {
        "ok":        True,
        "action":    "files",
        "branch":    branch,
        "remote_url": DEFAULT_REMOTE_URL,
        "files":     rows,
        "counts":    counts,
        "status":    status(),
    }


def check():
    """Same as files() but reports only the count summary. Kept for compatibility
    with the existing dashboard banner."""
    r = files()
    if not r.get("ok"): return r
    c = r["counts"]
    return {
        "ok":           True,
        "action":       "check",
        "remote_files": sum(c.values()),
        "new_files":    c.get(sc.STATE_MISSING, 0)
                      + c.get(sc.STATE_UPDATE_AVAILABLE, 0),
        "modified":     c.get(sc.STATE_MODIFIED, 0) + c.get(sc.STATE_CONFLICT, 0),
        "files":        r["files"],
        "counts":       c,
        "status":       status(),
    }


def sync(actions=None, branch=None):
    """Apply per-file actions. `actions` is a dict {rel_path: ACTION_*}.

    If `actions` is None, the default safe policy is applied: install missing,
    update clean-but-outdated, skip modified and conflicts. The dashboard's
    "Sync All Safe" button uses this; a per-row dialog passes an explicit dict
    when the user wants to force or adopt.
    """
    branch = branch or DEFAULT_BRANCH

    with _LOCK:
        if plugin_runner.is_running():
            msg = "a plugin is currently running. stop it before syncing."
            logmod.warn("plugins-sync", msg)
            _record("sync", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        try:
            remote = _fetch_remote_files(branch, use_cache=False)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
            msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
            logmod.error("plugins-sync", msg)
            _record("sync", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        remote_sha_only = {rel: sha for rel, (sha, _b) in remote.items()}
        state = sc.load_state(PLUGINS_DIR)
        rows  = sc.classify_set(PLUGINS_DIR, remote_sha_only, state)

        # Resolve action per file.
        resolved = {}
        for r in rows:
            rel = r["path"]
            wanted = (actions or {}).get(rel)
            if wanted is None:
                wanted = sc.default_action_for_state(r["state"])
            if wanted not in sc.VALID_ACTIONS:
                wanted = sc.ACTION_SKIP
            resolved[rel] = wanted

        # Execute. Track results per file.
        results = {"installed": [], "updated": [], "forced": [], "adopted": [],
                   "skipped": [], "errors": []}
        os.makedirs(PLUGINS_DIR, exist_ok=True)

        for r in rows:
            rel = r["path"]
            st  = r["state"]
            act = resolved[rel]

            if act == sc.ACTION_SKIP:
                results["skipped"].append({"path": rel, "state": st})
                continue

            if act == sc.ACTION_ADOPT:
                # Record current local SHA as the baseline. No file write.
                lsha = r["local_sha"] or sc.sha256_file(os.path.join(PLUGINS_DIR, rel))
                if lsha is None:
                    results["errors"].append({"path": rel, "error": "cannot adopt: file missing"})
                    continue
                state[rel] = {
                    "synced_sha":    lsha,
                    "synced_at":     time.time(),
                    "remote_branch": branch,
                    "via":           "adopt",
                }
                results["adopted"].append({"path": rel})
                continue

            if act == sc.ACTION_INSTALL and st != sc.STATE_MISSING:
                # install only valid for missing; otherwise demote to update
                act = sc.ACTION_UPDATE

            if act in (sc.ACTION_INSTALL, sc.ACTION_UPDATE, sc.ACTION_FORCE):
                if rel not in remote:
                    results["errors"].append({"path": rel, "error": "not in remote"})
                    continue
                # Refuse non-force overwrite of modified/conflict files.
                if act != sc.ACTION_FORCE and st in (sc.STATE_MODIFIED, sc.STATE_CONFLICT):
                    results["skipped"].append({
                        "path": rel, "state": st,
                        "reason": "locally modified; requires force",
                    })
                    continue
                dest = _safe_dest(rel, PLUGINS_DIR)
                if dest is None:
                    results["errors"].append({"path": rel, "error": "unsafe path"})
                    continue
                try:
                    parent = os.path.dirname(dest)
                    if parent: os.makedirs(parent, exist_ok=True)
                    sha, content = remote[rel]
                    with open(dest, "wb") as f:
                        f.write(content)
                except OSError as e:
                    results["errors"].append({"path": rel, "error": f"write failed: {e}"})
                    continue
                state[rel] = {
                    "synced_sha":    sha,
                    "synced_at":     time.time(),
                    "remote_branch": branch,
                    "via":           act,
                }
                if   act == sc.ACTION_INSTALL: results["installed"].append({"path": rel})
                elif act == sc.ACTION_FORCE:   results["forced"].append({"path": rel, "was": st})
                else:                          results["updated"].append({"path": rel})

        # Persist new state and write a summary log line.
        sc.save_state(PLUGINS_DIR, state, remote_branch=branch)
        n_inst = len(results["installed"])
        n_upd  = len(results["updated"])
        n_frc  = len(results["forced"])
        n_adp  = len(results["adopted"])
        n_skp  = len(results["skipped"])
        n_err  = len(results["errors"])
        msg = (f"installed {n_inst}, updated {n_upd}, forced {n_frc}, "
               f"adopted {n_adp}, skipped {n_skp}, errors {n_err}")
        if n_err:
            logmod.warn("plugins-sync", msg)
            _record("sync", False, msg)
        else:
            logmod.ok("plugins-sync", msg)
            _record("sync", True, msg)

        return {
            "ok":         n_err == 0,
            "action":     "sync",
            "results":    results,
            "downloaded": n_inst + n_upd + n_frc,
            "skipped":    n_skp,
            "status":     status(),
            # back-compat fields the old dashboard banner reads:
            "new_files":  n_inst + n_upd,
        }
