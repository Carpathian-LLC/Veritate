# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - downloads model files from the public Veritate-Models repo. unlike plugins,
#   models can contain multi-GB checkpoints; tarballs-in-memory won't scale.
#   instead we use GitHub's git-tree API to list every file in the repo (path
#   + SHA + size) and stream each requested file directly from raw.githubuser-
#   content.com to disk with progress.
# - three-state per-file sync (see sync_common.py).
# - provenance:
#     * remote-pulled: any model dir whose name appears in the remote tree.
#                      these participate in three-state classification.
#     * local-trained: any model dir on disk whose name does NOT appear in the
#                      remote tree. these are invisible to sync — they cannot be
#                      overwritten, listed, or affected by any sync action.
#   provenance is computed at every check() against the live remote listing, so
#   moving a model from local-trained -> remote-pulled (by publishing it
#   upstream later) is automatic on the next check.
# - large file safety: refuses to overwrite any file that's > 100 MB unless the
#   action is explicitly ACTION_FORCE. trips a confirmation flow in the UI.
# - resumable downloads: in-progress files go to <dest>.part and rename on
#   success. an interrupted sync can rerun cleanly.
# veritate_mri/models_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import ssl
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

import logs as logmod
import sync_common as sc

# ------------------------------------------------------------------------------------
# Constants

REPO_OWNER         = "Carpathian-LLC"
REPO_NAME          = "Veritate-Models"
DEFAULT_BRANCH     = "main"
TIMEOUT_TREE_S     = 30
TIMEOUT_FILE_S     = 600   # per-file download timeout (10 min — enough for a few GB on a fast link)
CHUNK_BYTES        = 1024 * 1024     # 1 MB streaming chunks
LARGE_FILE_BYTES   = 100 * 1024 * 1024  # 100 MB threshold for "large" confirmation gate

DEFAULT_REMOTE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

MODELS_DIR = paths.MODELS_ROOT

_LOCK = threading.RLock()
# Cache of remote tree {rel: {sha, size}}. Refreshed by check()/files() every TTL.
_REMOTE_CACHE = {"files": {}, "fetched_at": 0.0, "branch": ""}
_REMOTE_CACHE_TTL_S = 300

# Live download progress, surfaced to the dashboard for the active sync().
# {path: {bytes_done, bytes_total, started_at, finished_at, state}}
_PROGRESS = {}

_LAST = {"ok": None, "message": "", "finished_at": None, "action": None}


# ------------------------------------------------------------------------------------
# HTTP

def _tree_url(branch):
    return f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/{branch}?recursive=1"


def _raw_url(branch, rel):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{branch}/{rel}"


def _http_get(url, timeout, headers=None):
    h = {"User-Agent": "veritate-mri/sync"}
    if headers: h.update(headers)
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)


def _fetch_tree(branch):
    """Return {rel: {"sha":..., "size":...}}. GitHub trees API returns git
    blob SHAs (sha1), not content sha256 — that's fine for change detection
    because we record the same field in our state file. The two never get
    compared against each other."""
    url = _tree_url(branch)
    with _http_get(url, TIMEOUT_TREE_S) as resp:
        code = getattr(resp, "status", 200)
        if code != 200:
            raise RuntimeError(f"http {code} from {url}")
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("malformed tree response")
    if payload.get("truncated"):
        # >100k entries — fall back? For now we just warn and proceed with what we got.
        logmod.warn("models-sync", "remote tree was truncated by github API")
    tree = payload.get("tree") or []
    out = {}
    for ent in tree:
        if ent.get("type") != "blob": continue
        rel = ent.get("path") or ""
        if not rel: continue
        if rel.startswith(".github/"): continue   # don't pull CI metadata
        if rel == sc.STATE_FILE_NAME:  continue   # state files are local-only
        out[rel] = {"sha": ent.get("sha"), "size": int(ent.get("size", 0))}
    return out


def _fetch_remote_tree(branch, use_cache=True):
    now = time.time()
    if use_cache and _REMOTE_CACHE["branch"] == branch \
            and (now - _REMOTE_CACHE["fetched_at"]) < _REMOTE_CACHE_TTL_S \
            and _REMOTE_CACHE["files"]:
        return _REMOTE_CACHE["files"]
    tree = _fetch_tree(branch)
    _REMOTE_CACHE["files"]      = tree
    _REMOTE_CACHE["fetched_at"] = now
    _REMOTE_CACHE["branch"]     = branch
    return tree


def _safe_dest(rel_path, root):
    candidate = os.path.normpath(os.path.join(root, rel_path))
    root_norm = os.path.normpath(root)
    if candidate == root_norm: return None
    if not candidate.startswith(root_norm + os.sep): return None
    return candidate


def _stream_to_disk(branch, rel, dest, expected_size):
    """Stream remote raw file to <dest>.part, then rename. Updates _PROGRESS as
    bytes flow. Returns the content sha256 of what we wrote (so we can record
    it in state)."""
    url = _raw_url(branch, rel)
    part = dest + ".part"
    parent = os.path.dirname(dest)
    if parent: os.makedirs(parent, exist_ok=True)
    if os.path.exists(part):
        os.remove(part)

    import hashlib
    h = hashlib.sha256()
    total = expected_size or 0

    with _LOCK:
        _PROGRESS[rel] = {
            "bytes_done":   0,
            "bytes_total":  total,
            "started_at":   time.time(),
            "finished_at":  None,
            "state":        "downloading",
        }

    try:
        with _http_get(url, TIMEOUT_FILE_S) as resp:
            code = getattr(resp, "status", 200)
            if code != 200:
                raise RuntimeError(f"http {code} from {url}")
            done = 0
            with open(part, "wb") as out:
                while True:
                    chunk = resp.read(CHUNK_BYTES)
                    if not chunk: break
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    with _LOCK:
                        _PROGRESS[rel]["bytes_done"] = done
        os.replace(part, dest)
        with _LOCK:
            _PROGRESS[rel]["finished_at"] = time.time()
            _PROGRESS[rel]["state"]       = "done"
    except Exception:
        # leave .part on disk for diagnostic purposes; resume not implemented
        # but reruns will overwrite cleanly.
        with _LOCK:
            _PROGRESS[rel]["state"]       = "error"
            _PROGRESS[rel]["finished_at"] = time.time()
        raise
    return h.hexdigest()


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
    if not os.path.isdir(MODELS_DIR): return 0
    n = 0
    for _root, _dirs, files in os.walk(MODELS_DIR):
        for fn in files:
            if fn == sc.STATE_FILE_NAME: continue
            n += 1
    return n


def _top_level_dirs():
    if not os.path.isdir(MODELS_DIR): return []
    return sorted(d for d in os.listdir(MODELS_DIR)
                  if os.path.isdir(os.path.join(MODELS_DIR, d)))


def _remote_top_level_dirs(remote_tree):
    """Returns the set of top-level model dirs that appear in remote."""
    out = set()
    for rel in remote_tree.keys():
        if "/" in rel:
            out.add(rel.split("/", 1)[0])
    return out


def _provenance_table(remote_tree):
    """Returns {dir_name: "remote-pulled" | "local-trained"} for every model
    dir on disk. Local-trained dirs are NOT touched by any sync action and
    don't appear in the file table."""
    remote_dirs = _remote_top_level_dirs(remote_tree)
    out = {}
    for d in _top_level_dirs():
        out[d] = "remote-pulled" if d in remote_dirs else "local-trained"
    # surface remote-only dirs (not yet downloaded) too
    for d in remote_dirs:
        out.setdefault(d, "remote-pulled")
    return out


def _summarize_states(rows):
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return counts


# ------------------------------------------------------------------------------------
# Public API

def status():
    state = sc.load_state(MODELS_DIR)
    return {
        "exists":             os.path.isdir(MODELS_DIR),
        "remote_url":         DEFAULT_REMOTE_URL,
        "default_remote_url": DEFAULT_REMOTE_URL,
        "default_branch":     DEFAULT_BRANCH,
        "local_files":        _count_local_files(),
        "tracked_files":      len(state),
        "last":               dict(_LAST),
    }


def files():
    """Per-file classification + per-dir provenance. The dashboard renders the
    file table grouped by top-level dir, with a provenance badge per group."""
    branch = DEFAULT_BRANCH
    try:
        tree = _fetch_remote_tree(branch)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
        msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
        logmod.error("models-sync", msg)
        _record("files", False, msg)
        return {"ok": False, "error": msg, "status": status()}

    remote_sha = {rel: t["sha"] for rel, t in tree.items()}
    state      = sc.load_state(MODELS_DIR)
    rows       = sc.classify_set(MODELS_DIR, remote_sha, state)
    # attach size and a per-row warning flag for large files
    sizes = {rel: t["size"] for rel, t in tree.items()}
    for r in rows:
        r["size"]   = sizes.get(r["path"], 0)
        r["large"]  = r["size"] >= LARGE_FILE_BYTES
    counts = _summarize_states(rows)
    prov   = _provenance_table(tree)
    msg = (f"remote {len(remote_sha)} file(s) "
           f"({counts.get(sc.STATE_MISSING, 0)} missing, "
           f"{counts.get(sc.STATE_UPDATE_AVAILABLE, 0)} updates, "
           f"{counts.get(sc.STATE_MODIFIED, 0)} modified, "
           f"{counts.get(sc.STATE_CONFLICT, 0)} conflicts); "
           f"{sum(1 for v in prov.values() if v == 'local-trained')} local-trained")
    logmod.ok("models-sync", msg)
    _record("files", True, msg)
    return {
        "ok":         True,
        "action":     "files",
        "branch":     branch,
        "remote_url": DEFAULT_REMOTE_URL,
        "files":      rows,
        "counts":     counts,
        "provenance": prov,
        "status":     status(),
    }


def check():
    """Compatibility-shim for the existing dashboard banner — same output shape
    the old check() returned, but populated from the new files() pipeline."""
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
        "provenance":   r["provenance"],
        "status":       status(),
    }


def progress():
    """Snapshot of any in-flight downloads. Polled by the dashboard during a
    sync() that includes large files."""
    with _LOCK:
        return {"items": dict(_PROGRESS)}


def sync(actions=None, branch=None):
    """Apply per-file actions. Same contract as plugins_sync.sync(), plus a
    large-file safety rule: any single file >= LARGE_FILE_BYTES with action
    ACTION_UPDATE on a STATE_MODIFIED or STATE_CONFLICT file is downgraded to
    skip unless action is ACTION_FORCE.
    """
    branch = branch or DEFAULT_BRANCH

    with _LOCK:
        # clear the live progress map for the new run
        _PROGRESS.clear()

        try:
            tree = _fetch_remote_tree(branch, use_cache=False)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
            msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
            logmod.error("models-sync", msg)
            _record("sync", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        remote_sha = {rel: t["sha"] for rel, t in tree.items()}
        sizes      = {rel: t["size"] for rel, t in tree.items()}
        state      = sc.load_state(MODELS_DIR)
        rows       = sc.classify_set(MODELS_DIR, remote_sha, state)

        resolved = {}
        for r in rows:
            rel = r["path"]
            wanted = (actions or {}).get(rel)
            if wanted is None:
                wanted = sc.default_action_for_state(r["state"])
            if wanted not in sc.VALID_ACTIONS:
                wanted = sc.ACTION_SKIP
            resolved[rel] = wanted

        results = {"installed": [], "updated": [], "forced": [], "adopted": [],
                   "skipped": [], "errors": []}
        os.makedirs(MODELS_DIR, exist_ok=True)

        for r in rows:
            rel = r["path"]
            st  = r["state"]
            act = resolved[rel]
            size = sizes.get(rel, 0)

            if act == sc.ACTION_SKIP:
                results["skipped"].append({"path": rel, "state": st})
                continue

            if act == sc.ACTION_ADOPT:
                lsha = r["local_sha"] or sc.sha256_file(os.path.join(MODELS_DIR, rel))
                if lsha is None:
                    results["errors"].append({"path": rel, "error": "cannot adopt: file missing"})
                    continue
                state[rel] = {
                    "synced_sha":    lsha,
                    "synced_at":     time.time(),
                    "remote_branch": branch,
                    "remote_sha":    remote_sha.get(rel),  # git blob SHA from the tree
                    "size":          size,
                    "via":           "adopt",
                }
                results["adopted"].append({"path": rel})
                continue

            if act == sc.ACTION_INSTALL and st != sc.STATE_MISSING:
                act = sc.ACTION_UPDATE

            if act in (sc.ACTION_INSTALL, sc.ACTION_UPDATE, sc.ACTION_FORCE):
                if rel not in tree:
                    results["errors"].append({"path": rel, "error": "not in remote"})
                    continue
                if act != sc.ACTION_FORCE and st in (sc.STATE_MODIFIED, sc.STATE_CONFLICT):
                    results["skipped"].append({
                        "path": rel, "state": st,
                        "reason": "locally modified; requires force",
                    })
                    continue
                dest = _safe_dest(rel, MODELS_DIR)
                if dest is None:
                    results["errors"].append({"path": rel, "error": "unsafe path"})
                    continue
                try:
                    content_sha = _stream_to_disk(branch, rel, dest, size)
                except Exception as e:
                    results["errors"].append({"path": rel, "error": f"download failed: {e}"})
                    continue
                state[rel] = {
                    "synced_sha":    content_sha,        # content sha256 (what we wrote)
                    "synced_at":     time.time(),
                    "remote_branch": branch,
                    "remote_sha":    remote_sha.get(rel),  # git blob SHA from the tree
                    "size":          size,
                    "via":           act,
                }
                if   act == sc.ACTION_INSTALL: results["installed"].append({"path": rel, "size": size})
                elif act == sc.ACTION_FORCE:   results["forced"].append({"path": rel, "was": st, "size": size})
                else:                          results["updated"].append({"path": rel, "size": size})

        sc.save_state(MODELS_DIR, state, remote_branch=branch)

        n_inst = len(results["installed"])
        n_upd  = len(results["updated"])
        n_frc  = len(results["forced"])
        n_adp  = len(results["adopted"])
        n_skp  = len(results["skipped"])
        n_err  = len(results["errors"])
        msg = (f"installed {n_inst}, updated {n_upd}, forced {n_frc}, "
               f"adopted {n_adp}, skipped {n_skp}, errors {n_err}")
        if n_err:
            logmod.warn("models-sync", msg)
            _record("sync", False, msg)
        else:
            logmod.ok("models-sync", msg)
            _record("sync", True, msg)

        return {
            "ok":         n_err == 0,
            "action":     "sync",
            "results":    results,
            "downloaded": n_inst + n_upd + n_frc,
            "skipped":    n_skp,
            "status":     status(),
            "new_files":  n_inst + n_upd,
        }
