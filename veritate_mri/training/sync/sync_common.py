# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared three-state sync engine used by plugins_sync.py and models_sync.py.
# - the world view: for each remote file, compare three things — local-on-disk,
#   last-synced (snapshot of what we wrote OR what existed at first adoption),
#   and what remote has now. that yields four user-visible states (plus a fifth
#   for the conflict case where both moved).
# - never overwrites silently. action policy is explicit at the API boundary.
# veritate_mri/sync/sync_common.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import time

# ------------------------------------------------------------------------------------
# Constants

STATE_FILE_NAME = ".sync_state.json"
STATE_FILE_VERSION = 1
SHA_CHUNK = 1024 * 1024  # 1 MB read buffer for streaming sha

# Per-file classification:
#   "missing"          - file is in remote but not on disk
#   "current"          - local matches remote (no action needed)
#   "update_available" - local matches last-sync but remote moved
#   "modified"         - local differs from last-sync, remote unchanged
#   "conflict"         - local differs from last-sync AND remote moved
#   "orphan"           - locally tracked file no longer in remote
STATE_MISSING          = "missing"
STATE_CURRENT          = "current"
STATE_UPDATE_AVAILABLE = "update_available"
STATE_MODIFIED         = "modified"
STATE_CONFLICT         = "conflict"
STATE_ORPHAN           = "orphan"

ACTION_INSTALL  = "install"   # write remote bytes (only valid for missing)
ACTION_UPDATE   = "update"    # overwrite local with remote, record new state
ACTION_FORCE    = "force"     # overwrite even when modified/conflict
ACTION_ADOPT    = "adopt"     # keep local as-is, record current SHA as the baseline
ACTION_SKIP     = "skip"      # do nothing this round

VALID_ACTIONS = {ACTION_INSTALL, ACTION_UPDATE, ACTION_FORCE, ACTION_ADOPT, ACTION_SKIP}


# ------------------------------------------------------------------------------------
# State file I/O

def state_path(root_dir):
    return os.path.join(root_dir, STATE_FILE_NAME)


def load_state(root_dir):
    """Return {rel_path: {synced_sha, synced_at, remote_branch}}. Empty dict on
    missing/corrupt file."""
    p = state_path(root_dir)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict): return {}
    if data.get("version") != STATE_FILE_VERSION: return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def save_state(root_dir, files, remote_branch=None):
    """Atomically replace .sync_state.json with the given file map."""
    p = state_path(root_dir)
    tmp = p + ".tmp"
    payload = {
        "version":        STATE_FILE_VERSION,
        "last_sync_at":   time.time(),
        "remote_branch":  remote_branch or "",
        "files":          dict(files),
    }
    os.makedirs(root_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------
# Hashing

def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path):
    """Streaming sha256 of a file on disk. Returns None if file doesn't exist."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1_file(path):
    """Streaming git-blob-sha1 of a file on disk. Returns None if file doesn't
    exist. Format: sha1 of "blob {size}\\0{content}", matching what the GitHub
    trees API returns in the `sha` field."""
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------
# Path + walk helpers

def safe_dest(rel_path, root):
    """Reject path traversal: returns the absolute path only if it stays inside
    root. Returns None if rel_path escapes (`..`) or resolves to root itself."""
    candidate = os.path.normpath(os.path.join(root, rel_path))
    root_norm = os.path.normpath(root)
    if candidate == root_norm: return None
    if not candidate.startswith(root_norm + os.sep): return None
    return candidate


def count_local_files(root):
    """Count files under root, skipping the sync state file. Returns 0 if root
    does not exist."""
    if not os.path.isdir(root): return 0
    n = 0
    for _root, _dirs, files in os.walk(root):
        for fn in files:
            if fn == STATE_FILE_NAME: continue
            n += 1
    return n


def summarize_states(rows):
    """Counts of state values across the rows produced by classify_set."""
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return counts


# ------------------------------------------------------------------------------------
# Classification

def classify_one(local_path, remote_sha, state_entry, sha_fn=sha256_file):
    """Return (state, local_sha) for a single file.

    - local_path:   absolute path on disk
    - remote_sha:   hash of remote content in the SAME format as sha_fn produces
                    (None if remote unknown / not in remote)
    - state_entry:  dict from load_state()[rel_path] or None if untracked
    - sha_fn:       callable(path) -> hex digest. Default sha256_file. Pass
                    git_blob_sha1_file when the remote uses GitHub's tree-API
                    `sha` field.
    """
    local_exists = os.path.isfile(local_path)
    local_sha    = sha_fn(local_path) if local_exists else None
    last_sha     = state_entry.get("synced_sha") if isinstance(state_entry, dict) else None

    if not local_exists:
        if remote_sha is None:
            # tracked but gone from both local and remote -> orphan
            return (STATE_ORPHAN, None) if last_sha else (STATE_MISSING, None)
        return (STATE_MISSING, None)

    if remote_sha is None:
        # local exists, remote doesn't have it (or wasn't passed) -> orphan
        return (STATE_ORPHAN, local_sha)

    if local_sha == remote_sha:
        return (STATE_CURRENT, local_sha)

    if last_sha is None:
        # never tracked — adopt mode treats local as the baseline. Until the user
        # confirms an update, flag as modified so they see remote drift.
        return (STATE_MODIFIED, local_sha)

    if local_sha == last_sha:
        # user has not touched since last sync; remote moved
        return (STATE_UPDATE_AVAILABLE, local_sha)

    # user modified locally
    if remote_sha == last_sha:
        # only local moved
        return (STATE_MODIFIED, local_sha)
    # both moved
    return (STATE_CONFLICT, local_sha)


def classify_set(root_dir, remote_files, state, sha_fn=sha256_file):
    """remote_files: dict {rel_path: remote_sha}. Returns:
        [{
          "path": rel_path,
          "state": STATE_*,
          "local_sha":  hex or None,
          "remote_sha": hex or None,
          "synced_sha": hex or None,
        }, ...]
    plus the same for any orphans (tracked or local files no longer in remote).
    sha_fn: see classify_one. Must match the hash format of remote_files values
    and of any synced_sha already recorded in state.
    """
    out = []
    seen = set()
    for rel, rsha in remote_files.items():
        seen.add(rel)
        st, lsha = classify_one(os.path.join(root_dir, rel), rsha, state.get(rel), sha_fn=sha_fn)
        out.append({
            "path":       rel,
            "state":      st,
            "local_sha":  lsha,
            "remote_sha": rsha,
            "synced_sha": (state.get(rel) or {}).get("synced_sha"),
        })
    # tracked-but-missing-from-remote (orphans the user owns via state file)
    for rel, entry in state.items():
        if rel in seen: continue
        st, lsha = classify_one(os.path.join(root_dir, rel), None, entry, sha_fn=sha_fn)
        out.append({
            "path":       rel,
            "state":      st,
            "local_sha":  lsha,
            "remote_sha": None,
            "synced_sha": entry.get("synced_sha"),
        })
    out.sort(key=lambda r: r["path"])
    return out


# ------------------------------------------------------------------------------------
# Default action policy

def default_action_for_state(file_state):
    """The action the dashboard applies under the 'sync all safe' bulk button.
    Anything modified/conflict requires an explicit per-file decision."""
    return {
        STATE_MISSING:          ACTION_INSTALL,
        STATE_UPDATE_AVAILABLE: ACTION_UPDATE,
        STATE_CURRENT:          ACTION_SKIP,
        STATE_MODIFIED:         ACTION_SKIP,
        STATE_CONFLICT:         ACTION_SKIP,
        STATE_ORPHAN:           ACTION_SKIP,
    }.get(file_state, ACTION_SKIP)


def action_is_destructive(action, file_state):
    """Returns True if the action would overwrite user changes that haven't been
    explicitly accepted. Used by callers to require an extra confirmation."""
    if action == ACTION_FORCE: return True
    if action == ACTION_UPDATE and file_state in (STATE_MODIFIED, STATE_CONFLICT):
        return True
    return False
