# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - downloads plugin files from the public Veritate-Plugins repo over plain
#   https. plugins/ is NOT a git repo locally; this is a one-way file pull.
# - additive only: any file that already exists locally is left alone. only
#   files that are missing locally get written. user authored plugins and
#   local edits are never touched.
# - if plugins/ does not exist, it is created.
# - refuses to sync while a plugin is running.
# veritate_mri/plugins_sync.py
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

import logs as logmod
import plugin_runner

# ------------------------------------------------------------------------------------
# Constants

REPO_OWNER         = "Carpathian-LLC"
REPO_NAME          = "Veritate-Plugins"
DEFAULT_BRANCH     = "main"
DOWNLOAD_TIMEOUT_S = 120

DEFAULT_REMOTE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

PLUGINS_DIR = paths.PLUGINS_ROOT

_LOCK = threading.RLock()
_LAST = {
    "ok":         None,
    "message":    "",
    "finished_at": None,
    "action":     None,
}

# ------------------------------------------------------------------------------------
# Functions

def _tarball_url(branch):
    return f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/tar.gz/refs/heads/{branch}"


def _count_local_files():
    if not os.path.isdir(PLUGINS_DIR):
        return 0
    n = 0
    for _root, _dirs, files in os.walk(PLUGINS_DIR):
        n += len(files)
    return n


def _record(action, ok, message):
    with _LOCK:
        _LAST.update({
            "ok":          bool(ok),
            "message":     message,
            "finished_at": time.time(),
            "action":      action,
        })


def status():
    return {
        "exists":             os.path.isdir(PLUGINS_DIR),
        "remote_url":         DEFAULT_REMOTE_URL,
        "default_remote_url": DEFAULT_REMOTE_URL,
        "default_branch":     DEFAULT_BRANCH,
        "local_files":        _count_local_files(),
        "last":               dict(_LAST),
    }


def _download_tarball(branch):
    url = _tarball_url(branch)
    req = urllib.request.Request(url, headers={"User-Agent": "veritate-mri/sync"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S, context=_SSL_CTX) as resp:
        code = getattr(resp, "status", 200)
        if code != 200:
            raise RuntimeError(f"http {code} from {url}")
        return resp.read()


def _safe_dest(rel_path, root):
    """Return the absolute destination path if it stays under root, else None."""
    candidate = os.path.normpath(os.path.join(root, rel_path))
    root_norm = os.path.normpath(root)
    if candidate == root_norm:
        return None
    if not candidate.startswith(root_norm + os.sep):
        return None
    return candidate


def _strip_top_dir(member_name):
    parts = member_name.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    return parts[1]


def check():
    branch = DEFAULT_BRANCH
    try:
        data = _download_tarball(branch)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
        msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
        logmod.error("plugins-sync", msg)
        _record("check", False, msg)
        return {"ok": False, "error": msg, "status": status()}

    remote = 0
    missing = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                rel = _strip_top_dir(member.name)
                if rel is None:
                    continue
                dest = _safe_dest(rel, PLUGINS_DIR)
                if dest is None:
                    continue
                remote += 1
                if not os.path.exists(dest):
                    missing += 1
    except tarfile.TarError as e:
        msg = f"tarball parse failed: {e}"
        logmod.error("plugins-sync", msg)
        _record("check", False, msg)
        return {"ok": False, "error": msg, "status": status()}

    msg = f"remote has {remote} file(s); {missing} missing locally"
    logmod.ok("plugins-sync", msg)
    _record("check", True, msg)
    return {
        "ok":           True,
        "action":       "check",
        "remote_files": remote,
        "new_files":    missing,
        "status":       status(),
    }


def sync():
    branch = DEFAULT_BRANCH

    with _LOCK:
        if plugin_runner.is_running():
            msg = "a plugin is currently running. stop it before syncing."
            logmod.warn("plugins-sync", msg)
            _record("update", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        try:
            data = _download_tarball(branch)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as e:
            msg = f"could not reach {DEFAULT_REMOTE_URL}: {e}"
            logmod.error("plugins-sync", msg)
            _record("update", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        os.makedirs(PLUGINS_DIR, exist_ok=True)

        added = 0
        skipped = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    rel = _strip_top_dir(member.name)
                    if rel is None:
                        continue
                    dest = _safe_dest(rel, PLUGINS_DIR)
                    if dest is None:
                        continue
                    if os.path.exists(dest):
                        skipped += 1
                        continue
                    parent = os.path.dirname(dest)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    with open(dest, "wb") as out:
                        out.write(src.read())
                    added += 1
        except tarfile.TarError as e:
            msg = f"tarball extract failed: {e}"
            logmod.error("plugins-sync", msg)
            _record("update", False, msg)
            return {"ok": False, "error": msg, "status": status()}
        except OSError as e:
            msg = f"write failed: {e}"
            logmod.error("plugins-sync", msg)
            _record("update", False, msg)
            return {"ok": False, "error": msg, "status": status()}

        msg = f"downloaded {added} new file(s); kept {skipped} existing"
        logmod.ok("plugins-sync", msg)
        _record("update", True, msg)
        return {
            "ok":         True,
            "action":     "update",
            "downloaded": added,
            "skipped":    skipped,
            "status":     status(),
        }
