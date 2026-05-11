# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - HTTP-tarball updater. Replaces the git-based app_sync. Same public API
#   (status, check, pull, check_update, pull_update, switch_channel,
#   set_reload_hook, start) but downloads the GitHub-published source tarball
#   for the channel's branch and overwrites tracked source in place. No git on
#   PATH required, no dirty-tree gate, no diverging-branch failure mode.
# - Channels select which branch to follow:
#       stable        -> main
#       experimental  -> experimental
#       development   -> dev
# - Repo URL comes from env VERITATE_REPO_URL (e.g.
#   https://github.com/carpathian/veritate). When unset the module imports
#   cleanly and surfaces a useful error at update time.
# - User data dirs (data/, models/, plugins/, experiments/) plus .git and
#   .venv are preserved across updates.
# - urllib + tarfile + shutil only. No requests, no gitpython.
# veritate_mri/app_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

from readers import paths

import logs as logmod
import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

REPO_DIR    = paths.REPO_ROOT
STATE_PATH  = os.path.join(REPO_DIR, "data", "http_updater_state.json")

HTTP_TIMEOUT_SECS     = 60
DOWNLOAD_CHUNK_BYTES  = 64 * 1024
POLL_INTERVAL_SECS    = 30 * 60
POLL_FIRST_DELAY      = 60

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

# Top-level dirs in the repo root that hold gitignored user data or virtualenvs
# and must NEVER be overwritten by an update.
DEFAULT_SKIP_DIRS = ("data", "models", "plugins", "experiments", ".git", ".venv")

_REPO_URL_ENV         = "VERITATE_REPO_URL"
_REPO_URL_PLACEHOLDER = "https://github.com/<owner>/<repo>"

_LOCK         = threading.RLock()
_STATE_CACHE  = None
_THREAD       = None
_RELOAD_HOOK  = None

# ------------------------------------------------------------------------------------
# State helpers

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


# ------------------------------------------------------------------------------------
# Channel / URL helpers

def _channel():
    s = settings_mod.get()
    ch = s.get("update_channel")
    return ch if ch in ALL_CHANNELS else CHANNEL_STABLE


def _channel_branch():
    return CHANNEL_BRANCHES[_channel()]


def _repo_url_base():
    """Return the GitHub repo base URL. Strips trailing .git and slashes."""
    url = os.environ.get(_REPO_URL_ENV, "").strip()
    if not url:
        return None
    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def _tarball_url(branch):
    base = _repo_url_base()
    if not base:
        return None
    return f"{base}/archive/refs/heads/{branch}.tar.gz"


def _tarball_urls():
    return {ch: _tarball_url(br) for ch, br in CHANNEL_BRANCHES.items()}


# ------------------------------------------------------------------------------------
# HTTP helpers

def _build_request(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", "veritate-http-updater/1")
    req.add_header("Accept", "application/octet-stream")
    return req


def _etag_cached(url):
    """HEAD the tarball URL. Returns (etag, last_modified, error). Either of
    the first two may be None when the server omits the header."""
    req = _build_request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS) as resp:
            return resp.headers.get("ETag"), resp.headers.get("Last-Modified"), None
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code} on HEAD"
    except urllib.error.URLError as e:
        return None, None, f"network error on HEAD: {e.reason}"
    except Exception as e:
        return None, None, f"HEAD failed: {e}"


def _download_tarball(url, dst_path, progress_cb=None):
    """Stream-download `url` into `dst_path`. progress_cb(done, total) is
    optional. Returns (ok, error)."""
    req = _build_request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS) as resp:
            total_hdr = resp.headers.get("Content-Length")
            try:
                total = int(total_hdr) if total_hdr else None
            except ValueError:
                total = None
            done = 0
            with open(dst_path, "wb") as f:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        try:
                            progress_cb(done, total)
                        except Exception:
                            pass
        return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} on GET"
    except urllib.error.URLError as e:
        return False, f"network error on GET: {e.reason}"
    except Exception as e:
        return False, f"download failed: {e}"


# ------------------------------------------------------------------------------------
# Tarball extraction + copy

def _safe_extract(tar, dst_root):
    """Extract `tar` into `dst_root`, refusing entries that would escape via
    .. or absolute paths (CVE-2007-4559 style)."""
    dst_root_abs = os.path.realpath(dst_root)
    members = []
    for m in tar.getmembers():
        name = m.name
        if name.startswith("/") or ".." in name.split("/"):
            raise RuntimeError(f"unsafe path in tarball: {name!r}")
        target = os.path.realpath(os.path.join(dst_root_abs, name))
        if not target.startswith(dst_root_abs + os.sep) and target != dst_root_abs:
            raise RuntimeError(f"path escapes destination: {name!r}")
        members.append(m)
    tar.extractall(dst_root_abs, members=members)


def _find_extracted_root(extract_dir):
    """GitHub tarballs nest everything under a top-level `<repo>-<branch>/`
    directory. Return that dir, or `extract_dir` itself if there is no single
    top-level entry."""
    entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
    if len(entries) == 1:
        cand = os.path.join(extract_dir, entries[0])
        if os.path.isdir(cand):
            return cand
    return extract_dir


def _extract_and_copy(tarball_path, repo_root, skip_dirs=None):
    """Extract the tarball, then copy every file into `repo_root` EXCEPT files
    whose top-level path component is in `skip_dirs`. Returns
    {ok, copied, skipped, error}. Cleans up the temp dir on every exit."""
    skip = set(skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS)
    temp_dir = tempfile.mkdtemp(prefix="veritate-http-updater-")
    copied = 0
    skipped = 0
    try:
        try:
            with tarfile.open(tarball_path, "r:*") as tar:
                _safe_extract(tar, temp_dir)
        except (tarfile.TarError, RuntimeError) as e:
            return {"ok": False, "copied": 0, "skipped": 0, "error": f"extract failed: {e}"}

        src_root = _find_extracted_root(temp_dir)

        for dirpath, dirnames, filenames in os.walk(src_root):
            rel_dir = os.path.relpath(dirpath, src_root)
            if rel_dir == ".":
                rel_dir = ""
            if rel_dir == "":
                dirnames[:] = [d for d in dirnames if d not in skip]
            else:
                top = rel_dir.split(os.sep, 1)[0]
                if top in skip:
                    skipped += len(filenames)
                    dirnames[:] = []
                    continue

            for fname in filenames:
                src_file = os.path.join(dirpath, fname)
                rel_file = os.path.normpath(os.path.join(rel_dir, fname)) if rel_dir else fname
                top = rel_file.split(os.sep, 1)[0]
                if top in skip:
                    skipped += 1
                    continue
                dst_file = os.path.join(repo_root, rel_file)
                os.makedirs(os.path.dirname(dst_file) or repo_root, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied += 1

        return {"ok": True, "copied": copied, "skipped": skipped, "error": None}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------
# Public API

def status():
    url_base = _repo_url_base()
    last = _state()
    return {
        "is_repo":          True,
        "channel":          _channel(),
        "channel_branch":   _channel_branch(),
        "channels":         list(ALL_CHANNELS),
        "channel_map":      dict(CHANNEL_BRANCHES),
        "branch":           _channel_branch(),
        "head_short":       (last.get("etag") or "")[:7] or None,
        "remote_url":       url_base,
        "behind":           1 if last.get("update_available") else 0,
        "ahead":            0,
        "dirty":            False,
        "update_available": bool(last.get("update_available")),
        "tarball_urls":     _tarball_urls(),
        "last":             last,
    }


def check_update():
    """HEAD the channel tarball URL. Flags update_available when ETag/
    Last-Modified differs from the values recorded at the last successful pull."""
    url = _tarball_url(_channel_branch())
    if not url:
        msg = f"{_REPO_URL_ENV} is not set; cannot check for updates"
        logmod.warn("http-updater", msg)
        _update_state({
            "last_check_ts":  time.time(),
            "last_check_ok":  False,
            "last_check_msg": msg,
        })
        return {"ok": False, "error": msg, "status": status()}

    etag, last_modified, err = _etag_cached(url)
    if err:
        logmod.error("http-updater", f"check failed: {err}")
        _update_state({
            "last_check_ts":  time.time(),
            "last_check_ok":  False,
            "last_check_msg": err,
        })
        return {"ok": False, "error": err, "status": status()}

    cur = _state()
    pulled_etag = cur.get("pulled_etag")
    pulled_lm   = cur.get("pulled_last_modified")
    update_available = False
    if etag and pulled_etag:
        update_available = (etag != pulled_etag)
    elif last_modified and pulled_lm:
        update_available = (last_modified != pulled_lm)
    else:
        update_available = True

    _update_state({
        "last_check_ts":      time.time(),
        "last_check_ok":      True,
        "last_check_msg":     "",
        "etag":               etag,
        "last_modified":      last_modified,
        "update_available":   update_available,
        "remote_branch":      _channel_branch(),
        "tarball_url":        url,
    })
    return {"ok": True, "status": status()}


def pull_update(reload=False):
    """Download the channel tarball, extract, overwrite tracked source. User
    data dirs (data/, models/, plugins/, experiments/) are preserved. Fires
    the registered reload hook when `reload=True` or settings'
    `auto_reload_on_update` is on."""
    url = _tarball_url(_channel_branch())
    if not url:
        msg = f"{_REPO_URL_ENV} is not set; cannot pull updates"
        logmod.warn("http-updater", msg)
        return {"ok": False, "error": msg}

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="veritate-tarball-", suffix=".tar.gz")
    os.close(tmp_fd)
    try:
        logmod.info("http-updater", f"downloading {url}")
        ok, err = _download_tarball(url, tmp_path)
        if not ok:
            logmod.error("http-updater", f"download failed: {err}")
            _update_state({
                "last_pull_ts":  time.time(),
                "last_pull_ok":  False,
                "last_pull_msg": err,
            })
            return {"ok": False, "error": err}

        post_etag, post_lm, _ = _etag_cached(url)

        result = _extract_and_copy(tmp_path, REPO_DIR, DEFAULT_SKIP_DIRS)
        if not result["ok"]:
            logmod.error("http-updater", f"apply failed: {result['error']}")
            _update_state({
                "last_pull_ts":  time.time(),
                "last_pull_ok":  False,
                "last_pull_msg": result["error"],
            })
            return {"ok": False, "error": result["error"]}

        msg = f"synced {_channel_branch()} ({result['copied']} files; {result['skipped']} preserved)"
        logmod.ok("http-updater", msg)
        _update_state({
            "last_pull_ts":         time.time(),
            "last_pull_ok":         True,
            "last_pull_msg":        msg,
            "pulled_etag":          post_etag,
            "pulled_last_modified": post_lm,
            "update_available":     False,
        })

        if reload or settings_mod.get().get("auto_reload_on_update"):
            if _RELOAD_HOOK is not None:
                logmod.warn("http-updater", "reload hook firing after update")
                try:
                    _RELOAD_HOOK()
                except Exception as e:
                    logmod.error("http-updater", f"reload hook failed: {e}")
        return {"ok": True, "status": status(), "copied": result["copied"], "skipped": result["skipped"]}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def switch_channel(channel):
    if channel not in ALL_CHANNELS:
        return {"ok": False, "error": f"unknown channel: {channel}"}
    settings_mod.update({"update_channel": channel})
    _update_state({"pulled_etag": None, "pulled_last_modified": None, "update_available": True})
    logmod.ok("http-updater", f"switched channel to {channel} (branch {CHANNEL_BRANCHES[channel]})")
    return {"ok": True, "status": status()}


def set_reload_hook(fn):
    """Caller registers a parameterless callable that triggers a soft reload."""
    global _RELOAD_HOOK
    _RELOAD_HOOK = fn


def _poll_loop():
    time.sleep(POLL_FIRST_DELAY)
    while True:
        try:
            check_update()
        except Exception as e:
            logmod.error("http-updater", f"poll error: {e}")
        time.sleep(POLL_INTERVAL_SECS)


def start():
    global _THREAD
    if _THREAD is not None:
        return
    t = threading.Thread(target=_poll_loop, name="http-updater-poll", daemon=True)
    t.start()
    _THREAD = t
    logmod.info("http-updater", f"channel={_channel()} branch={_channel_branch()} url={_repo_url_base() or '(unset)'}")


# Back-compat aliases. The previous git-based app_sync exposed `check()` and
# `pull()` as the call-site names; keep them so existing handlers in app.py do
# not need to change.
check = check_update
pull  = pull_update
