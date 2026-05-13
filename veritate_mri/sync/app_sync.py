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
# veritate_mri/sync/app_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import ssl
import urllib.error
import urllib.request

from readers import paths
from runtime import logs as logmod
from runtime import settings as settings_mod
from training import plugin_runner

# ------------------------------------------------------------------------------------
# Constants

REPO_DIR        = paths.REPO_ROOT
STATE_PATH      = os.path.join(REPO_DIR, "data", "http_updater_state.json")
# Per-file SHA snapshot of what the last successful pull wrote. Used to detect
# files the user has edited locally since the upstream baseline. Stored under
# data/ so it survives updates (data/ is in DEFAULT_SKIP_DIRS).
BASELINE_PATH   = os.path.join(REPO_DIR, "data", "http_updater_baseline.json")
SHA_CHUNK_BYTES = 1024 * 1024

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


def _is_git_checkout():
    return os.path.isdir(os.path.join(REPO_DIR, ".git"))


def _local_git_branch():
    """Read `.git/HEAD` to determine the currently checked-out branch. Returns
    the branch name (e.g. "dev") or None if `.git/HEAD` is missing, detached,
    or unreadable."""
    head_path = os.path.join(REPO_DIR, ".git", "HEAD")
    if not os.path.isfile(head_path):
        return None
    try:
        with open(head_path, "r", encoding="utf-8", errors="replace") as f:
            line = f.read().strip()
    except OSError:
        return None
    if line.startswith("ref:"):
        ref = line.partition(":")[2].strip()
        prefix = "refs/heads/"
        if ref.startswith(prefix):
            return ref[len(prefix):] or None
    return None


def _active_branch():
    """Branch the updater should actually track. In a git checkout the locally
    checked-out branch wins — developers may be testing on a branch and must
    not be prompted to overwrite it with a different one. Falls back to the
    channel branch only when no `.git/HEAD` is present (tarball install)."""
    return _local_git_branch() or _channel_branch()


def _active_channel():
    """Channel name corresponding to the active branch. Returns None for
    branches that don't map to a known channel (e.g. feature/PR branches)."""
    return BRANCH_TO_CHANNEL.get(_active_branch())


def _normalize_github_url(url):
    """Strip trailing .git / slashes; rewrite SSH (git@github.com:owner/repo)
    form to https://github.com/owner/repo so tarball URLs are constructable."""
    url = (url or "").strip()
    if not url:
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def _git_remote_url():
    """Read `origin` from .git/config without shelling out. Returns None if
    .git/config is absent or doesn't contain a remote.origin url."""
    cfg = os.path.join(REPO_DIR, ".git", "config")
    if not os.path.isfile(cfg):
        return None
    in_origin = False
    try:
        with open(cfg, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("[remote "):
                    in_origin = (line == '[remote "origin"]')
                    continue
                if in_origin and line.startswith("url"):
                    _, _, value = line.partition("=")
                    return value.strip() or None
    except OSError:
        return None
    return None


def _repo_url_base():
    """Return the GitHub repo base URL. Prefers `VERITATE_REPO_URL` env var;
    falls back to the git remote `origin` URL so the in-app updater works
    zero-config on any checkout. Returns None only if both are unavailable."""
    return _normalize_github_url(
        os.environ.get(_REPO_URL_ENV, "") or _git_remote_url()
    )


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


def _ssl_context():
    """SSL context with a usable trust store. The macOS framework Python
    installer ships its own and that breaks if "Install Certificates.command"
    wasn't run. We prefer `certifi` (bundled on most modern Python installs)
    so the updater works regardless of which Python the user runs against."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _etag_cached(url):
    """HEAD the tarball URL. Returns (etag, last_modified, error). Either of
    the first two may be None when the server omits the header."""
    req = _build_request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS,
                                     context=_ssl_context()) as resp:
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
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS,
                                     context=_ssl_context()) as resp:
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


def _sha256_file(path):
    """Streaming sha256 of a file on disk. Returns None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(SHA_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _normalize_rel(rel):
    """Normalize a relative path to forward slashes for stable storage in JSON."""
    return rel.replace(os.sep, "/")


def _read_baseline():
    """Returns {rel_path: sha256} for the last successful pull. Empty if no
    baseline yet (first run after switching to per-file tracking, or never
    pulled)."""
    if not os.path.isfile(BASELINE_PATH):
        return {}
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict): return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def _write_baseline(files, branch=""):
    """Atomically replace BASELINE_PATH."""
    payload = {
        "version":       1,
        "written_at":    time.time(),
        "branch":        branch or "",
        "files":         dict(files),
    }
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    tmp = BASELINE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, BASELINE_PATH)


def _extract_and_copy(tarball_path, repo_root, skip_dirs=None):
    """Extract the tarball, then copy every file into `repo_root` EXCEPT files
    whose top-level path component is in `skip_dirs`. Also hashes each written
    file so the caller can persist a baseline. Returns
    {ok, copied, skipped, error, baseline}. Cleans up temp dir on every exit."""
    skip = set(skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS)
    temp_dir = tempfile.mkdtemp(prefix="veritate-http-updater-")
    copied = 0
    skipped = 0
    baseline = {}
    try:
        try:
            with tarfile.open(tarball_path, "r:*") as tar:
                _safe_extract(tar, temp_dir)
        except (tarfile.TarError, RuntimeError) as e:
            return {"ok": False, "copied": 0, "skipped": 0, "error": f"extract failed: {e}",
                    "baseline": {}}

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
                sha = _sha256_file(dst_file)
                if sha:
                    baseline[_normalize_rel(rel_file)] = sha

        return {"ok": True, "copied": copied, "skipped": skipped,
                "error": None, "baseline": baseline}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def local_edits(skip_dirs=None):
    """Return the set of repo files that differ from the last-pulled baseline.

    Three relations:
      - "modified" — file exists locally with a different SHA than baseline
      - "missing"  — file is in baseline but deleted locally (user/git/build)
      - "added"    — file is on disk but never tracked by a pull (user-added)
                     reported only when its top-level dir is NOT in skip_dirs
                     and only if it has a Veritate-relevant extension (.py/.js/.html/.css/.md)

    Returns {ok, has_baseline, modified, missing, added, counts}. has_baseline
    is False on the very first call (no pull has run with this code path yet);
    callers treat that case as "no protection available, proceed."""
    skip = set(skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS)
    baseline = _read_baseline()
    if not baseline:
        return {
            "ok":           True,
            "has_baseline": False,
            "modified":     [],
            "missing":      [],
            "added":        [],
            "counts":       {"modified": 0, "missing": 0, "added": 0},
        }

    modified = []
    missing  = []
    seen     = set()

    # Pass 1: walk the baseline. For each tracked file, compare local SHA.
    for rel, base_sha in baseline.items():
        seen.add(rel)
        local_path = os.path.join(REPO_DIR, rel)
        if not os.path.isfile(local_path):
            missing.append({"path": rel, "baseline_sha": base_sha})
            continue
        local_sha = _sha256_file(local_path)
        if local_sha != base_sha:
            modified.append({
                "path":         rel,
                "baseline_sha": base_sha,
                "local_sha":    local_sha,
            })

    # Pass 2: surface files the user added that aren't in baseline. Only check
    # source-y extensions so we don't flood the dashboard with pyc, generated
    # binaries, IDE droppings, etc.
    added = []
    SOURCE_EXTS = (".py", ".js", ".html", ".css", ".md", ".json", ".sh", ".toml", ".yaml", ".yml")
    for dirpath, dirnames, filenames in os.walk(REPO_DIR):
        rel_dir = os.path.relpath(dirpath, REPO_DIR)
        if rel_dir == ".":
            rel_dir = ""
            dirnames[:] = [d for d in dirnames if d not in skip]
        else:
            top = rel_dir.split(os.sep, 1)[0]
            if top in skip:
                dirnames[:] = []
                continue
        for fname in filenames:
            if not fname.endswith(SOURCE_EXTS): continue
            rel_file = _normalize_rel(os.path.normpath(os.path.join(rel_dir, fname)) if rel_dir else fname)
            if rel_file in seen: continue
            added.append({"path": rel_file})

    return {
        "ok":           True,
        "has_baseline": True,
        "modified":     modified,
        "missing":      missing,
        "added":        added,
        "counts": {
            "modified": len(modified),
            "missing":  len(missing),
            "added":    len(added),
        },
    }


# ------------------------------------------------------------------------------------
# Public API

def status():
    url_base = _repo_url_base()
    last = _state()
    active_branch = _active_branch()
    active_channel = _active_channel()
    return {
        "is_repo":          True,
        "channel":          _channel(),
        "channel_branch":   _channel_branch(),
        "channels":         list(ALL_CHANNELS),
        "channel_map":      dict(CHANNEL_BRANCHES),
        "branch":           active_branch,
        "tracked_channel":  active_channel,
        "local_branch":     _local_git_branch(),
        "is_git_checkout":  _is_git_checkout(),
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
    """HEAD the tarball URL for the active branch (locally checked-out branch
    in a git checkout, channel branch otherwise). Flags `update_available`
    when ETag/Last-Modified differs from values recorded at the last pull on
    the same branch. In a git checkout with no prior pull baseline, silently
    records the current remote head as the baseline rather than claiming an
    update — the developer's working tree is the source of truth, and we only
    surface a banner when the remote actually moves away from what we've seen."""
    branch = _active_branch()
    url = _tarball_url(branch)
    if not url:
        msg = (f"no repo URL: set {_REPO_URL_ENV} or check that .git/config "
               f"has a `remote.origin.url` pointing at the Veritate repo")
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
    pulled_branch = cur.get("pulled_branch")

    # A baseline only applies to the branch it was recorded on. If we've
    # switched branches (e.g. dev -> main) since the last pull, discard the
    # old baseline so we don't compare across branches.
    if pulled_branch and pulled_branch != branch:
        pulled_etag = None
        pulled_lm = None

    git_checkout = _is_git_checkout()
    baseline_patch = {}

    if etag and pulled_etag:
        update_available = (etag != pulled_etag)
    elif last_modified and pulled_lm:
        update_available = (last_modified != pulled_lm)
    elif git_checkout:
        # Developer checkout, no baseline for this branch yet: their local
        # tree is the source of truth. Silently baseline against the current
        # remote head so future checks compare to a real prior observation
        # instead of always returning True.
        update_available = False
        baseline_patch = {
            "pulled_etag":          etag,
            "pulled_last_modified": last_modified,
            "pulled_branch":        branch,
        }
    else:
        # Plain tarball install with no pull history: assume update available
        # until the user pulls once.
        update_available = True

    patch = {
        "last_check_ts":      time.time(),
        "last_check_ok":      True,
        "last_check_msg":     "",
        "etag":               etag,
        "last_modified":      last_modified,
        "update_available":   update_available,
        "remote_branch":      branch,
        "tarball_url":        url,
    }
    patch.update(baseline_patch)
    _update_state(patch)
    return {"ok": True, "status": status()}


def pull_update(reload=False, force=False, ignore_training=False):
    """Download the channel tarball, extract, overwrite tracked source. User
    data dirs (data/, models/, plugins/, experiments/) are preserved. Fires
    the registered reload hook when `reload=True` or settings'
    `auto_reload_on_update` is on.

    Safety gates (both bypassable by the caller):
      - if a plugin/trainer is currently running, refuses unless `ignore_training`
        is True. Overwriting source files mid-run is the most common foot-gun
        (especially on Windows file locks).
      - if local_edits() reports any modified/missing/added source files,
        refuses unless `force` is True. The dashboard surfaces this as a
        confirm() dialog with the file list."""
    if not ignore_training and plugin_runner.is_running():
        msg = "a plugin/trainer is running. stop it first or pass ignore_training=true."
        logmod.warn("http-updater", msg)
        return {"ok": False, "error": msg, "training_active": True}

    if not force:
        edits = local_edits()
        # Only gate on modified/missing — those are the cases where pull would
        # overwrite or fail to restore user work. "added" files are user-created
        # and never touched by the updater (it only writes files present in
        # the tarball), so they don't need to block the pull.
        if edits.get("has_baseline") and (
            edits["counts"]["modified"] > 0
            or edits["counts"]["missing"]  > 0
        ):
            c = edits["counts"]
            msg = (f"local edits detected: {c['modified']} modified, "
                   f"{c['missing']} missing. pass force=true to overwrite.")
            logmod.warn("http-updater", msg)
            return {
                "ok":             False,
                "error":          msg,
                "requires_force": True,
                "edits":          edits,
            }

    branch = _active_branch()
    url = _tarball_url(branch)
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

        msg = f"synced {branch} ({result['copied']} files; {result['skipped']} preserved)"
        logmod.ok("http-updater", msg)
        # Persist the per-file SHA snapshot so the next pull can detect local
        # edits. Failure to write the baseline is non-fatal — the pull itself
        # succeeded.
        try:
            _write_baseline(result.get("baseline") or {}, branch=branch)
        except OSError as e:
            logmod.warn("http-updater", f"baseline write failed (non-fatal): {e}")
        _update_state({
            "last_pull_ts":         time.time(),
            "last_pull_ok":         True,
            "last_pull_msg":        msg,
            "pulled_etag":          post_etag,
            "pulled_last_modified": post_lm,
            "pulled_branch":        branch,
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
    _update_state({
        "pulled_etag":         None,
        "pulled_last_modified": None,
        "pulled_branch":       None,
        "update_available":    True,
    })
    target_branch = CHANNEL_BRANCHES[channel]
    msg = f"switched channel to {channel} (branch {target_branch})"
    if _is_git_checkout():
        local = _local_git_branch()
        if local and local != target_branch:
            msg += (f"; note: git checkout is on {local!r}, which takes "
                    f"precedence over the channel setting")
    logmod.ok("http-updater", msg)
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
    logmod.info(
        "http-updater",
        f"channel={_channel()} tracking_branch={_active_branch()} "
        f"local_branch={_local_git_branch() or '(none)'} "
        f"url={_repo_url_base() or '(unset)'}"
    )


# Back-compat aliases. The previous git-based app_sync exposed `check()` and
# `pull()` as the call-site names; keep them so existing handlers in app.py do
# not need to change.
check = check_update
pull  = pull_update
