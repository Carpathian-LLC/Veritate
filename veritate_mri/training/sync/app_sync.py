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
# - Repo URL resolves env VERITATE_REPO_URL -> git remote origin -> the canonical
#   public repo default, so the updater works zero-config on any checkout, even one
#   that was not `git clone`d. The env var still lets a fork or test branch override.
# - User data dirs (data/, models/, trainers/, experiments/) plus .git and
#   .venv are preserved across updates.
# - urllib + tarfile + shutil only. No requests, no gitpython.
# veritate_mri/sync/app_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

from readers import paths
from runtime import logs as logmod
from runtime import net
from runtime import settings as settings_mod

from training import trainer_runner as plugin_runner

from . import sync_common as sc

# ------------------------------------------------------------------------------------
# Constants

REPO_DIR        = paths.REPO_ROOT
STATE_PATH      = os.path.join(REPO_DIR, "data", "http_updater_state.json")
# Per-file SHA snapshot of what the last successful pull wrote. Used to detect
# files the user has edited locally since the upstream baseline. Stored under
# data/ so it survives updates (data/ is in DEFAULT_SKIP_DIRS).
BASELINE_PATH   = os.path.join(REPO_DIR, "data", "http_updater_baseline.json")

HTTP_TIMEOUT_SECS     = 60
DOWNLOAD_CHUNK_BYTES  = 64 * 1024
POLL_INTERVAL_SECS    = 30 * 60
POLL_FIRST_DELAY      = 60

TARBALL_TMP_PREFIX = "veritate-tarball-"
TARBALL_TMP_SUFFIX = ".tar.gz"
EXTRACT_TMP_PREFIX = "veritate-http-updater-"

# Extensions reported as user-added files. Anything else (pyc, binaries, editor
# droppings) would drown the list.
TRACKED_SOURCE_EXTS = (".py", ".js", ".html", ".css", ".md", ".json", ".sh", ".toml", ".yaml", ".yml")

# A baseline this far out of sync with the tree is drift (git pull, manual
# unzip, moved layout), not user edits: discard it instead of gating on it.
STALE_BASELINE_MIN_FILES = 20
STALE_BASELINE_SHARE     = 0.5

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

# Derived from paths.py so any future rename of the user-data roots flows
# through automatically.
DEFAULT_SKIP_DIRS = tuple(sorted({
    os.path.basename(paths.MODELS_ROOT), os.path.basename(paths.PLUGINS_ROOT),
    "data", "experiments", ".git", ".venv", "venv", "__pycache__",
}))

_REPO_URL_ENV     = "VERITATE_REPO_URL"
# Canonical public repo. Final fallback so a checkout with no origin remote and
# no env override still resolves an update source instead of going dead.
_REPO_URL_DEFAULT = "https://github.com/Carpathian-LLC/Veritate"
GITHUB_API_BASE   = "https://api.github.com"
# Source-tarball path GitHub serves for a branch, appended to the repo base URL.
GITHUB_BRANCH_TARBALL_FMT = "{base}/archive/refs/heads/{branch}.tar.gz"

# .git layout the updater parses directly (no git binary required).
GIT_DIR_NAME       = ".git"
GIT_CONFIG_NAME    = "config"
GIT_REMOTE_SECTION = "[remote "
GIT_ORIGIN_SECTION = '[remote "origin"]'
GIT_URL_KEY        = "url"
GIT_HEAD_NAME      = "HEAD"
GIT_PACKED_REFS_NAME = "packed-refs"
GIT_REF_PREFIX     = "ref:"
GIT_HEADS_PREFIX   = "refs/heads/"

_LOCK         = threading.RLock()
_STATE_CACHE  = None
_THREAD       = None
_RELOAD_HOOK  = None

# Throttle key for coalescing repeated offline check failures (see
# logs.emit_throttled). Cleared on the first successful check.
_CHECK_THROTTLE_KEY = "http-updater-check"

# ------------------------------------------------------------------------------------
# State helpers

def _read_state():
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
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
    return os.path.isdir(os.path.join(REPO_DIR, GIT_DIR_NAME))


def _local_git_branch():
    """Read `.git/HEAD` to determine the currently checked-out branch. Returns
    the branch name (e.g. "dev") or None if `.git/HEAD` is missing, detached,
    or unreadable."""
    head_path = os.path.join(REPO_DIR, GIT_DIR_NAME, GIT_HEAD_NAME)
    if not os.path.isfile(head_path):
        return None
    try:
        with open(head_path, encoding="utf-8", errors="replace") as f:
            line = f.read().strip()
    except OSError:
        return None
    if line.startswith(GIT_REF_PREFIX):
        ref = line.partition(":")[2].strip()
        if ref.startswith(GIT_HEADS_PREFIX):
            return ref[len(GIT_HEADS_PREFIX):] or None
    return None


def _local_head_sha():
    """Resolve the local HEAD commit SHA from `.git` without shelling out.
    Follows `.git/HEAD` to its branch ref, reading the loose ref file and
    falling back to `.git/packed-refs`. Returns the SHA, or None on a detached
    or unreadable HEAD with no resolvable ref."""
    git_dir = os.path.join(REPO_DIR, GIT_DIR_NAME)
    try:
        with open(os.path.join(git_dir, GIT_HEAD_NAME), encoding="utf-8", errors="replace") as f:
            head = f.read().strip()
    except OSError:
        return None
    if not head.startswith(GIT_REF_PREFIX):
        return head or None
    ref = head.partition(":")[2].strip()
    try:
        with open(os.path.join(git_dir, *ref.split("/")), encoding="utf-8", errors="replace") as f:
            return f.read().strip() or None
    except OSError:
        pass
    try:
        with open(os.path.join(git_dir, GIT_PACKED_REFS_NAME), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("#", "^")):
                    continue
                sha, _, name = line.partition(" ")
                if name == ref:
                    return sha or None
    except OSError:
        return None
    return None


def _active_branch():
    """Branch the updater should actually track. In a git checkout the locally
    checked-out branch wins: developers may be testing on a branch and must
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
    cfg = os.path.join(REPO_DIR, GIT_DIR_NAME, GIT_CONFIG_NAME)
    if not os.path.isfile(cfg):
        return None
    in_origin = False
    try:
        with open(cfg, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith(GIT_REMOTE_SECTION):
                    in_origin = (line == GIT_ORIGIN_SECTION)
                    continue
                if in_origin and line.startswith(GIT_URL_KEY):
                    _, _, value = line.partition("=")
                    return value.strip() or None
    except OSError:
        return None
    return None


def _repo_url_base():
    """Return the GitHub repo base URL. Resolves `VERITATE_REPO_URL` env var ->
    git remote `origin` -> the canonical public repo default, so the in-app
    updater works zero-config on any checkout, including one that was not
    `git clone`d. The env var lets a fork or test branch override the default."""
    return _normalize_github_url(
        os.environ.get(_REPO_URL_ENV, "") or _git_remote_url() or _REPO_URL_DEFAULT
    )


def _tarball_url(branch):
    base = _repo_url_base()
    if not base:
        return None
    return GITHUB_BRANCH_TARBALL_FMT.format(base=base, branch=branch)


def _tarball_urls():
    return {ch: _tarball_url(br) for ch, br in CHANNEL_BRANCHES.items()}


def _repo_slug():
    """`owner/repo` from the resolved repo URL, or None."""
    base = _repo_url_base()
    parts = (base or "").rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else None


def _remote_branch_sha(branch):
    """Commit SHA at the tip of the remote `branch`, or None.

    This is what a tarball actually contains. `pull_update` records it so a later
    check has a real base to compare against: a tarball extract writes files but
    creates no commit, so `.git/HEAD` never advances and comparing against it
    reports the same `behind` count forever, however many times the user updates."""
    slug = _repo_slug()
    if not slug:
        return None
    req = urllib.request.Request(f"{GITHUB_API_BASE}/repos/{slug}/commits/{branch}",
                                 method="GET")
    req.add_header("User-Agent", "veritate-http-updater/1")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS,
                                     context=net.ssl_context()) as resp:
            return (json.loads(resp.read().decode("utf-8")) or {}).get("sha") or None
    except Exception:
        return None


def _compare_base(git_checkout, branch):
    """SHA to measure `behind` from: what this updater last pulled, falling back
    to `.git/HEAD` only for a checkout this updater has never written. A commit
    recorded on a different branch is discarded, matching how the ETag baseline
    is dropped across a channel switch."""
    st = _state() or {}
    pulled = st.get("pulled_commit")
    if pulled and (st.get("pulled_branch") or branch) == branch:
        return pulled
    return _local_head_sha() if git_checkout else None


def _remote_ahead_behind(branch, local_sha):
    """Commits the remote `branch` tip has that the local HEAD lacks, via the
    GitHub compare API (base=local, head=remote). Returns (behind, error).
    `behind` is 0 when identical or when local is ahead of the remote, so a
    developer's own pushes never read as an update; None means undetermined and
    the caller should fall back to the tarball ETag. A 404 (GitHub doesn't know
    the local SHA: unpushed commits) means local is the source of truth -> 0."""
    slug = _repo_slug()
    if not slug or not local_sha:
        return None, "no slug/sha"
    url = f"{GITHUB_API_BASE}/repos/{slug}/compare/{local_sha}...{branch}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "veritate-http-updater/1")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS,
                                     context=net.ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return (0, None) if e.code == 404 else (None, f"HTTP {e.code} on compare")
    except urllib.error.URLError as e:
        return None, f"network error on compare: {e.reason}"
    except Exception as e:
        return None, f"compare failed: {e}"
    ahead_by = data.get("ahead_by")
    return (ahead_by, None) if isinstance(ahead_by, int) else (None, "compare missing ahead_by")


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
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS,
                                     context=net.ssl_context()) as resp:
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
                                     context=net.ssl_context()) as resp:
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
        with open(BASELINE_PATH, encoding="utf-8") as f:
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


def _iter_tracked(root, skip):
    """Yield (rel, abs_path) for every file under `root` whose top-level dir is
    not skipped. rel is forward-slashed."""
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
            dirnames[:] = [d for d in dirnames if d not in skip]
        else:
            top = rel_dir.split(os.sep, 1)[0]
            if top in skip:
                dirnames[:] = []
                continue
        for fname in filenames:
            rel = os.path.join(rel_dir, fname) if rel_dir else fname
            yield _normalize_rel(rel), os.path.join(dirpath, fname)


def _scan_incoming(tarball_path, temp_dir, skip_dirs=None):
    """Extract the tarball into `temp_dir` and hash every file it would write.
    Returns (src_root, {rel: sha}, error)."""
    skip = set(skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS)
    try:
        with tarfile.open(tarball_path, "r:*") as tar:
            _safe_extract(tar, temp_dir)
    except (tarfile.TarError, RuntimeError) as e:
        return None, {}, f"extract failed: {e}"

    src_root = _find_extracted_root(temp_dir)
    incoming = {}
    for rel, abs_path in _iter_tracked(src_root, skip):
        sha = sc.sha256_file(abs_path)
        if sha:
            incoming[rel] = sha
    return src_root, incoming, None


def _copy_incoming(src_root, incoming, repo_root):
    """Write the scanned files into `repo_root`. Bytes match the scan, so
    `incoming` doubles as the new baseline."""
    for rel in incoming:
        native = rel.replace("/", os.sep)
        dst_file = os.path.join(repo_root, native)
        os.makedirs(os.path.dirname(dst_file) or repo_root, exist_ok=True)
        shutil.copy2(os.path.join(src_root, native), dst_file)


def _no_baseline(stale=False):
    return {
        "ok":             True,
        "has_baseline":   False,
        "stale_baseline": stale,
        "modified":       [],
        "missing":        [],
        "added":          [],
        "counts":         {"modified": 0, "missing": 0, "added": 0},
    }


def _local_noise_dirs():
    """Top-level dirs that exist only on this machine: virtualenvs (any name, so
    `pyvenv.cfg` is the test rather than a name list) and tooling caches. They are
    never in the baseline, so without this every file inside one is reported as a
    file the user added — a single venv turns a clean tree into thousands of
    'changes'. Applies to the local walk only; the incoming tarball has none of
    these, and dot-dirs that DO ship (.github) are in the baseline and filtered
    out by the `seen` check anyway."""
    noise = set()
    try:
        entries = os.listdir(REPO_DIR)
    except OSError:
        return noise
    for entry in entries:
        abs_entry = os.path.join(REPO_DIR, entry)
        if not os.path.isdir(abs_entry):
            continue
        if entry.startswith(".") or os.path.isfile(os.path.join(abs_entry, "pyvenv.cfg")):
            noise.add(entry)
    return noise


def local_edits(skip_dirs=None, incoming=None):
    """Return the repo files a pull would overwrite against the user's wishes.

    `incoming` is {rel: sha} for the files the pull is about to write. Given it,
    a baseline entry only counts when the pull would actually touch that path
    AND the local bytes differ from what lands there: a file already identical
    to the incoming version is not a conflict, and a path upstream no longer
    ships cannot be overwritten at all. Without it the comparison is advisory
    (the /app/local_edits diagnostic) and reports every drift from the last pull.

    Three relations:
      - "modified": file exists locally with a different SHA than baseline
      - "missing" : file is in baseline but deleted locally (user/git/build)
      - "added"   : file is on disk but never tracked by a pull (user-added)
                     reported only when its top-level dir is NOT in skip_dirs
                     and only if it has a Veritate-relevant extension (.py/.js/.html/.css/.md)

    Returns {ok, has_baseline, modified, missing, added, counts}. has_baseline
    is False on the very first call (no pull has run with this code path yet);
    callers treat that case as "no protection available, proceed."""
    skip = set(skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS)
    baseline = _read_baseline()
    if not baseline:
        return _no_baseline()

    modified = []
    missing  = []
    seen     = set()

    # Pass 1: walk the baseline. For each tracked file, compare local SHA.
    for rel, base_sha in baseline.items():
        seen.add(rel)
        if incoming is not None and rel not in incoming:
            continue
        local_path = os.path.join(REPO_DIR, rel)
        if not os.path.isfile(local_path):
            missing.append({"path": rel, "baseline_sha": base_sha})
            continue
        local_sha = sc.sha256_file(local_path)
        if local_sha == base_sha:
            continue
        if incoming is not None and local_sha == incoming[rel]:
            continue
        modified.append({
            "path":         rel,
            "baseline_sha": base_sha,
            "local_sha":    local_sha,
        })

    # Self-heal: if the baseline barely matches reality (mass rename, manual
    # restore, dev did `git pull`, user unzipped over the install), treat it
    # as obsolete and report no baseline. The next successful pull writes a
    # fresh baseline naturally; healthy installs never hit this branch.
    diverged = len(modified) + len(missing)
    if diverged >= max(STALE_BASELINE_MIN_FILES, int(len(baseline) * STALE_BASELINE_SHARE)):
        logmod.warn("http-updater",
                    f"baseline appears stale ({diverged}/{len(baseline)} files "
                    f"diverged); ignoring it: next pull will rebuild")
        return _no_baseline(stale=True)

    # Pass 2: surface files the user added that aren't in baseline. Only check
    # source-y extensions so we don't flood the dashboard with pyc, generated
    # binaries, IDE droppings, etc.
    added = [{"path": rel} for rel, _ in _iter_tracked(REPO_DIR, skip | _local_noise_dirs())
             if rel.endswith(TRACKED_SOURCE_EXTS) and rel not in seen]

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
        # The commit this updater last pulled. Was built from the ETag, which is
        # not a commit and rendered as a quote-prefixed string like `"613e9c`.
        "head_short":       (last.get("pulled_commit") or _local_head_sha() or "")[:7] or None,
        "remote_url":       url_base,
        "behind":           last.get("behind", 1 if last.get("update_available") else 0),
        "ahead":            0,
        "dirty":            False,
        "update_available": bool(last.get("update_available")),
        "tarball_urls":     _tarball_urls(),
        "last":             last,
    }


def check_update():
    """Decide whether the active branch has commits the local tree lacks.

    In a git checkout the local HEAD is the source of truth: compare it to the
    remote branch tip (GitHub compare API) and treat the remote-ahead count as
    `behind`. This is what makes the banner intelligent: a commit the developer
    made and pushed from this machine leaves local HEAD == remote tip, so
    `behind` is 0 and no banner shows; unpushed local commits (local ahead) are
    0 too. Only commits the remote genuinely has beyond local raise the banner,
    with the real count.

    The opaque tarball ETag can't tell "remote moved ahead" from "I pushed my
    own commit", so it is only the fallback: for tarball installs (no `.git`),
    or when the compare can't be reached. HEAD the tarball either way to keep
    the ETag baseline and `head_short` fresh."""
    branch = _active_branch()
    url = _tarball_url(branch)
    etag, last_modified, err = _etag_cached(url)
    if err:
        # A transient transport failure (no route to host, DNS, TLS timeout) on
        # an offline box is not an application error: don't fire the error hook
        # (which inflates the heartbeat error count) and don't relog it on every
        # 30-min poll. Genuine HTTP errors (repo gone, rate-limited) stay loud.
        if err.startswith("network error"):
            logmod.emit_throttled("warn", "http-updater", f"check failed: {err}",
                                  key=_CHECK_THROTTLE_KEY)
        else:
            logmod.error("http-updater", f"check failed: {err}")
        _update_state({
            "last_check_ts":  time.time(),
            "last_check_ok":  False,
            "last_check_msg": err,
        })
        return {"ok": False, "error": err, "status": status()}
    logmod.clear_throttle(_CHECK_THROTTLE_KEY)

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
    behind = None

    # Compare against what this updater last pulled, or `.git/HEAD` on a checkout
    # it has never written. Comparing a tarball-updated checkout against HEAD is
    # what made `behind` stick at the same number after every successful update.
    base_sha = _compare_base(git_checkout, branch)
    if base_sha:
        behind, _cmp_err = _remote_ahead_behind(branch, base_sha)

    if behind is not None:
        update_available = behind > 0
        # Refresh the etag baseline only when in sync, so the fallback path
        # still detects a difference if a later compare can't be reached.
        if not update_available:
            baseline_patch = {
                "pulled_etag":          etag,
                "pulled_last_modified": last_modified,
                "pulled_branch":        branch,
            }
    elif etag and pulled_etag:
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

    if behind is None:
        behind = 1 if update_available else 0

    patch = {
        "last_check_ts":      time.time(),
        "last_check_ok":      True,
        "last_check_msg":     "",
        "etag":               etag,
        "last_modified":      last_modified,
        "update_available":   update_available,
        "behind":             behind,
        "remote_branch":      branch,
        "tarball_url":        url,
    }
    patch.update(baseline_patch)
    _update_state(patch)
    return {"ok": True, "status": status()}


def pull_update(reload=False, force=False, ignore_training=False):
    """Download the channel tarball, extract, overwrite tracked source. User
    data dirs (data/, models/, trainers/, experiments/) are preserved. Fires
    the registered reload hook when `reload=True` or settings'
    `auto_reload_on_update` is on.

    Safety gates (both bypassable by the caller):
      - if a plugin/trainer is currently running, refuses unless `ignore_training`
        is True. Overwriting source files mid-run is the most common foot-gun
        (especially on Windows file locks).
      - once the tarball is on disk, if local_edits() reports any file whose
        local bytes differ from the version about to be written, refuses unless
        `force` is True. The dashboard surfaces this as a confirm() dialog with
        the file list. The check runs post-download so it compares against the
        incoming files, not against a snapshot of the last pull: a file already
        matching upstream, or one upstream dropped, is never reported."""
    if not ignore_training and plugin_runner.is_running():
        msg = "a plugin/trainer is running. stop it first or pass ignore_training=true."
        logmod.warn("http-updater", msg)
        return {"ok": False, "error": msg, "training_active": True}

    branch = _active_branch()
    url = _tarball_url(branch)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=TARBALL_TMP_PREFIX, suffix=TARBALL_TMP_SUFFIX)
    os.close(tmp_fd)
    temp_dir = tempfile.mkdtemp(prefix=EXTRACT_TMP_PREFIX)
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

        src_root, incoming, err = _scan_incoming(tmp_path, temp_dir, DEFAULT_SKIP_DIRS)
        if err:
            logmod.error("http-updater", f"apply failed: {err}")
            _update_state({
                "last_pull_ts":  time.time(),
                "last_pull_ok":  False,
                "last_pull_msg": err,
            })
            return {"ok": False, "error": err}

        # Only gate on modified/missing: those are the cases where pull would
        # overwrite or fail to restore user work. "added" files are user-created
        # and never touched by the updater (it only writes files present in
        # the tarball), so they don't need to block the pull.
        if not force:
            edits = local_edits(incoming=incoming)
            c = edits["counts"]
            if edits["has_baseline"] and (c["modified"] > 0 or c["missing"] > 0):
                msg = (f"local edits detected: {c['modified']} modified, "
                       f"{c['missing']} missing. pass force=true to overwrite.")
                logmod.warn("http-updater", msg)
                return {
                    "ok":             False,
                    "error":          msg,
                    "requires_force": True,
                    "edits":          edits,
                }

        _copy_incoming(src_root, incoming, REPO_DIR)
        msg = f"synced {branch} ({len(incoming)} files)"
        logmod.ok("http-updater", msg)
        # Persist the per-file SHA snapshot so the next pull can detect local
        # edits. Failure to write the baseline is non-fatal: the pull itself
        # succeeded.
        try:
            _write_baseline(incoming, branch=branch)
        except OSError as e:
            logmod.warn("http-updater", f"baseline write failed (non-fatal): {e}")
        # The pull just made the local tree the remote tip, so the check-derived
        # fields have to say so too. status() reads `behind` and `etag` straight
        # from state and the settings panel renders `behind` ("up to date" only at
        # 0) and `etag` as the version, so clearing update_available alone left the
        # panel reading "N behind" at the old SHA until the next 30-min check.
        _update_state({
            "last_pull_ts":         time.time(),
            "last_pull_ok":         True,
            "last_pull_msg":        msg,
            "pulled_etag":          post_etag,
            "pulled_last_modified": post_lm,
            "pulled_branch":        branch,
            # The commit the tarball came from. Without it the next check_update
            # recomputes `behind` against `.git/HEAD` -- which a tarball never
            # advances -- and overwrites the 0 below with the same stale count,
            # so the panel reads "N behind" again seconds after a good update.
            "pulled_commit":        _remote_branch_sha(branch),
            "update_available":     False,
            "behind":               0,
            "etag":                 post_etag,
            "last_modified":        post_lm,
        })

        # After a successful pull, invalidate the launcher's requirements-hash
        # sentinel so the NEXT boot re-runs pip install. requirements.txt may
        # have changed in the tarball; the launcher's fast-path skips pip if
        # the hash still matches its stored copy, so we clear the sentinel to
        # force a fresh check. Non-fatal on failure: the launcher will
        # re-check on its own if the hash mismatches. This is what makes
        # "run install helper after updating" cheap: the launcher already has
        # all the platform-specific install logic; we just re-arm it.
        try:
            sentinel = os.path.join(REPO_DIR, "venv", ".req_hash")
            if os.path.exists(sentinel):
                os.remove(sentinel)
                logmod.info("http-updater", "cleared venv/.req_hash; launcher will re-run pip on next boot")
        except OSError as e:
            logmod.warn("http-updater", f"could not clear .req_hash (non-fatal): {e}")

        # Fire a dep snapshot log so the analytics/telemetry stream captures
        # the post-pull state. Never blocks and never surfaces to the user:
        # the frontend's Detect Hardware / auto-tune paths still handle the
        # interactive case. This is just diagnostic bread-crumbs so we can
        # tell after-the-fact whether a pull left a box with a broken torch.
        try:
            from veritate_core.plugin import deps as deps_mod
            snap = deps_mod.status_snapshot()
            logmod.info("http-updater",
                        f"post-pull dep state: "
                        f"torch_cuda_ok={snap['torch'].get('cuda_available')} "
                        f"needs_torch_cuda={snap.get('needs_torch_cuda')}")
        except Exception as e:
            logmod.info("http-updater", f"post-pull dep probe skipped: {e}")

        if (reload or settings_mod.get().get("auto_reload_on_update")) and _RELOAD_HOOK is not None:
            logmod.warn("http-updater", "reload hook firing after update")
            try:
                _RELOAD_HOOK()
            except Exception as e:
                logmod.error("http-updater", f"reload hook failed: {e}")
        return {"ok": True, "status": status(), "copied": len(incoming)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
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
        # A commit on the old branch is not a valid base for the new one.
        "pulled_commit":       None,
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
