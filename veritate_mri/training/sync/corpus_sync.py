# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - apt-style corpus library. unlike trainers/ and models/ (which are full git
#   repos), corpora are large opaque binaries downloaded one at a time over
#   HTTP into trainers/corpus/<stem>_train.bin and <stem>_val.bin.
# - the catalog has three layers, merged by stem (later layers override):
#     1. local catalog file: corpus_catalog.json shipped next to this module.
#        single source of truth for shipped corpora. ships in the repo, gets
#        updated alongside the codebase, works offline.
#     2. remote catalog: optional JSON manifest fetched from corpus_catalog_url
#        in mri_settings.json. only used when the user opts in by setting that
#        URL. populates train_url for entries the local catalog leaves blank.
#        fetch failures are non-fatal; catalog_status surfaces them to the UI.
#     3. corpus_user_sources (mri_settings.json): per-machine entries the user
#        added by hand. takes precedence over remote and local.
# - format='raw_bytes' means the URL points at a single file written directly
#   as a uint8 byte stream. since Veritate trains byte-level (np.uint8 memmap),
#   plaintext bytes ARE tokens — no tokenizer step. val_split_ratio carves off
#   the tail of the downloaded train file as val.bin when no val_url is given.
# - install() does the HTTP download, optionally verifies sha256, and writes
#   atomically (.part -> rename). uninstall() deletes the two .bin files.
# - this module never touches git. it lives next to plugins_sync / models_sync
#   only because it shares the "settings page library card" UX.
# veritate_mri/sync/corpus_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
import zipfile

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

from readers import paths
from runtime import logs as logmod
from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

CORPUS_DIR = paths.CORPUS_ROOT

LOCAL_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_catalog.json")

CATALOG_FETCH_TIMEOUT_SECS  = 15
DOWNLOAD_TIMEOUT_SECS       = 60 * 60   # 1 hour cap per file
DOWNLOAD_CHUNK_BYTES        = 1024 * 1024

SUPPORTED_FORMATS = {"raw_bytes", "hf_dataset", "raw_bytes_zip"}

# Inserted between rows when streaming HF text columns into bytes. Keeps
# document boundaries visible to the byte-level model without inflating output
# size much.
HF_ROW_SEP = b"\n\n"

# Free-disk safety pad: refuse install if free space is below
# expected_size * DISK_FREE_HEADROOM. 1.2x leaves room for the .part file
# during atomic writes plus the val split.
DISK_FREE_HEADROOM = 1.2

# Anything above this without explicit user confirmation aborts. The dashboard
# is responsible for the confirm UX; if it forgets, the backend still refuses.
LARGE_DOWNLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def _load_local_catalog():
    """Read the JSON catalog shipped next to this module. Returns [] if the
    file is missing or unreadable so the rest of the system still works."""
    try:
        with open(LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logmod.warn("corpus-sync", f"local catalog unreadable ({LOCAL_CATALOG_PATH}): {e}")
        return []
    entries = data.get("corpora") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("stem")]

_LOCK = threading.RLock()
_LAST = {
    "ok":          None,
    "message":     "",
    "finished_at": None,
    "action":      None,
    "stem":        None,
}

_PROGRESS = {}  # stem -> {"bytes": int, "total": int|None, "started_at": float}

# ------------------------------------------------------------------------------------
# Helpers

def _record(action, ok, message, stem=None):
    with _LOCK:
        _LAST.update({
            "ok":          bool(ok),
            "message":     message,
            "finished_at": time.time(),
            "action":      action,
            "stem":        stem,
        })


def _train_path(stem):
    return os.path.join(CORPUS_DIR, f"{stem}{paths.CORPUS_TRAIN_SUFFIX}")


def _val_path(stem):
    return os.path.join(CORPUS_DIR, f"{stem}{paths.CORPUS_VAL_SUFFIX}")


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _entry_skeleton(src):
    """Normalize any catalog entry shape (local / remote / user) into the
    common shape the dashboard renders. Missing keys are filled with None so
    the JS side never has to defend against undefined."""
    val_split = src.get("val_split_ratio")
    try:
        val_split = float(val_split) if val_split is not None else None
    except (TypeError, ValueError):
        val_split = None
    return {
        "stem":            src.get("stem"),
        "label":           src.get("label") or src.get("stem"),
        "description":     src.get("description") or "",
        "format":          (src.get("format") or "raw_bytes"),
        # raw_bytes / direct download fields
        "train_url":       src.get("train_url") or None,
        "val_url":         src.get("val_url") or None,
        "val_split_ratio": val_split,
        "sha256_train":    src.get("sha256_train") or None,
        "sha256_val":      src.get("sha256_val") or None,
        # hf_dataset fields
        "hf_dataset":       src.get("hf_dataset") or None,
        "hf_config":        src.get("hf_config") or None,
        "hf_split_train":   src.get("hf_split_train") or None,
        "hf_split_val":     src.get("hf_split_val") or None,
        "hf_text_column":   src.get("hf_text_column") or None,
        "max_bytes_train":  src.get("max_bytes_train"),
        "max_bytes_val":    src.get("max_bytes_val"),
        # shared metadata
        "size_train":      src.get("size_train"),
        "size_val":        src.get("size_val"),
        "recommended_min_params": src.get("recommended_min_params"),
        "recommended_max_params": src.get("recommended_max_params"),
        "notes":           src.get("notes") or None,
    }


def _merge(into, src):
    """Overlay non-null fields from src onto into."""
    out = dict(into)
    for k, v in src.items():
        if v is None or v == "":
            continue
        out[k] = v
    return out


def _fetch_remote_catalog(url):
    """Returns (entries, error_message). entries is [] on failure."""
    if not url:
        return [], "no catalog URL configured"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Veritate-MRI"})
        with urllib.request.urlopen(req, timeout=CATALOG_FETCH_TIMEOUT_SECS, context=_SSL_CTX) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        return [], f"fetch failed: {type(e).__name__}: {e}"
    except json.JSONDecodeError as e:
        return [], f"catalog is not valid JSON: {e}"
    entries = data.get("corpora") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return [], "catalog JSON has no 'corpora' list"
    out = []
    for e in entries:
        if isinstance(e, dict) and e.get("stem"):
            out.append(_entry_skeleton(e))
    return out, None

# ------------------------------------------------------------------------------------
# Catalog assembly

def catalog():
    """Combined catalog: builtin <- remote <- user_sources, indexed by stem.
    Each entry is annotated with installed/disk metadata. catalog_status
    surfaces the remote fetch state so the UI can warn when links are
    unreachable."""
    cur = settings_mod.get()
    catalog_url = (cur.get("corpus_catalog_url") or "").strip()
    user_sources = cur.get("corpus_user_sources") or []
    if not isinstance(user_sources, list):
        user_sources = []

    by_stem = {}
    for e in _load_local_catalog():
        by_stem[e["stem"]] = _entry_skeleton(e)

    remote, remote_err = _fetch_remote_catalog(catalog_url)
    for e in remote:
        stem = e.get("stem")
        if not stem:
            continue
        by_stem[stem] = _merge(by_stem.get(stem, _entry_skeleton(e)), e)

    for e in user_sources:
        if not isinstance(e, dict):
            continue
        stem = (e.get("stem") or "").strip()
        if not stem:
            continue
        norm = _entry_skeleton(e)
        by_stem[stem] = _merge(by_stem.get(stem, norm), norm)

    out = []
    for stem in sorted(by_stem.keys()):
        entry = by_stem[stem]
        tp = _train_path(stem)
        vp = _val_path(stem)
        installed_train = os.path.isfile(tp)
        installed_val   = os.path.isfile(vp)
        entry["installed_train"]      = installed_train
        entry["installed_val"]        = installed_val
        entry["installed_size_train"] = _file_size(tp) if installed_train else None
        entry["installed_size_val"]   = _file_size(vp) if installed_val   else None
        entry["is_user_source"] = any(
            isinstance(s, dict) and s.get("stem") == stem for s in user_sources
        )
        entry["progress"] = _PROGRESS.get(stem)
        out.append(entry)

    probe = hf_probe()
    has_hf_entries = any(c["format"] == "hf_dataset" for c in out)
    return {
        "ok":              True,
        "catalog_url":     catalog_url,
        "catalog_status":  {"ok": remote_err is None and bool(catalog_url), "error": remote_err},
        "hf_available":    probe["available"],
        "hf_required":     has_hf_entries,
        "hf_probe":        probe,
        "user_sources":    list(user_sources),
        "corpora":         out,
        "last":            dict(_LAST),
    }

# ------------------------------------------------------------------------------------
# Install / uninstall

def _download(url, dest_path, stem, kind):
    """Stream url -> dest_path.part -> rename. Updates _PROGRESS as it runs."""
    parent = os.path.dirname(dest_path)
    os.makedirs(parent, exist_ok=True)
    tmp = dest_path + ".part"
    try:
        if os.path.isfile(tmp):
            os.remove(tmp)
    except OSError:
        pass
    started = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Veritate-MRI"})
        with urllib.request.urlopen(req, timeout=CATALOG_FETCH_TIMEOUT_SECS, context=_SSL_CTX) as resp:
            total = None
            try:
                total = int(resp.headers.get("Content-Length") or 0) or None
            except (TypeError, ValueError):
                total = None
            with _LOCK:
                _PROGRESS[stem] = {"kind": kind, "bytes": 0, "total": total, "started_at": started}
            wrote = 0
            deadline = started + DOWNLOAD_TIMEOUT_SECS
            with open(tmp, "wb") as out:
                while True:
                    if time.time() > deadline:
                        raise TimeoutError("download exceeded 1h cap")
                    chunk = resp.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    wrote += len(chunk)
                    with _LOCK:
                        p = _PROGRESS.get(stem)
                        if p is not None:
                            p["bytes"] = wrote
        os.replace(tmp, dest_path)
        return True, wrote, None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False, 0, f"{type(e).__name__}: {e}"


def hf_available():
    """True when the HuggingFace `datasets` library is importable in this
    process. Cheap probe."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def hf_probe():
    """Rich version of hf_available() — returns the running Python's
    executable path, whether `datasets` is importable in THIS process, the
    exact ImportError message if not, and a platform-correct install command
    that points at the same interpreter. Used by the dashboard to surface a
    fix that always lands in the right site-packages regardless of how Flask
    was launched (system python / venv / py launcher / Microsoft-Store stub /
    Conda / etc)."""
    import sys
    info = {
        "available":      False,
        "executable":     sys.executable,
        "version":        sys.version.split(" ", 1)[0],
        "platform":       sys.platform,
        "error":          None,
        "install_command": _suggest_install_command(),
    }
    try:
        import datasets  # noqa: F401
        info["available"] = True
        info["datasets_version"] = getattr(__import__("datasets"), "__version__", None)
    except ImportError as e:
        info["error"] = str(e)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _suggest_install_command():
    """Build the most-likely-to-work `pip install` command for the current
    interpreter. Uses sys.executable so the install lands in the same Python
    Flask is running in, regardless of platform or launcher."""
    import sys
    import shlex
    exe = sys.executable
    # Quote the path if it contains spaces (very common on Windows).
    if " " in exe and not exe.startswith('"'):
        exe = f'"{exe}"'
    return f"{exe} -m pip install -r requirements.txt"


def install_hf_deps(extra_packages=None):
    """Run `<sys.executable> -m pip install -r requirements.txt` (plus any
    extra_packages) as a subprocess and return {ok, stdout, stderr, returncode,
    command}. Because we use sys.executable, the install always lands in the
    Python that's running this Flask process — `import datasets` will succeed
    immediately after this returns ok=True, no restart needed."""
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    requirements_path = os.path.join(repo_root, "requirements.txt")
    if not os.path.isfile(requirements_path):
        return {
            "ok": False,
            "command": None,
            "stdout": "",
            "stderr": f"requirements.txt not found at {requirements_path}",
            "returncode": -1,
        }
    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    if extra_packages:
        cmd.extend(extra_packages)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0:
            # Newly-installed packages may not be visible to imports cached at
            # process start. Flush the finder caches so the next hf_probe()
            # in this same Flask process sees the install without restart.
            import importlib
            importlib.invalidate_caches()
        return {
            "ok":         proc.returncode == 0,
            "command":    " ".join(cmd),
            "stdout":     proc.stdout[-4000:] if proc.stdout else "",
            "stderr":     proc.stderr[-4000:] if proc.stderr else "",
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok":         False,
            "command":    " ".join(cmd),
            "stdout":     "",
            "stderr":     "pip install timed out after 600s",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "ok":         False,
            "command":    " ".join(cmd),
            "stdout":     "",
            "stderr":     f"{type(e).__name__}: {e}",
            "returncode": -1,
        }


def _extract_zip_largest_member(zip_path):
    """In-place: replace zip_path with the contents of its largest member.
    Used by raw_bytes_zip format (enwik8 etc). Streams via a .extract.part
    sibling and atomically replaces the original. Returns error message on
    failure, None on success."""
    if not os.path.isfile(zip_path):
        return f"zip file missing: {zip_path}"
    tmp = zip_path + ".extract.part"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if not members:
                return "zip archive is empty"
            largest = max(members, key=lambda m: m.file_size)
            with zf.open(largest) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=DOWNLOAD_CHUNK_BYTES)
        os.replace(tmp, zip_path)
        return None
    except (zipfile.BadZipFile, OSError) as e:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return f"{type(e).__name__}: {e}"


def _free_disk_bytes(path):
    try:
        return shutil.disk_usage(os.path.dirname(path) or ".").free
    except OSError:
        return None


def _disk_precheck(stem, expected_bytes):
    """Refuse install if the local filesystem can't fit the expected payload
    plus a 20% pad. Returns (ok, error_message_or_None)."""
    if not expected_bytes or expected_bytes <= 0:
        return True, None
    free = _free_disk_bytes(_train_path(stem))
    if free is None:
        return True, None  # don't block on stat failure
    needed = int(expected_bytes * DISK_FREE_HEADROOM)
    if free < needed:
        return False, (
            f"not enough free disk: have {free / 1e9:.1f} GB free, "
            f"need ~{needed / 1e9:.1f} GB ({expected_bytes / 1e9:.1f} GB corpus + "
            f"{int((DISK_FREE_HEADROOM - 1) * 100)}% pad). free up space and retry."
        )
    return True, None


def _install_hf_dataset(entry):
    """Stream rows from a HuggingFace dataset, encode the text column as UTF-8
    bytes, and write to <stem>_train.bin (and _val.bin if hf_split_val is set,
    or via val_split_ratio post-hoc). HuggingFace's `datasets` library is
    imported lazily — installing a corpus is the first time the user pays the
    import cost."""
    stem = entry["stem"]
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return {"ok": False, "error": "the `datasets` package is not installed. run: pip install -r requirements.txt"}

    hf_name   = entry.get("hf_dataset")
    hf_config = entry.get("hf_config")
    split_tr  = entry.get("hf_split_train") or "train"
    split_va  = entry.get("hf_split_val")
    col       = entry.get("hf_text_column") or "text"
    max_tr    = entry.get("max_bytes_train")
    max_va    = entry.get("max_bytes_val")
    val_split = entry.get("val_split_ratio")

    if not hf_name:
        return {"ok": False, "error": f"corpus '{stem}' has format=hf_dataset but no hf_dataset name"}

    # ---- train split ----
    train_path = _train_path(stem)
    tmp_train  = train_path + ".part"
    if os.path.isfile(tmp_train):
        try: os.remove(tmp_train)
        except OSError: pass
    started = time.time()
    with _LOCK:
        _PROGRESS[stem] = {"kind": "train", "bytes": 0, "total": max_tr, "started_at": started}

    try:
        try:
            ds = load_dataset(hf_name, hf_config, split=split_tr, streaming=True) if hf_config \
                 else load_dataset(hf_name, split=split_tr, streaming=True)
        except Exception as e:
            msg = f"load_dataset failed: {type(e).__name__}: {e}"
            logmod.error("corpus-sync", f"{stem}: {msg}")
            _record("install", False, msg, stem=stem)
            return {"ok": False, "error": msg}

        wrote = 0
        deadline = started + DOWNLOAD_TIMEOUT_SECS
        try:
            with open(tmp_train, "wb") as out:
                for row in ds:
                    if time.time() > deadline:
                        raise TimeoutError("hf install exceeded 1h cap; partial bytes discarded")
                    text = row.get(col)
                    if not text:
                        continue
                    if not isinstance(text, str):
                        text = str(text)
                    chunk = text.encode("utf-8", errors="replace")
                    out.write(chunk)
                    out.write(HF_ROW_SEP)
                    wrote += len(chunk) + len(HF_ROW_SEP)
                    with _LOCK:
                        p = _PROGRESS.get(stem)
                        if p is not None:
                            p["bytes"] = wrote
                    if max_tr and wrote >= max_tr:
                        break
        except (OSError, TimeoutError) as e:
            try:
                if os.path.isfile(tmp_train):
                    os.remove(tmp_train)
            except OSError:
                pass
            msg = f"hf train stream failed: {type(e).__name__}: {e}"
            logmod.error("corpus-sync", f"{stem}: {msg}")
            _record("install", False, msg, stem=stem)
            return {"ok": False, "error": msg}

        if wrote == 0:
            try: os.remove(tmp_train)
            except OSError: pass
            msg = f"hf dataset returned no text in column '{col}'"
            _record("install", False, msg, stem=stem)
            return {"ok": False, "error": msg}

        os.replace(tmp_train, train_path)
        logmod.ok("corpus-sync", f"{stem} train: {wrote/1e6:.1f} MB streamed from {hf_name}")

        # ---- val split ----
        if split_va:
            tmp_val = _val_path(stem) + ".part"
            with _LOCK:
                _PROGRESS[stem] = {"kind": "val", "bytes": 0, "total": max_va, "started_at": time.time()}
            try:
                vds = load_dataset(hf_name, hf_config, split=split_va, streaming=True) if hf_config \
                      else load_dataset(hf_name, split=split_va, streaming=True)
            except Exception as e:
                msg = f"hf val load failed: {type(e).__name__}: {e}"
                logmod.warn("corpus-sync", f"{stem}: {msg}")
                _record("install", True, f"train ok; {msg}", stem=stem)
                return {"ok": True, "warning": msg, "stem": stem, "bytes_train": wrote}
            v_wrote = 0
            try:
                with open(tmp_val, "wb") as out:
                    for row in vds:
                        text = row.get(col)
                        if not text:
                            continue
                        if not isinstance(text, str):
                            text = str(text)
                        chunk = text.encode("utf-8", errors="replace")
                        out.write(chunk)
                        out.write(HF_ROW_SEP)
                        v_wrote += len(chunk) + len(HF_ROW_SEP)
                        with _LOCK:
                            p = _PROGRESS.get(stem)
                            if p is not None:
                                p["bytes"] = v_wrote
                        if max_va and v_wrote >= max_va:
                            break
                os.replace(tmp_val, _val_path(stem))
                logmod.ok("corpus-sync", f"{stem} val: {v_wrote/1e6:.1f} MB streamed")
            except OSError as e:
                try:
                    if os.path.isfile(tmp_val):
                        os.remove(tmp_val)
                except OSError:
                    pass
                msg = f"hf val stream failed: {e}"
                logmod.warn("corpus-sync", f"{stem}: {msg}")
                _record("install", True, f"train ok; {msg}", stem=stem)
                return {"ok": True, "warning": msg, "stem": stem, "bytes_train": wrote}
        elif val_split is not None and 0.0 < val_split < 0.5:
            _, split_err = _split_val_from_train(stem, val_split)
            if split_err:
                logmod.warn("corpus-sync", f"{stem} val split failed: {split_err}")
                _record("install", True, f"train ok; val split failed: {split_err}", stem=stem)
                return {"ok": True, "warning": f"val split failed: {split_err}", "stem": stem, "bytes_train": wrote}

        _record("install", True, f"installed {stem} (hf_dataset)", stem=stem)
        return {"ok": True, "stem": stem, "bytes_train": wrote}
    finally:
        # _PROGRESS pop is handled by the caller in install().
        pass


def _split_val_from_train(stem, ratio):
    """After train.bin lands, carve the last `ratio` fraction of bytes off the
    end into val.bin and shrink train.bin to match. Used when the catalog
    entry has val_split_ratio set instead of a separate val_url. Returns
    (val_bytes, error_or_None)."""
    train_path = _train_path(stem)
    val_path   = _val_path(stem)
    try:
        size = os.path.getsize(train_path)
    except OSError as e:
        return 0, f"could not stat train file: {e}"
    if size <= 0:
        return 0, "train file is empty"
    val_bytes = int(size * float(ratio))
    if val_bytes <= 0:
        return 0, None
    if val_bytes >= size:
        return 0, "val_split_ratio would consume the entire train file"
    try:
        with open(train_path, "rb") as src:
            src.seek(size - val_bytes)
            tail = src.read(val_bytes)
        with open(val_path + ".part", "wb") as dst:
            dst.write(tail)
        os.replace(val_path + ".part", val_path)
        # Shrink train file so train and val don't overlap.
        with open(train_path, "rb+") as f:
            f.truncate(size - val_bytes)
        return val_bytes, None
    except OSError as e:
        try:
            if os.path.isfile(val_path + ".part"):
                os.remove(val_path + ".part")
        except OSError:
            pass
        return 0, f"val split write failed: {e}"


def install(entry):
    """entry: {stem, train_url, val_url?, val_split_ratio?, format?, sha256_train?, sha256_val?}.
    Downloads files into trainers/corpus/. Verifies sha256 if provided. The
    val file is optional — if val_split_ratio is set instead, the trailing
    fraction of the train file becomes val.bin. Returns {ok, ...}."""
    if not isinstance(entry, dict):
        return {"ok": False, "error": "install body must be a JSON object"}
    stem = (entry.get("stem") or "").strip()
    train_url = (entry.get("train_url") or "").strip()
    val_url   = (entry.get("val_url")   or "").strip()
    sha_train = (entry.get("sha256_train") or "").strip().lower() or None
    sha_val   = (entry.get("sha256_val")   or "").strip().lower() or None
    fmt       = (entry.get("format") or "raw_bytes").strip()
    val_split_raw = entry.get("val_split_ratio")
    try:
        val_split = float(val_split_raw) if val_split_raw is not None else None
    except (TypeError, ValueError):
        val_split = None

    if not stem:
        return {"ok": False, "error": "missing stem"}
    if "/" in stem or "\\" in stem or stem.startswith(".") or ":" in stem:
        return {"ok": False, "error": f"invalid stem: {stem!r}"}
    if fmt not in SUPPORTED_FORMATS:
        return {"ok": False, "error": f"format '{fmt}' is not supported by this build (supported: {sorted(SUPPORTED_FORMATS)})"}
    if val_split is not None and not (0.0 < val_split < 0.5):
        return {"ok": False, "error": f"val_split_ratio must be between 0 and 0.5 (got {val_split})"}

    # Disk-space precheck: estimate from declared size_train, plus val if we'll
    # download one. Caller (the dashboard) is expected to surface a confirm
    # dialog for large downloads; the backend additionally requires
    # confirm_large=true when expected size exceeds LARGE_DOWNLOAD_BYTES.
    expected = entry.get("size_train") or 0
    if entry.get("hf_split_val") or entry.get("val_url"):
        expected += entry.get("size_val") or 0
    if expected and expected > LARGE_DOWNLOAD_BYTES and not entry.get("confirm_large"):
        return {"ok": False, "error": f"this corpus is ~{expected/1e9:.1f} GB. confirm in the dashboard before installing.",
                "needs_confirm": True, "expected_bytes": expected}
    ok, derr = _disk_precheck(stem, expected)
    if not ok:
        return {"ok": False, "error": derr}

    # Format-specific arg requirements
    if fmt == "raw_bytes":
        if not train_url:
            return {"ok": False, "error": f"corpus '{stem}' has no train_url. set one in the catalog, or add the corpus as a custom source."}
    elif fmt == "hf_dataset":
        if not entry.get("hf_dataset"):
            return {"ok": False, "error": f"corpus '{stem}' has format=hf_dataset but no hf_dataset name"}

    with _LOCK:
        if stem in _PROGRESS:
            return {"ok": False, "error": f"download already in progress for {stem}"}
        _PROGRESS[stem] = {"kind": "train", "bytes": 0, "total": expected or None, "started_at": time.time()}

    try:
        if fmt == "hf_dataset":
            return _install_hf_dataset(entry)
        logmod.info("corpus-sync", f"installing {stem}: train={train_url}")
        ok, wrote, err = _download(train_url, _train_path(stem), stem, "train")
        if not ok:
            logmod.error("corpus-sync", f"{stem} train download failed: {err}")
            _record("install", False, err, stem=stem)
            return {"ok": False, "error": err}

        # raw_bytes_zip: the URL is a .zip; replace the downloaded bytes with
        # the largest member's contents. enwik8 ships this way (one big xml
        # file inside enwik8.zip).
        if fmt == "raw_bytes_zip":
            zip_path = _train_path(stem)
            extract_err = _extract_zip_largest_member(zip_path)
            if extract_err:
                logmod.error("corpus-sync", f"{stem} zip extract failed: {extract_err}")
                _record("install", False, extract_err, stem=stem)
                return {"ok": False, "error": extract_err}
            wrote = _file_size(zip_path) or wrote

        if sha_train:
            got = sc.sha256_file(_train_path(stem))
            if got != sha_train:
                try: os.remove(_train_path(stem))
                except OSError: pass
                msg = f"sha256 mismatch on train: expected {sha_train[:12]}, got {got[:12]}"
                logmod.error("corpus-sync", f"{stem}: {msg}")
                _record("install", False, msg, stem=stem)
                return {"ok": False, "error": msg}

        if val_url:
            logmod.info("corpus-sync", f"installing {stem}: val={val_url}")
            ok, _, err = _download(val_url, _val_path(stem), stem, "val")
            if not ok:
                # train succeeded; val failed. Surface as warning, not fatal.
                logmod.warn("corpus-sync", f"{stem} val download failed: {err}")
                _record("install", True, f"train ok; val failed: {err}", stem=stem)
                return {"ok": True, "warning": f"val download failed: {err}", "stem": stem}
            if sha_val:
                got = sc.sha256_file(_val_path(stem))
                if got != sha_val:
                    try: os.remove(_val_path(stem))
                    except OSError: pass
                    msg = f"sha256 mismatch on val: expected {sha_val[:12]}, got {got[:12]}"
                    logmod.warn("corpus-sync", f"{stem}: {msg}")
                    _record("install", True, f"train ok; {msg}", stem=stem)
                    return {"ok": True, "warning": msg, "stem": stem}
        elif val_split is not None:
            logmod.info("corpus-sync", f"{stem}: splitting last {val_split*100:.1f}% off train as val")
            val_bytes, split_err = _split_val_from_train(stem, val_split)
            if split_err:
                logmod.warn("corpus-sync", f"{stem} val split failed: {split_err}")
                _record("install", True, f"train ok; val split failed: {split_err}", stem=stem)
                return {"ok": True, "warning": f"val split failed: {split_err}", "stem": stem}

        logmod.ok("corpus-sync", f"installed {stem}")
        _record("install", True, f"installed {stem}", stem=stem)
        return {"ok": True, "stem": stem, "bytes_train": wrote}
    finally:
        with _LOCK:
            _PROGRESS.pop(stem, None)


def uninstall(stem):
    stem = (stem or "").strip()
    if not stem:
        return {"ok": False, "error": "missing stem"}
    if "/" in stem or "\\" in stem or stem.startswith(".") or ":" in stem:
        return {"ok": False, "error": f"invalid stem: {stem!r}"}
    removed = []
    for label, p in (("train", _train_path(stem)), ("val", _val_path(stem))):
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(label)
            except OSError as e:
                msg = f"failed to remove {p}: {e}"
                logmod.error("corpus-sync", msg)
                _record("uninstall", False, msg, stem=stem)
                return {"ok": False, "error": msg}
    if not removed:
        msg = f"{stem} is not installed"
        _record("uninstall", False, msg, stem=stem)
        return {"ok": False, "error": msg}
    logmod.ok("corpus-sync", f"uninstalled {stem} ({', '.join(removed)})")
    _record("uninstall", True, f"removed: {', '.join(removed)}", stem=stem)
    return {"ok": True, "stem": stem, "removed": removed}

# ------------------------------------------------------------------------------------
# Settings mutators

def set_catalog_url(url):
    url = (url or "").strip()
    if url and not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "catalog URL must start with http:// or https://"}
    settings_mod.update({"corpus_catalog_url": url})
    return {"ok": True, "catalog_url": url}


def add_user_source(entry):
    if not isinstance(entry, dict):
        return {"ok": False, "error": "source must be a JSON object"}
    stem = (entry.get("stem") or "").strip()
    if not stem:
        return {"ok": False, "error": "missing stem"}
    if "/" in stem or "\\" in stem or stem.startswith(".") or ":" in stem:
        return {"ok": False, "error": f"invalid stem: {stem!r}"}
    train_url = (entry.get("train_url") or "").strip()
    if not train_url:
        return {"ok": False, "error": "missing train_url"}
    if not (train_url.startswith("http://") or train_url.startswith("https://")):
        return {"ok": False, "error": "train_url must start with http:// or https://"}
    val_url = (entry.get("val_url") or "").strip()
    if val_url and not (val_url.startswith("http://") or val_url.startswith("https://")):
        return {"ok": False, "error": "val_url must start with http:// or https://"}

    norm = {
        "stem":         stem,
        "label":        (entry.get("label") or stem).strip(),
        "description":  (entry.get("description") or "").strip(),
        "train_url":    train_url,
        "val_url":      val_url or None,
        "sha256_train": (entry.get("sha256_train") or "").strip().lower() or None,
        "sha256_val":   (entry.get("sha256_val")   or "").strip().lower() or None,
    }
    cur = settings_mod.get()
    sources = list(cur.get("corpus_user_sources") or [])
    sources = [s for s in sources if not (isinstance(s, dict) and s.get("stem") == stem)]
    sources.append(norm)
    settings_mod.update({"corpus_user_sources": sources})
    return {"ok": True, "stem": stem}


def remove_user_source(stem):
    stem = (stem or "").strip()
    if not stem:
        return {"ok": False, "error": "missing stem"}
    cur = settings_mod.get()
    sources = list(cur.get("corpus_user_sources") or [])
    new_sources = [s for s in sources if not (isinstance(s, dict) and s.get("stem") == stem)]
    if len(new_sources) == len(sources):
        return {"ok": False, "error": f"no user source with stem {stem!r}"}
    settings_mod.update({"corpus_user_sources": new_sources})
    return {"ok": True, "stem": stem}
