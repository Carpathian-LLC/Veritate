# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - generic per-extension supplemental data. an extension ships a data_catalog.json
#   declaring large optional datasets; this module reports local presence, downloads
#   the hosted archive, and deletes a dataset to reclaim disk. storage is per
#   extension at installed/<id>/data/extension_data/<source> (disposable cache,
#   gitignored); a dataset dir may itself be a symlink to an external drive.
# - mechanism is platform-owned and extension-agnostic; the catalog (what datasets,
#   urls) is owned by each extension. surfaced per-extension in the marketplace.
# extensions/data.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import ssl
import tarfile
import urllib.request
import zipfile

import certifi

from runtime import logs as logmod

from . import registry

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE   = "extensions"
DATA_CATALOG = "data_catalog.json"

EXTENSIONS_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTALLED_ROOT  = os.path.join(EXTENSIONS_ROOT, "installed")

DOWNLOAD_CTX     = ssl.create_default_context(cafile=certifi.where())
DOWNLOAD_UA      = "Mozilla/5.0"
DOWNLOAD_TIMEOUT = 120
TMP_BASENAME     = "_download"

# ------------------------------------------------------------------------------------
# Functions

def _entries(ext_id):
    manifest = registry.manifest_for(ext_id)
    if not manifest:
        return []
    path = os.path.join(manifest["_dir"], DATA_CATALOG)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("datasets") or []
    except (OSError, ValueError) as e:
        logmod.error(LOG_SOURCE, f"data_catalog read failed for {ext_id}: {e}")
        return []


def _data_root(ext_id):
    return os.path.join(INSTALLED_ROOT, ext_id, "data", "extension_data")


def _source_path(ext_id, source):
    if not source or not isinstance(source, str):
        return None
    base = os.path.normpath(_data_root(ext_id))
    target = os.path.normpath(os.path.join(base, source))
    if os.path.commonpath([base, target]) != base or target == base:
        return None
    return target


def _dir_stats(path):
    if not path or not os.path.isdir(path):
        return False, 0, 0
    files = total = 0
    with os.scandir(path) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".csv"):
                files += 1
                try:
                    total += e.stat().st_size
                except OSError:
                    pass
    return files > 0, files, total


def catalog(ext_id):
    out = []
    for e in _entries(ext_id):
        present, files, nbytes = _dir_stats(_source_path(ext_id, e.get("source")))
        out.append({
            **e,
            "present":      present,
            "files":        files,
            "size_gb":      round(nbytes / 1e9, 3),
            "downloadable": bool(e.get("url")),
        })
    return out


def _extract(archive, target):
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")


def download(ext_id, source):
    """Fetch the dataset's single hosted archive (tar.gz or zip) from its catalog url and extract the
    CSVs into installed/<id>/data/extension_data/<source>. The url is a bucket object; the archive
    holds the dataset's CSVs at top level."""
    entry = next((e for e in _entries(ext_id) if e.get("source") == source), None)
    if entry is None:
        return {"ok": False, "error": f"unknown dataset: {source!r}"}
    url = entry.get("url")
    if not url:
        return {"ok": False, "error": f"{entry.get('label', source)} is not hosted yet (placeholder)."}
    target = _source_path(ext_id, source)
    if not target:
        return {"ok": False, "error": f"invalid dataset source: {source!r}"}
    os.makedirs(target, exist_ok=True)
    tmp = os.path.join(target, TMP_BASENAME + (".zip" if url.lower().endswith(".zip") else ".tgz"))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": DOWNLOAD_UA})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT, context=DOWNLOAD_CTX) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        _extract(tmp, target)
    except Exception as e:
        return {"ok": False, "error": f"download failed: {e}"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    present, files, nbytes = _dir_stats(target)
    logmod.ok(LOG_SOURCE, f"downloaded {ext_id}/{source}: {files} files, {round(nbytes / 1e9, 3)} GB")
    return {"ok": present, "source": source, "files": files, "size_gb": round(nbytes / 1e9, 3)}


def delete(ext_id, source):
    if source not in {e.get("source") for e in _entries(ext_id)}:
        return {"ok": False, "error": f"unknown dataset: {source!r}"}
    target = _source_path(ext_id, source)
    if not target or not os.path.lexists(target):
        return {"ok": False, "error": f"{source} is not downloaded."}
    if os.path.islink(target):
        archive = os.path.realpath(target)
        os.unlink(target)
        return {"ok": True, "source": source, "deleted": True, "unlinked": True,
                "note": f"removed the link; archive left intact at {archive}"}
    _, _, nbytes = _dir_stats(target)
    shutil.rmtree(target)
    return {"ok": True, "source": source, "deleted": True, "reclaimed_gb": round(nbytes / 1e9, 3)}
