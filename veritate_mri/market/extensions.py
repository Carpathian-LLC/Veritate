# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Extension datasets: optional, downloadable market data (stocks, forex, the broader
#   crypto sets, ...) that lives under external_data/extension_data/<source> as a
#   disposable cache. Active training/serving sources (crypto_of, funding, sentiment,
#   live) are NOT extensions and are never touched here.
# - Mirrors the corpus-library pattern (catalog / install / uninstall) but over raw CSV
#   dataset dirs instead of trainer .bin corpora. Powers the Extensions section in the
#   settings corpus-library card.
# - catalog() reports each entry's local presence + size on disk. delete() reclaims a
#   dataset's disk (it re-downloads from the catalog url). download() pulls the hosted
#   archive; entries with a null url are placeholders (not hosted yet).
# veritate_mri/market/extensions.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil

import data as md

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(HERE, "extensions_catalog.json")

# ------------------------------------------------------------------------------------
# Functions

def _entries():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return (json.load(f) or {}).get("extensions") or []


def _ext_dir(source):
    """Logical dir for an extension source under EXTENSION_DIR. Validates the normalized
    (not realpath) path so a `..` traversal in `source` is rejected while a dataset dir
    that is itself a symlink (e.g. to an external SSD) still resolves."""
    if not source or not isinstance(source, str):
        return None
    base = os.path.normpath(md.EXTENSION_DIR)
    target = os.path.normpath(os.path.join(md.EXTENSION_DIR, source))
    if os.path.commonpath([base, target]) != base or target == base:
        return None
    return target


def _dir_stats(path):
    """(present, files, bytes) for a dataset dir without walking file contents."""
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


def catalog():
    """Catalog entries enriched with live local status (present / file count / size)."""
    out = []
    for e in _entries():
        src = e.get("source")
        present, files, nbytes = _dir_stats(_ext_dir(src))
        out.append({
            **e,
            "present": present,
            "files": files,
            "size_gb": round(nbytes / 1e9, 3),
            "downloadable": bool(e.get("url")),
        })
    return {"ok": True, "extensions": out}


def delete(source):
    """Remove an extension dataset to reclaim disk. Only catalog sources, only under
    extension_data. A real local dir is removed (reclaims that disk). A symlinked dir
    (a dataset the user parked on an external drive) only has its link removed; the
    underlying archive is left intact, so a dashboard click never wipes an external store."""
    if source not in {e.get("source") for e in _entries()}:
        return {"ok": False, "error": f"unknown extension: {source!r}"}
    target = _ext_dir(source)
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


def download(source):
    """Download an extension's hosted archive into extension_data/<source>. Entries with
    a null url are placeholders (not hosted yet)."""
    entry = next((e for e in _entries() if e.get("source") == source), None)
    if entry is None:
        return {"ok": False, "error": f"unknown extension: {source!r}"}
    if not entry.get("url"):
        return {"ok": False, "error": f"{entry.get('label', source)} is not hosted yet (placeholder)."}
    return {"ok": False, "error": "download wiring pending: set the catalog url + fetch implementation."}
