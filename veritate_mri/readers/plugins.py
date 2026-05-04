# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - walk plugins/ recursively. two plugin forms:
#   * single-file:  plugins/<name>.py  paired with plugins/<name>.json
#                   (or any depth: plugins/<group>/<name>.py + .json)
#   * bundle:       plugins/<name>/plugin.py + plugins/<name>/manifest.json
#                   + optional sibling corpus/ folder. bundles are
#                   self-contained units; ship the folder, get the trainer +
#                   its data + its docs in one drop.
# - the manifest is a plain JSON file. plugin code never has to expose a
#   MANIFEST symbol; the dashboard reads the .json directly.
# - subfolders are pure organization. the manifest's "kind" field is the only
#   source of truth for what a plugin is; folder names mean nothing to the
#   platform. ids are path-from-plugins-root with forward slashes.
# - skips entries starting with _ or . (dunder, hidden, __pycache__, etc.)
# veritate_mri/readers/plugins.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

PLUGINS_ROOT     = paths.PLUGINS_ROOT
BUNDLE_ENTRY     = "plugin.py"
BUNDLE_MANIFEST  = "manifest.json"
BUNDLE_CORPUS    = "corpus"

# directories under plugins/ that are NOT plugins. the scanner skips them
# (and does not recurse into them).
RESERVED_DIRS = {"corpus", "common", "__pycache__", ".git", "node_modules"}

# ------------------------------------------------------------------------------------
# Functions

def _read_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _id_from_rel(rel_path):
    return rel_path.replace(os.sep, "/")


def _record(plugin_id, path, manifest, bundle_dir=None):
    rec = {
        "id":          plugin_id,
        "file":        os.path.basename(path),
        "path":        path,
        "manifest":    manifest,
        "bundle_dir":  bundle_dir,
    }
    if bundle_dir:
        corpus_dir = os.path.join(bundle_dir, BUNDLE_CORPUS)
        rec["bundle_corpus_dir"] = corpus_dir if os.path.isdir(corpus_dir) else None
    else:
        rec["bundle_corpus_dir"] = None
    return rec


def _walk(rel_path, out):
    abs_dir = os.path.join(PLUGINS_ROOT, rel_path) if rel_path else PLUGINS_ROOT
    if not os.path.isdir(abs_dir):
        return
    for entry in sorted(os.listdir(abs_dir)):
        if entry.startswith("_") or entry.startswith("."):
            continue
        if entry in RESERVED_DIRS:
            continue
        entry_rel = os.path.join(rel_path, entry) if rel_path else entry
        entry_abs = os.path.join(PLUGINS_ROOT, entry_rel)
        if os.path.isfile(entry_abs) and entry.endswith(".py"):
            manifest_abs = entry_abs[:-3] + ".json"
            if os.path.isfile(manifest_abs):
                manifest = _read_manifest(manifest_abs)
                if manifest:
                    out.append(_record(_id_from_rel(entry_rel[:-3]), entry_abs, manifest))
            continue
        if os.path.isdir(entry_abs):
            plugin_py    = os.path.join(entry_abs, BUNDLE_ENTRY)
            manifest_abs = os.path.join(entry_abs, BUNDLE_MANIFEST)
            if os.path.isfile(plugin_py) and os.path.isfile(manifest_abs):
                manifest = _read_manifest(manifest_abs)
                if manifest:
                    out.append(_record(_id_from_rel(entry_rel), plugin_py, manifest, bundle_dir=entry_abs))
            else:
                _walk(entry_rel, out)


def scan():
    out = []
    if not os.path.isdir(PLUGINS_ROOT):
        return out
    _walk("", out)
    return out


def by_id(plugin_id):
    for p in scan():
        if p["id"] == plugin_id:
            return p
    return None
