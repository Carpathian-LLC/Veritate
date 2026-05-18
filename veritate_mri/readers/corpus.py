# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - discover corpus stems from two roots:
#     1. shared:   training/corpus/<stem>_train.bin
#     2. bundled:  trainers/<sub>/<plugin_dir>/corpus/<stem>_train.bin
#   bundled stems are namespaced as "<plugin_id>:<stem>" to avoid collisions and
#   so the dashboard can show which plugin shipped them.
# - resolve_paths(stem) returns the (train, val) on-disk paths for any stem,
#   shared or bundled.
# - usage(stem) scans every model's config.json for a matching corpus_sha256 so
#   the dashboard can show "shared with N models" and gate fair comparisons.
# veritate_mri/readers/corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import os

from . import paths, models, config as cfg_reader, trainers as plugins_reader

# ------------------------------------------------------------------------------------
# Constants

SHA256_CHUNK = 1024 * 1024

NAMESPACE_SEP = ":"

# ------------------------------------------------------------------------------------
# Functions

def _stems_in(root):
    if not os.path.isdir(root):
        return []
    out = []
    for fn in sorted(os.listdir(root)):
        if fn.endswith(paths.CORPUS_TRAIN_SUFFIX):
            out.append(fn[:-len(paths.CORPUS_TRAIN_SUFFIX)])
    return out


def _split_namespace(stem):
    if NAMESPACE_SEP in stem:
        plugin_id, _, leaf = stem.rpartition(NAMESPACE_SEP)
        return plugin_id, leaf
    return None, stem


def list_stems():
    out = []
    for s in _stems_in(paths.corpus_dir()):
        out.append({"stem": s, "source": "shared", "plugin_id": None, "label": s})
    for p in plugins_reader.scan():
        bdir = p.get("bundle_corpus_dir")
        if not bdir:
            continue
        for s in _stems_in(bdir):
            out.append({
                "stem":      f"{p['id']}{NAMESPACE_SEP}{s}",
                "source":    "bundle",
                "plugin_id": p["id"],
                "label":     f"{s} (bundled with {p['id']})",
            })
    return out


def has_val(stem):
    _, val = resolve_paths(stem)
    return val is not None


def resolve_paths(stem):
    plugin_id, leaf = _split_namespace(stem)
    if plugin_id is None:
        train = paths.corpus_train_path(leaf)
        val   = paths.corpus_val_path(leaf)
    else:
        rec = plugins_reader.by_id(plugin_id)
        if rec is None or not rec.get("bundle_corpus_dir"):
            return (None, None)
        bdir = rec["bundle_corpus_dir"]
        train = paths.bundled_corpus_train_path(bdir, leaf)
        val   = paths.bundled_corpus_val_path(bdir, leaf)
    if not os.path.isfile(train):
        return (None, None)
    return (train, val if os.path.isfile(val) else None)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA256_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_block(path):
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return {"path": path, "bytes": st.st_size, "mtime": st.st_mtime, "sha256": _sha256(path)}


def usage(stem):
    train_path, val_path = resolve_paths(stem)
    train = _file_block(train_path)
    val   = _file_block(val_path)
    if train is None:
        return None
    matches = []
    for name in models.list_models():
        cfg = cfg_reader.load(name) or {}
        ta  = cfg.get("training_args") or {}
        sha = ta.get("corpus_sha256")
        if sha and sha == train["sha256"]:
            matches.append({"name": name, "step": int(cfg.get("step") or 0)})
    matches.sort(key=lambda r: r["name"])
    plugin_id, _ = _split_namespace(stem)
    return {"stem": stem, "source": "bundle" if plugin_id else "shared",
            "plugin_id": plugin_id, "train": train, "val": val, "models": matches}
