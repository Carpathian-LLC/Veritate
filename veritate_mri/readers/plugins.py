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

# Synthetic in-tree trainer. Shown to the user as a regular trainer so they can
# train / refine / continue any size from the dashboard without dropping a
# plugin folder. The dashboard form is schema-driven (TRAINER_SCHEMA in
# static/index.js), so all knobs render and route through the same CLI surface.
NATIVE_TRAINER_ID   = "native/trainer"
NATIVE_TRAINER_PATH = os.path.normpath(os.path.join(paths.MRI_ROOT, "native_trainer.py"))
NATIVE_TRAINER_MANIFEST = {
    "name":        "Native trainer (no plugin)",
    "description": "Train, continue, or refine any size from the dashboard. Canonical Veritate (GELU FFN + RMSNorm + learned pos-emb + tied LM head); QAT-aware; same save.save / append_train_row contract as a plugin.",
    "kind":        "trainer",
    "flow":        ["scratch", "continue"],
    "sizes": {
        "5m":   {"layers":  6, "hidden":  256, "ffn":  1024, "heads":  4, "params":      5000000},
        "7m":   {"layers":  8, "hidden":  256, "ffn":  1024, "heads":  4, "params":      7000000},
        "10m":  {"layers":  8, "hidden":  320, "ffn":  1280, "heads":  8, "params":     10000000},
        "20m":  {"layers":  8, "hidden":  512, "ffn":  2048, "heads":  8, "params":     20000000},
        "30m":  {"layers": 10, "hidden":  512, "ffn":  2048, "heads":  8, "params":     31000000},
        "50m":  {"layers": 10, "hidden":  640, "ffn":  2560, "heads": 10, "params":     50000000},
        "70m":  {"layers": 12, "hidden":  640, "ffn":  2560, "heads": 10, "params":     70000000},
        "80m":  {"layers": 12, "hidden":  768, "ffn":  3072, "heads": 12, "params":     85000000},
        "85m":  {"layers": 12, "hidden":  768, "ffn":  3072, "heads": 12, "params":     85000000},
        "120m": {"layers": 12, "hidden":  896, "ffn":  3584, "heads": 14, "params":    115000000},
        "160m": {"layers": 12, "hidden": 1024, "ffn":  4096, "heads": 16, "params":    162000000},
        "200m": {"layers": 16, "hidden": 1024, "ffn":  4096, "heads": 16, "params":    202000000},
        "350m": {"layers": 24, "hidden": 1024, "ffn":  4096, "heads": 16, "params":    330000000},
        "400m": {"layers": 24, "hidden": 1280, "ffn":  5120, "heads": 20, "params":    472000000},
        "800m": {"layers": 28, "hidden": 1536, "ffn":  6144, "heads": 24, "params":    793000000},
        "1b3":  {"layers": 24, "hidden": 2048, "ffn":  8192, "heads": 16, "params":   1300000000},
        "2b":   {"layers": 24, "hidden": 2560, "ffn": 10240, "heads": 20, "params":   2700000000},
        "3b":   {"layers": 32, "hidden": 2560, "ffn": 10240, "heads": 32, "params":   2900000000},
        "4b5":  {"layers": 36, "hidden": 3200, "ffn": 12800, "heads": 25, "params":   4400000000},
        "7b":   {"layers": 32, "hidden": 4096, "ffn": 18432, "heads": 32, "params":   7000000000},
        "13b":  {"layers": 40, "hidden": 5120, "ffn": 21504, "heads": 40, "params":  13000000000},
        "30b":  {"layers": 60, "hidden": 6656, "ffn": 26624, "heads": 52, "params":  32000000000},
        "50b":  {"layers": 64, "hidden": 8192, "ffn": 32768, "heads": 64, "params":  52000000000}
    },
    "defaults": {
        "size":         "85m",
        "precision":    "bf16",
        "vocab":        256,
        "seq":          1024,
        "total_steps":  20000,
        "batch_size":   8,
        "n_chunks":     1,
        "base_lr":      3e-4,
        "min_lr":       3e-6,
        "warmup_steps": 200,
        "lr_schedule":  "wsd",
        "wsd_decay_frac": 0.1,
        "wsd_decay_kind": "sqrt",
        "weight_decay": 0.1,
        "beta1":        0.9,
        "beta2":        0.95,
        "label_smoothing": 0.0,
        "grad_clip":    1.0,
        "ckpt_every":   500,
        "log_every":    50,
        "eval_every":   500,
        "eval_iters":   16,
        "seed":         0,
        "use_act_ckpt": False,
        "qat_enabled":  False,
        "quant_mode":   "int8",
    },
}

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


def _native_record():
    """Synthetic trainer entry — no manifest on disk, persisted defaults live
    in memory only (a future refinement could mirror them to a JSON next to
    native_trainer.py). The runner builds argv off `path` like any plugin."""
    return {
        "id":                NATIVE_TRAINER_ID,
        "file":              os.path.basename(NATIVE_TRAINER_PATH),
        "path":              NATIVE_TRAINER_PATH,
        "manifest":          dict(NATIVE_TRAINER_MANIFEST),  # copy so callers can't mutate the constant
        "bundle_dir":        None,
        "bundle_corpus_dir": None,
        "native":            True,
    }


def scan():
    out = []
    # Native trainer first — surfaces "no plugin needed" at the top of the picker.
    if os.path.isfile(NATIVE_TRAINER_PATH):
        out.append(_native_record())
    if not os.path.isdir(PLUGINS_ROOT):
        return out
    _walk("", out)
    return out


def by_id(plugin_id):
    for p in scan():
        if p["id"] == plugin_id:
            return p
    return None


def _manifest_path(plugin):
    if plugin.get("bundle_dir"):
        return os.path.join(plugin["bundle_dir"], BUNDLE_MANIFEST)
    return plugin["path"][:-3] + ".json"


def update_defaults(plugin_id, args):
    """Merge submitted args into the plugin manifest's `defaults` block. Only
    keys already present in defaults are overwritten so run-only fields
    (corpus, model, description, step) do not pollute the schema."""
    plugin = by_id(plugin_id)
    if plugin is None or not isinstance(args, dict):
        return False
    if plugin.get("native"):
        # Native trainer has no on-disk manifest; defaults are constants.
        # Skip the merge so we never try to write outside MRI_ROOT.
        return False
    mpath = _manifest_path(plugin)
    if not os.path.isfile(mpath):
        return False
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        return False
    changed = False
    for k, v in args.items():
        if k in defaults and defaults[k] != v:
            defaults[k] = v
            changed = True
    if not changed:
        return False
    data["defaults"] = defaults
    tmp = mpath + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, mpath)
    except OSError:
        try: os.remove(tmp)
        except OSError: pass
        return False
    return True
