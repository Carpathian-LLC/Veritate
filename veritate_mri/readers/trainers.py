# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - walk trainers/ recursively. two plugin forms:
#   * single-file:  trainers/<name>.py  paired with trainers/<name>.json
#                   (or any depth: trainers/<group>/<name>.py + .json)
#   * bundle:       trainers/<name>/trainer.py + trainers/<name>/manifest.json
#                   + optional sibling corpus/ folder. bundles are
#                   self-contained units; ship the folder, get the trainer +
#                   its data + its docs in one drop.
# - the manifest is a plain JSON file. plugin code never has to expose a
#   MANIFEST symbol; the dashboard reads the .json directly.
# - subfolders are pure organization. the manifest's "kind" field is the only
#   source of truth for what a plugin is; folder names mean nothing to the
#   platform. ids are path-from-plugins-root with forward slashes.
# - skips entries starting with _ or . (dunder, hidden, __pycache__, etc.)
# veritate_mri/readers/trainers.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re

from . import paths, trainer_tuning

# ------------------------------------------------------------------------------------
# Constants

PLUGINS_ROOT     = paths.PLUGINS_ROOT
BUNDLE_ENTRY     = "trainer.py"
BUNDLE_MANIFEST  = "manifest.json"
BUNDLE_CORPUS    = "corpus"

# directories under trainers/ that are NOT plugins. the scanner skips them
# (and does not recurse into them).
RESERVED_DIRS = {"corpus", "common", "__pycache__", ".git", "node_modules"}

# Synthetic in-tree trainer. Shown to the user as a regular trainer so they can
# train / refine / continue any size from the dashboard without dropping a
# plugin folder. The dashboard form is schema-driven (TRAINER_SCHEMA in
# static/index.js), so all knobs render and route through the same CLI surface.
NATIVE_TRAINER_ID   = "native/trainer"
# The one trainer. Lives in the synced trainers checkout, not in veritate_mri:
# it is owned by the canonical Veritate-Trainers repo and ships its own size table.
NATIVE_TRAINER_PATH = os.path.normpath(os.path.join(paths.REPO_ROOT, "trainers", "common", "vanilla_trainer.py"))

# Single owner of the size -> shape table (rule 20), and it is DATA, not code:
# veritate_mri/data/trainer_sizes.json, or any path the user puts in the
# trainer_sizes_path setting. vanilla_trainer.py resolves --size against the same
# dict, so the offered set can never drift from the supported set.
TRAINER_SIZES_FILE    = "trainer_sizes.json"
SETTING_SIZES_PATH    = "trainer_sizes_path"
SIZES_KEY             = "sizes"


def native_sizes_path():
    try:
        from runtime import settings as _settings
        override = (_settings.get().get(SETTING_SIZES_PATH) or "").strip()
    except Exception:
        override = ""
    if override:
        return override
    return os.path.join(paths.REPO_ROOT, "trainers", "common", TRAINER_SIZES_FILE)


SHARED_DEFAULTS_KEY = "shared_defaults"
SHAPE_KEY           = "shape"
DEFAULTS_KEY        = "defaults"


def load_sizes_doc():
    """The whole size document, read fresh so an edit lands without a restart."""
    with open(native_sizes_path(), encoding="utf-8") as f:
        return json.load(f)


def load_native_sizes():
    """Size -> shape. The shape table every caller has always consumed."""
    return {name: entry[SHAPE_KEY] for name, entry in load_sizes_doc()[SIZES_KEY].items()}


def size_defaults(size=None):
    """Tuned defaults for one size: shared defaults, then that size's overrides.
    Returns the shared set alone when `size` is unknown, so a caller asking for a
    size that was retuned away still gets something sane rather than nothing."""
    doc = load_sizes_doc()
    out = dict(doc.get(SHARED_DEFAULTS_KEY) or {})
    entry = (doc[SIZES_KEY].get(size) or {}) if size else {}
    out.update(entry.get(DEFAULTS_KEY) or {})
    if size:
        out["size"] = size
    return out


def all_size_defaults():
    """size -> merged defaults, for the Training tab to switch on without a round
    trip per size."""
    doc = load_sizes_doc()
    shared = dict(doc.get(SHARED_DEFAULTS_KEY) or {})
    out = {}
    for name, entry in doc[SIZES_KEY].items():
        merged = dict(shared)
        merged.update(entry.get(DEFAULTS_KEY) or {})
        merged["size"] = name
        out[name] = merged
    return out


NATIVE_SIZES = load_native_sizes()

NATIVE_DEFAULT_SIZE  = "85m"
NATIVE_DEFAULT_SEQ   = 1024
NATIVE_DEFAULT_VOCAB = 256   # byte-level; mirrors veritate_core.model.VOCAB_BYTE_LEVEL

NATIVE_TRAINER_MANIFEST = {
    "name":        "Native trainer (no plugin)",
    "description": "Train, continue, or refine any size from the dashboard. Canonical Veritate (GELU FFN + "
                   "RMSNorm + learned pos-emb + tied LM head); QAT-aware; same save.save / append_train_row "
                   "contract as a plugin.",
    "kind":        "trainer",
    "bench":       True,
    "flow":        ["scratch", "continue"],
    "sizes": NATIVE_SIZES,
    "defaults": size_defaults(NATIVE_DEFAULT_SIZE),
    "size_defaults": all_size_defaults(),   # Training tab switches on this
}

# ------------------------------------------------------------------------------------
# Functions

def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
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
        if entry.startswith(("_", ".")):
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
    """Synthetic trainer entry: no manifest on disk, persisted defaults live
    in memory only (a future refinement could mirror them to a JSON next to
    vanilla_trainer.py). The runner builds argv off `path` like any plugin."""
    return {
        "id":                NATIVE_TRAINER_ID,
        "file":              os.path.basename(NATIVE_TRAINER_PATH),
        "path":              NATIVE_TRAINER_PATH,
        "manifest":          dict(NATIVE_TRAINER_MANIFEST),  # copy so callers can't mutate the constant
        "bundle_dir":        None,
        "bundle_corpus_dir": None,
        "native":            True,
    }


SIZE_UNITS = {"m": 1_000_000, "b": 1_000_000_000}


def _size_key(rec):
    # Headline scale from the canonical id token ("10m", "1b3", "50b"), so picker
    # order matches the displayed name regardless of which default size is set.
    tok = (rec.get("id") or "").rsplit("/", 1)[-1].rsplit("_", 1)[-1]
    m = re.match(r"^(\d+)([mb])(\d*)$", tok)
    if m:
        whole, unit, frac = m.groups()
        scaled = float(whole) + (float(frac) / 10.0 if frac else 0.0)
        return scaled * SIZE_UNITS[unit]
    # Non-scale trainers (SFT, etc.): fall back to declared max param count.
    sizes = (rec.get("manifest") or {}).get("sizes") or {}
    vals = [(v.get("params") or v.get("active_params") or 0) for v in sizes.values()]
    return float(max(vals)) if vals else 0.0


def scan():
    out = []
    # Native trainer first: surfaces "no plugin needed" at the top of the picker.
    if os.path.isfile(NATIVE_TRAINER_PATH):
        out.append(_native_record())
    if not os.path.isdir(PLUGINS_ROOT):
        return out
    plugins = []
    _walk("", plugins)
    # Picker order = model size (default-size param count), ascending. Stable sort
    # keeps the alphabetical _walk order for equal-size plugins.
    plugins.sort(key=_size_key)
    out.extend(plugins)
    _overlay_tuning(out)
    return out


def _overlay_tuning(records):
    """Overlay machine-local tuning onto each record's manifest defaults so the
    dashboard form prefills a machine's benchmarked settings. Only keys already
    present in the manifest defaults are replaced; the on-disk manifest is never
    touched."""
    for rec in records:
        tuned = trainer_tuning.args_for(rec["id"])
        if not tuned:
            continue
        manifest = dict(rec.get("manifest") or {})
        defaults = dict(manifest.get("defaults") or {})
        for k, v in tuned.items():
            if k in defaults:
                defaults[k] = v
        manifest["defaults"] = defaults
        rec["manifest"] = manifest


def by_id(plugin_id):
    for p in scan():
        if p["id"] == plugin_id:
            return p
    return None


def teaches(plugin_id):
    """Return the capability tier the trainer claims to teach. Falls back to
    the autocomplete default when no manifest declares it. Keeps the dashboard
    and save.py from needing to know which trainers are pretraining vs SFT."""
    from . import capabilities as caps
    p = by_id(plugin_id) if plugin_id else None
    if p is None:
        return caps.DEFAULT_TEACHES
    val = (p.get("manifest") or {}).get("teaches")
    if isinstance(val, str) and val in caps.TIERS:
        return val
    return caps.DEFAULT_TEACHES


def _coerce_like(template, v):
    # Dashboard form values arrive as strings; keep the manifest default's type
    # so parse_args registers the correct argparse type. bool subclasses int, check first.
    if isinstance(template, bool):
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    if isinstance(template, int):
        return int(v)
    if isinstance(template, float):
        return float(v)
    return v


def update_defaults(plugin_id, args):
    """Persist submitted args as machine-local tuning for this trainer. Only
    keys already present in the trainer's manifest defaults are kept (run-only
    fields like corpus/model/description/step are dropped) and each is coerced
    to the manifest default's type. Written to the local tuning store, never the
    upstream-synced manifest, so a machine's benchmarked settings survive
    /trainers/git/sync and don't leak to other machines. scan() overlays them
    onto the manifest defaults at read time."""
    plugin = by_id(plugin_id)
    if plugin is None or not isinstance(args, dict):
        return False
    defaults = (plugin.get("manifest") or {}).get("defaults") or {}
    if not isinstance(defaults, dict):
        return False
    keep = {}
    for k, v in args.items():
        if k not in defaults:
            continue
        try:
            keep[k] = _coerce_like(defaults[k], v)
        except (TypeError, ValueError):
            continue
    if not keep:
        return False
    return trainer_tuning.save(plugin_id, keep)
