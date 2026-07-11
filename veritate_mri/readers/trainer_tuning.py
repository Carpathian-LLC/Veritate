# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Machine-local per-trainer tuning overrides, keyed by plugin_id. Auto-optimize
#   results and last-used launch args land here instead of the upstream-synced
#   trainer manifest, so a machine's benchmarked settings survive
#   /trainers/git/sync and never leak to other machines. trainers.scan() overlays
#   these onto each trainer's manifest defaults at read time.
# - Lives under REPO_ROOT/data (gitignored) alongside the other machine-local
#   stores (mri_settings.json, system_specs.json).
# veritate_mri/readers/trainer_tuning.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

TUNING_PATH = os.path.join(paths.REPO_ROOT, "data", "trainer_tuning.json")

# ------------------------------------------------------------------------------------
# Functions

def _read():
    if not os.path.isfile(TUNING_PATH):
        return {}
    try:
        with open(TUNING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data):
    os.makedirs(os.path.dirname(TUNING_PATH), exist_ok=True)
    tmp = TUNING_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, TUNING_PATH)


def args_for(plugin_id):
    entry = _read().get(plugin_id) or {}
    a = entry.get("args")
    return a if isinstance(a, dict) else {}


def save(plugin_id, args, measured=None):
    """Merge `args` into the trainer's tuning entry (last write wins per key).
    Returns True when the stored entry changed."""
    data = _read()
    entry = data.get(plugin_id) or {}
    cur = entry.get("args") if isinstance(entry.get("args"), dict) else {}
    merged = {**cur, **args}
    if merged == cur and (measured is None or measured == entry.get("measured")):
        return False
    entry["args"] = merged
    if measured is not None:
        entry["measured"] = measured
    data[plugin_id] = entry
    _write(data)
    return True
