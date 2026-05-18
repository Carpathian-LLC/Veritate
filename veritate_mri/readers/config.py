# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - load a model's config.json. small file, no cache.
# veritate_mri/readers/config.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def load(name):
    p = paths.config_path(name)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def training_kind(name):
    data = load(name) or {}
    return (data.get("training") or "", data.get("activation") or "")


def description(name):
    data = load(name) or {}
    for key in ("description", "stage_description"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v
    ta = data.get("training_args") or {}
    v = ta.get("description")
    return v if isinstance(v, str) and v else ""


def qat_enabled(name):
    """Authoritative QAT flag. True iff `qat_enabled` is True at the top level
    of the model's config.json OR inside `training_args`. Trainers write the
    nested form (vars(args) lands under training_args); save.py promotes it to
    top-level on every save so the dashboard and engine wiring read one key.
    The bin's act_boost is magnitude-derived and can be > 1 on legitimately
    QAT-trained checkpoints; the engine's act_boost heuristic must defer to
    this flag, not to act_boost."""
    data = load(name) or {}
    if data.get("qat_enabled") is True:
        return True
    ta = data.get("training_args") or {}
    return bool(isinstance(ta, dict) and ta.get("qat_enabled"))
