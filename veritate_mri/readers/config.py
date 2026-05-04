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
