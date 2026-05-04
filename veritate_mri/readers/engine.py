# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read the engine binary registry at veritate_engine/engine_versions.json.
# veritate_mri/readers/engine.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def manifest():
    try:
        with open(paths.ENGINE_VERSIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"current": None, "engines": []}


def engines():
    return manifest().get("engines", [])


def by_path(abs_path):
    for e in engines():
        if e.get("path") and os.path.abspath(e["path"]) == abs_path:
            return e
    return None
