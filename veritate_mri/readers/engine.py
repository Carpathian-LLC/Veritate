# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read the engine binary registry at veritate_engine/v1/engine_versions.json.
# - the registry carries version metadata only. the binary path is resolved per
#   host by paths.engine_binary_path(), so the registry stays arch-neutral.
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
        with open(paths.ENGINE_VERSIONS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"current": None, "engines": []}


def engines():
    """Registry entries with `path` filled in for the running host. Consumers read
    `path` and never build one; an entry whose binary is not built for this host
    still lists, with `path` pointing at where the build would land."""
    host_binary = paths.engine_binary_path()
    return [{"path": host_binary, **e} for e in manifest().get("engines", [])]


def by_path(abs_path):
    for e in engines():
        if e.get("path") and os.path.abspath(e["path"]) == abs_path:
            return e
    return None
