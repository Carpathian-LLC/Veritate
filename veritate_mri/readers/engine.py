# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - describe the C engine for the running host: version from the platform ledger
#   versions.json, binary path from paths.engine_binary_path().
# - the descriptor lists whether or not the binary is built; consumers filter.
# veritate_mri/readers/engine.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

VERSION_KEY = "engine"

# ------------------------------------------------------------------------------------
# Functions

def version():
    try:
        with open(paths.VERSIONS_JSON_PATH, encoding="utf-8") as f:
            return json.load(f).get(VERSION_KEY)
    except (OSError, ValueError):
        return None


def engines():
    """The one engine, with `path` resolved for the running host. Consumers read
    `path` and never build one; the path is where the build lands whether or not
    it has been built yet."""
    return [{"version": version(), "path": paths.engine_binary_path()}]


def by_path(abs_path):
    for e in engines():
        if os.path.abspath(e["path"]) == abs_path:
            return e
    return None
