# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read versions.json, assert the required keys are present.
# tests/selftest/checks/check_versions_json.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA           = "platform"
REQUIRED_KEYS  = ("build", "engine", "mri", "format", "trainers")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """versions.json exists, parses, has every required version key."""
    path = _ctx.VERSIONS_JSON
    if not os.path.isfile(path):
        return _status.skip("versions_json", f"{path} missing")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        return _status.fail("versions_json", f"missing keys: {missing}", {"data": data})
    return _status.ok("versions_json", f"build={data['build']} engine={data['engine']}", {"data": data})
