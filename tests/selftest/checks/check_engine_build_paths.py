# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - engine source tree contract: build script + engine_versions.json exist;
#   per-arch kernel dirs are present. catches accidental deletes.
# tests/selftest/checks/check_engine_build_paths.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "engine"
ENGINE_PRIMARY  = "v1"
EXPECTED_FILES  = ("engine_versions.json",)
EXPECTED_DIRS   = ("build", "kernels", "src")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """veritate_engine/v1/ has engine_versions.json + build/kernels/src dirs."""
    root = os.path.join(_ctx.ENGINE_DIR, ENGINE_PRIMARY)
    if not os.path.isdir(root):
        return _status.fail("engine_build_paths", f"{root} missing")
    miss = []
    for f in EXPECTED_FILES:
        if not os.path.isfile(os.path.join(root, f)):
            miss.append(f"file: {f}")
    for d in EXPECTED_DIRS:
        if not os.path.isdir(os.path.join(root, d)):
            miss.append(f"dir: {d}")
    if miss:
        return _status.fail("engine_build_paths", miss[0], {"missing": miss})
    with open(os.path.join(root, "engine_versions.json"), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _status.ok("engine_build_paths", f"{ENGINE_PRIMARY}/ layout ok",
                      {"engine_versions": list(data) if isinstance(data, dict) else data})
