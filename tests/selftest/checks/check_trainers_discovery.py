# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - every trainer subdir under trainers/ (except common, corpus) has a manifest
#   and a plugin entry. manifest must be valid JSON.
# tests/selftest/checks/check_trainers_discovery.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "plugins"
EXCLUDE_DIRS    = {"common", "corpus"}
MANIFEST_NAME   = "manifest.json"
PLUGIN_NAME     = "trainer.py"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every trainers/<name>/ (excluding common, corpus) has manifest.json + trainer.py
    and the manifest parses as JSON."""
    root = _ctx.TRAINERS_DIR
    if not os.path.isdir(root):
        return _status.skip("trainers_discovery", f"trainers dir missing: {root}")
    discovered = []
    failures   = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub) or name in EXCLUDE_DIRS or name.startswith("_"):
            continue
        manifest = os.path.join(sub, MANIFEST_NAME)
        plugin   = os.path.join(sub, PLUGIN_NAME)
        if not os.path.isfile(manifest):
            failures.append(f"{name}: missing {MANIFEST_NAME}")
            continue
        if not os.path.isfile(plugin):
            failures.append(f"{name}: missing {PLUGIN_NAME}")
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                json.load(fh)
        except Exception as exc:
            failures.append(f"{name}: manifest parse error: {exc}")
            continue
        discovered.append(name)
    if failures:
        return _status.fail("trainers_discovery", failures[0],
                            {"failures": failures, "discovered": discovered})
    return _status.ok("trainers_discovery",
                      f"{len(discovered)} trainers ok",
                      {"trainers": discovered})
