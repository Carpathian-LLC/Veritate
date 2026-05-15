# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the committed platform contracts under documentation/. these are referenced
#   from CLAUDE.md and the trainer plugin contract; if one is missing, an
#   external plugin author hits a dead link.
# tests/selftest/checks/check_documentation.py
# ------------------------------------------------------------------------------------
# Imports

import os

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA           = "docs"
DOCS_DIR       = "documentation"
REQUIRED_DOCS  = (
    "hooks/contract.md",
    "hooks/brain_hooks.md",
    "kernels/architecture.md",
    "kernels/engine_versions.md",
    "kernels/platforms.md",
    "trainers/contract.md",
)

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every documented contract file exists under documentation/."""
    root = os.path.join(_ctx.REPO_ROOT, DOCS_DIR)
    if not os.path.isdir(root):
        return _status.fail("documentation", f"{root} missing")
    missing = [p for p in REQUIRED_DOCS if not os.path.isfile(os.path.join(root, p))]
    if missing:
        return _status.fail("documentation", f"missing: {missing[0]}", {"missing": missing})
    return _status.ok("documentation", f"{len(REQUIRED_DOCS)} required docs present")
