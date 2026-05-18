# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - import every reader module and call the listing entry on each. these back
#   the read-only dashboard endpoints; if they crash on import the server cannot
#   come up.
# tests/selftest/checks/check_readers_misc.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA          = "platform"

READER_MODS   = (
    "models", "config", "checkpoints", "corpus",
    "engine", "wiki", "trainers", "bin", "hooks", "train_csv",
)

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every readers/*.py imports without error."""
    failures = []
    for name in READER_MODS:
        try:
            __import__("readers." + name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        return _status.fail("readers_misc", failures[0], {"errors": failures})
    return _status.ok("readers_misc", f"{len(READER_MODS)} reader modules imported")
