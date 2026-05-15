# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - inference.addons registry: list_addons returns each manifest, instantiate()
#   loads one. exact addon set varies; just confirm the registry is functional.
# tests/selftest/checks/check_addons.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA   = "inference"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """inference.addons.list_addons works and returns a list."""
    try:
        import inference.addons as addons
    except Exception as exc:
        return _status.fail("addons", f"import failed: {exc}")
    list_fn = getattr(addons, "list_addons", None)
    if not callable(list_fn):
        return _status.fail("addons", "list_addons missing")
    listed = list_fn()
    if not isinstance(listed, (list, tuple)):
        return _status.fail("addons", f"list_addons returned {type(listed).__name__}")
    return _status.ok("addons", f"{len(listed)} addon(s) registered",
                      {"ids": [str(getattr(a, 'id', a))[:32] for a in listed][:6]})
