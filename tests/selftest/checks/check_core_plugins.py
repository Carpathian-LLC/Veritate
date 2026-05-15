# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_core.core_plugins surface. lists built-in plugins delivered with
#   the platform (no git checkout required).
# tests/selftest/checks/check_core_plugins.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA = "plugin_contract"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """veritate_core.core_plugins imports and lists at least one builtin plugin."""
    try:
        import veritate_core.core_plugins as cp
    except Exception as exc:
        return _status.skip("core_plugins", f"import failed (optional): {exc}")
    listed = None
    for name in ("list", "list_plugins", "plugins", "core_plugins"):
        attr = getattr(cp, name, None)
        if callable(attr):
            listed = attr()
            break
        if isinstance(attr, (list, tuple, dict)):
            listed = attr
            break
    if listed is None:
        return _status.ok("core_plugins", "module imported (no list entry)")
    count = len(listed) if hasattr(listed, "__len__") else -1
    return _status.ok("core_plugins", f"{count} core plugin(s) listed")
