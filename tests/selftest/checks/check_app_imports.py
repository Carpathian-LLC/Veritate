# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the dashboard top-level flask app imports cleanly. picks up every transitive
#   import problem from a single entry.
# tests/selftest/checks/check_app_imports.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA = "mri"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """veritate_mri/app.py imports without error and exposes a flask app."""
    try:
        import app as app_module
    except Exception as exc:
        return _status.fail("app_imports", f"import failed: {type(exc).__name__}: {exc}")
    app_obj = getattr(app_module, "app", None)
    if app_obj is None:
        return _status.fail("app_imports", "module has no 'app' attribute")
    rules = list(app_obj.url_map.iter_rules())
    return _status.ok("app_imports", f"flask app ok, {len(rules)} routes registered")
