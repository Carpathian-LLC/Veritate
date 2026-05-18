# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - c_engine subprocess wrapper imports; class surface is intact. live spawn
#   requires both the engine binary and a real .bin model. skip otherwise.
# tests/selftest/checks/check_backend_c.py
# ------------------------------------------------------------------------------------
# Imports

import os

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA          = "inference"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """CTracedSubprocess import succeeds and exposes a close() method."""
    try:
        from inference.backends.c_engine import CTracedSubprocess
    except Exception as exc:
        return _status.fail("backend_c", f"import failed: {exc}")
    if not callable(CTracedSubprocess):
        return _status.fail("backend_c", "CTracedSubprocess not callable")
    for attr in ("close",):
        if not hasattr(CTracedSubprocess, attr):
            return _status.fail("backend_c", f"missing {attr}")

    from readers import paths
    exe = paths.engine_binary_path() if hasattr(paths, "engine_binary_path") else None
    have_exe = bool(exe and os.path.isfile(exe))
    return _status.ok("backend_c",
                      f"class ok (engine binary {'present' if have_exe else 'absent'})",
                      {"exe": exe, "exe_present": have_exe})
