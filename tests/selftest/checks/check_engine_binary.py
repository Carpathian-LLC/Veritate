# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - invoke the engine binary in its no-arg self-test mode. checks kernel parity,
#   tokenizer round-trip, CPU dispatch print. skip if binary not built.
# tests/selftest/checks/check_engine_binary.py
# ------------------------------------------------------------------------------------
# Imports

import os
import subprocess

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA                = "engine"
REQUIRES_ENGINE_BUILT = True
ENGINE_TIMEOUT_S    = 30
VERIFY_TOKEN        = "verify OK"
CPU_TOKEN           = "cpu:"
FEATURES_TOKEN      = "features:"
DISPATCH_TOKEN      = "dispatch:"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """engine no-arg self-test exits 0 and prints verify / cpu / features /
    dispatch lines."""
    from readers import paths
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        return _status.skip("engine_binary", f"binary not built: {exe}")

    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=ENGINE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _status.fail("engine_binary", f"self-test exceeded {ENGINE_TIMEOUT_S}s")

    if r.returncode != 0:
        return _status.fail("engine_binary",
                            f"exit {r.returncode}",
                            {"stderr": r.stderr[:600]})

    out = r.stdout
    for tok in (VERIFY_TOKEN, CPU_TOKEN, FEATURES_TOKEN, DISPATCH_TOKEN):
        if tok not in out:
            return _status.fail("engine_binary", f"missing '{tok}' in stdout",
                                {"stdout_head": out[:600]})
    return _status.ok("engine_binary", "kernel verify + dispatch ok")
