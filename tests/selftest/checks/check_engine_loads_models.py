# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - bench each committed .bin in models/. either exit 0 or refuse with a known
#   error token. a segfault fails the check. slow: touches every model.
# tests/selftest/checks/check_engine_loads_models.py
# ------------------------------------------------------------------------------------
# Imports

import os
import subprocess

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA              = "engine"
SLOW              = True
REQUIRES_MODEL    = True
REQUIRES_ENGINE_BUILT = True

ENGINE_TIMEOUT_S  = 30
KNOWN_REJECTIONS  = (
    "act_boost", "magic version mismatch", "unknown version",
    "model_load:", "shape mismatch", "vocab", "version",
)
BIN_NAME          = "veritate.bin"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """for every veritate.bin under models/<name>/, engine bench 1 1 exits cleanly
    or with a known rejection token."""
    from readers import paths
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        return _status.skip("engine_loads_models", "engine binary not built")

    bins = _collect_bins(paths.MODELS_ROOT)
    if not bins:
        return _status.skip("engine_loads_models", "no .bin files in models/")

    errors = []
    for bin_path in bins:
        env = {**os.environ, "VERITATE_MODEL_PATH": bin_path}
        try:
            r = subprocess.run([exe, "bench", "1", "1"], capture_output=True, text=True,
                               timeout=ENGINE_TIMEOUT_S, env=env)
        except subprocess.TimeoutExpired:
            errors.append(f"{bin_path}: timeout")
            continue
        if r.returncode == 0:
            continue
        if r.returncode < 0:
            errors.append(f"{bin_path}: signal {r.returncode}")
            continue
        if not any(t in r.stderr for t in KNOWN_REJECTIONS):
            errors.append(f"{bin_path}: unknown error: {r.stderr[:300]}")

    if errors:
        return _status.fail("engine_loads_models", errors[0], {"errors": errors})
    return _status.ok("engine_loads_models", f"{len(bins)} bin(s) benched")


def _collect_bins(models_root):
    out = []
    if not os.path.isdir(models_root):
        return out
    for name in sorted(os.listdir(models_root)):
        sub = os.path.join(models_root, name)
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if fname.startswith("veritate") and fname.endswith(".bin"):
                out.append(os.path.join(sub, fname))
    return out
