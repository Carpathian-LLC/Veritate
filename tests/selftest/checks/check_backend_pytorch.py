# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the pytorch Brain backend imports + Brain class is callable. cold check
#   (no model load). loading a real ckpt is a deeper test; gate on first
#   available model name if present.
# tests/selftest/checks/check_backend_pytorch.py
# ------------------------------------------------------------------------------------
# Imports

import os

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA                 = "inference"
REQUIRES_TORCH       = True

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """import Brain from inference.backends.pytorch; assert the class has stream()
    and stream_fast() methods."""
    try:
        from inference.backends.pytorch import Brain
    except Exception as exc:
        return _status.fail("backend_pytorch", f"import failed: {exc}")
    if not callable(Brain):
        return _status.fail("backend_pytorch", "Brain is not callable")
    for attr in ("stream", "stream_fast"):
        if not hasattr(Brain, attr):
            return _status.fail("backend_pytorch", f"Brain.{attr} missing")
    available = _list_models(_ctx.MODELS_DIR)
    return _status.ok("backend_pytorch",
                      f"Brain class ok ({len(available)} model dir(s) discoverable)",
                      {"models": available[:8]})


def _list_models(root):
    if not os.path.isdir(root):
        return []
    return [n for n in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, n)) and not n.startswith("_")]
