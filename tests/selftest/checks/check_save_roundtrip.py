# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - save() round-trip via the trainer surface. writes to a temp model dir
#   under models/_selftest_save/, loads the .pt back, compares state_dict.
# tests/selftest/checks/check_save_roundtrip.py
# ------------------------------------------------------------------------------------
# Imports

import os
import shutil

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "platform"
REQUIRES_TORCH  = True

MODEL_NAME      = "selftestsave_1m_fp32_v1"
STEP            = 1
TOLERANCE       = 1e-5
SKIP_DUMP_SET   = ("probe", "lens", "classroom", "grades", "concepts",
                   "surprise", "quant_kl", "generation")
ARGS_TEMPLATE   = {
    "description": "selftest checkpoint roundtrip",
    "shape": {"layers": 1, "hidden": 8, "ffn": 16, "heads": 2, "seq": 4, "vocab": 256},
}

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """save() writes a checkpoint, torch.load() restores it, parameter norms
    match to TOLERANCE."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        return _status.skip("save_roundtrip", f"torch unavailable: {exc}")

    from readers import paths
    from training import save as save_mod

    target_dir = paths.model_dir(MODEL_NAME)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)

    try:
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        pre_norms = {k: float(v.norm().item()) for k, v in m.state_dict().items()}

        path = save_mod.save(m, MODEL_NAME, STEP,
                             dump_set=SKIP_DUMP_SET,
                             args=ARGS_TEMPLATE)
        if not isinstance(path, str) or not os.path.isfile(path):
            return _status.fail("save_roundtrip", f"save returned {path!r}")

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if "model" not in loaded:
            return _status.fail("save_roundtrip", "loaded payload missing 'model' key")
        sd = loaded["model"]
        if set(sd.keys()) != set(pre_norms.keys()):
            return _status.fail("save_roundtrip",
                                f"key set drift: {sorted(set(sd) ^ set(pre_norms))[:4]}")
        for k, want in pre_norms.items():
            got = float(sd[k].norm().item())
            if abs(got - want) > TOLERANCE:
                return _status.fail("save_roundtrip",
                                    f"{k} norm drift {abs(got - want):.2e}")
        return _status.ok("save_roundtrip", f"{len(pre_norms)} params matched",
                          {"checkpoint": path, "step": loaded.get("step")})
    finally:
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
