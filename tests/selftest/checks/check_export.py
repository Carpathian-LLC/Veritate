# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - synthetic export to .bin. builds a tiny canonical Veritate, runs the
#   exporter, checks the first four bytes match the VRTE magic.
# tests/selftest/checks/check_export.py
# ------------------------------------------------------------------------------------
# Imports

import os
import shutil

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "export"
REQUIRES_TORCH  = True

MODEL_NAME      = "selftestexport_1m_fp32_v1"
MAGIC           = b"VRTE"
STEP            = 1
TINY_VOCAB      = 256
TINY_HIDDEN     = 32
TINY_LAYERS     = 2
TINY_FFN        = 64
TINY_HEADS      = 2
TINY_SEQ        = 16
SKIP_DUMP_SET   = ("probe", "lens", "classroom", "grades", "concepts",
                   "surprise", "quant_kl", "generation")
ARGS_TEMPLATE   = {
    "description": "selftest export roundtrip",
    "shape": {
        "vocab":  TINY_VOCAB, "hidden": TINY_HIDDEN, "layers": TINY_LAYERS,
        "ffn":    TINY_FFN,   "heads":  TINY_HEADS,  "seq":    TINY_SEQ,
    },
}

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """build a tiny Veritate, export to .bin, verify magic header VRTE."""
    try:
        import torch
    except Exception as exc:
        return _status.skip("export", f"torch unavailable: {exc}")

    from readers import paths
    from training import export as export_mod
    from training import save as save_mod
    from veritate_core.model import Veritate

    target_dir = paths.model_dir(MODEL_NAME)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)

    try:
        torch.manual_seed(0)
        m = Veritate(TINY_VOCAB, TINY_HIDDEN, TINY_LAYERS, TINY_FFN, TINY_HEADS, TINY_SEQ)
        save_mod.save(m, MODEL_NAME, STEP, dump_set=SKIP_DUMP_SET, args=ARGS_TEMPLATE)

        if not callable(getattr(export_mod, "export_checkpoint", None)):
            return _status.skip("export", "export_checkpoint missing")

        out = export_mod.export_checkpoint(MODEL_NAME, STEP)
        bin_path = out if isinstance(out, str) else paths.bin_path(MODEL_NAME)
        if not os.path.isfile(bin_path):
            return _status.fail("export", f"bin not written: {bin_path}")

        with open(bin_path, "rb") as fh:
            head = fh.read(4)
        if head != MAGIC:
            return _status.fail("export", f"magic {head!r} != {MAGIC!r}")
        return _status.ok("export", f"bin written ({os.path.getsize(bin_path)} bytes)",
                          {"path": bin_path})
    finally:
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)


