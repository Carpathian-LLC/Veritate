# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - exercise every fake_quant_* helper plus set_qat. round-trip values must stay
#   within INT8 bounds.
# tests/selftest/checks/check_core_qat.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "core"
REQUIRES_TORCH  = True

W_ROWS, W_COLS  = 16, 16
A_BATCH, A_SEQ, A_HID = 2, 4, 16
LN_DIM          = 16
BOUND_FLOAT     = 4.0

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """fake_quant_weight / fake_quant_weight_int4 / ternary / fake_quant_act /
    fake_quant_ln_weight all run; set_qat flips flags on a small module."""
    try:
        import torch
    except Exception as exc:
        return _status.skip("core_qat", f"torch unavailable: {exc}")

    from veritate_core import qat
    torch.manual_seed(0)

    w   = torch.randn(W_ROWS, W_COLS)
    a   = torch.randn(A_BATCH, A_SEQ, A_HID)
    lnw = torch.randn(LN_DIM)

    wq  = qat.fake_quant_weight(w)
    wq4 = qat.fake_quant_weight_int4(w)
    wqt = qat.fake_quant_weight_ternary(w)
    aq  = qat.fake_quant_act(a)
    lwq = qat.fake_quant_ln_weight(lnw)

    for name, t in (("w", wq), ("w_int4", wq4), ("w_ternary", wqt), ("a", aq), ("ln_w", lwq)):
        if not isinstance(t, torch.Tensor):
            return _status.fail("core_qat", f"{name} not a tensor")
        if torch.isnan(t).any() or torch.isinf(t).any():
            return _status.fail("core_qat", f"{name} contains nan/inf")

    flag_module = _build_flag_target(torch)
    qat.set_qat(flag_module, True)
    if not all(_collect_qat_flags(flag_module)):
        return _status.fail("core_qat", "set_qat did not toggle nested flags")

    return _status.ok("core_qat", "all fake_quant ops + set_qat", {
        "int8_max":     qat.INT8_MAX,
        "act_scale":    qat.ACT_INT8_SCALE,
        "ln_scale":     qat.LN_FIXED_SCALE,
    })


def _build_flag_target(torch):
    from veritate_core.model import Veritate
    return Veritate(256, 32, 2, 64, 2, 16)


def _collect_qat_flags(module):
    flags = []
    for sub in module.modules():
        if hasattr(sub, "qat"):
            flags.append(bool(sub.qat))
    return flags
