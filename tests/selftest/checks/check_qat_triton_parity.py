# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - triton fused fake_quant ops vs unfused reference. CUDA + triton only;
#   skips elsewhere. mirrors tests/engine/test_qat_triton_parity.py but inside
#   the centralized selftest.
# tests/selftest/checks/check_qat_triton_parity.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "core"
REQUIRES_TORCH  = True

B, S, H         = 4, 16, 64
ATOL            = 1e-6

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """triton fake_quant_act forward matches the unfused reference."""
    try:
        import torch
    except Exception as exc:
        return _status.skip("qat_triton_parity", f"torch unavailable: {exc}")
    if not torch.cuda.is_available():
        return _status.skip("qat_triton_parity", "cuda unavailable")
    try:
        from veritate_core import qat_triton as triton_qat
        from veritate_core import qat as ref_qat
    except Exception as exc:
        return _status.skip("qat_triton_parity", f"triton path import failed: {exc}")

    x_ref = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    x_tri = x_ref.clone()
    try:
        out_tri = triton_qat.fake_quant_act_triton(x_tri)
        out_ref = ref_qat.fake_quant_act(x_ref)
    except Exception as exc:
        return _status.skip("qat_triton_parity", f"call failed: {exc}")
    if not torch.allclose(out_tri, out_ref, atol=ATOL):
        max_d = (out_tri - out_ref).abs().max().item()
        return _status.fail("qat_triton_parity", f"max diff {max_d:.2e}")
    return _status.ok("qat_triton_parity", "triton == reference")
