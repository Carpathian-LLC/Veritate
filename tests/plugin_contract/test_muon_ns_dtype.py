# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - torch.optim.Muon orthogonalizes in bf16 unconditionally. On a device without bf16
#   acceleration that addmm drops to a serial reference path: one 1024x4096 weight
#   costs 203.9 s on one core against 0.775 s across seven in fp32 (i7-9700T,
#   2026-08-24), which is hours per optimizer step on a 270M model. Such a device must
#   get the vendored Muon in fp32. Pins the dtype policy, the implementation pick, and
#   fp32/bf16 agreement so the swap does not change what the optimizer computes.
# tests/plugin_contract/test_muon_ns_dtype.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.plugin import optim

# ------------------------------------------------------------------------------------
# Constants

COEF = optim.MUON_NS_COEFFICIENTS
STEPS = optim.MUON_NS_STEPS
EPS = optim.MUON_EPS

# ------------------------------------------------------------------------------------
# Functions


class _Args:
    base_lr = 1e-4
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    use_8bit_adam = False


class _Tiny(torch.nn.Module):
    """One 2D hidden weight (Muon's territory) plus its 1D bias (AdamW's)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(16, 32)


def test_cpu_orthogonalizes_in_fp32():
    """A CPU has no bf16 acceleration, so Newton-Schulz must not run in bf16."""
    assert optim.ns_dtype("cpu") is torch.float32


def test_accelerated_devices_keep_bf16():
    """bf16 is the upstream Muon dtype and stays wherever the device is fast at it."""
    assert optim.ns_dtype("mps") is torch.bfloat16


def test_cpu_gets_the_vendored_muon_so_the_dtype_is_ours():
    """torch's Muon hardcodes bf16 and takes no dtype, so CPU cannot use it."""
    muon = optim.build_muon(_Tiny(), _Args(), "cpu").muon
    assert isinstance(muon, optim._VendoredMuon)
    assert muon.param_groups[0]["ns_dtype"] is torch.float32


def test_fp32_orthogonalization_matches_bf16_within_bf16_precision():
    """Swapping the working dtype must not change the update beyond bf16's own error."""
    torch.manual_seed(0)
    grad = torch.randn(64, 128)
    bf16 = _zeropower(grad, torch.bfloat16).float()
    fp32 = _zeropower(grad, torch.float32).float()
    assert torch.allclose(bf16, fp32, atol=0.05)


def test_orthogonalization_returns_its_working_dtype():
    """The caller adds the update straight onto fp32 weights; the dtype must follow."""
    grad = torch.randn(8, 16)
    assert _zeropower(grad, torch.float32).dtype is torch.float32
    assert _zeropower(grad, torch.bfloat16).dtype is torch.bfloat16


def _zeropower(grad, dtype):
    return optim._zeropower_via_newtonschulz(grad, COEF, STEPS, EPS, dtype)
