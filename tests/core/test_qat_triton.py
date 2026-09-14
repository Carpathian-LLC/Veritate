# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the fused Triton fake-quant kernels. Rule 25: a kernel ships with a scalar reference
#   and a bitwise-identity check against it. The parity tests need CUDA and Triton and
#   skip everywhere else, which is honest — they are the check, and they run on the box
#   that has the hardware. The gating logic is pure and always runs: a CPU tensor, a
#   missing Triton, or the env override must all fall back to the unfused path in qat.py,
#   because dispatching a CUDA kernel on a CPU tensor is a crash and silently skipping the
#   fallback is a wrong number.
# tests/core/test_qat_triton.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core import qat, qat_triton

# ------------------------------------------------------------------------------------
# Constants

needs_triton = pytest.mark.skipif(
    not (qat_triton.TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="fused path needs CUDA and Triton")

# ------------------------------------------------------------------------------------
# Functions


def test_a_cpu_tensor_never_takes_the_fused_path():
    """The kernels are CUDA-only; dispatching one on a CPU tensor is a crash."""
    assert qat_triton.triton_enabled(torch.zeros(4)) is False


def test_the_env_override_disables_it(monkeypatch):
    """VERITATE_NO_TRITON is the escape hatch when a kernel misbehaves on a box."""
    monkeypatch.setenv(qat_triton._ENV_DISABLE, "1")
    assert qat_triton.triton_enabled(torch.zeros(4)) is False


@pytest.mark.parametrize("value", ["", "0", "false", "False"])
def test_the_override_is_off_for_falsey_values(monkeypatch, value):
    """An unset-looking value must not disable the fast path by accident."""
    monkeypatch.setenv(qat_triton._ENV_DISABLE, value)
    monkeypatch.setattr(qat_triton, "TRITON_AVAILABLE", True)
    cpu = torch.zeros(4)
    assert qat_triton.triton_enabled(cpu) == cpu.is_cuda


def test_without_triton_nothing_is_enabled(monkeypatch):
    monkeypatch.setattr(qat_triton, "TRITON_AVAILABLE", False)
    assert qat_triton.triton_enabled(torch.zeros(4)) is False


def test_the_two_paths_share_their_constants():
    """The fused kernel and the reference must quantize to the same levels and the same
    fixed activation scale, or a model trains against one scheme and exports as another."""
    assert qat_triton.ACT_INT8_SCALE == qat.ACT_INT8_SCALE
    assert qat_triton.INT8_MAX == qat.INT8_MAX


@needs_triton
@pytest.mark.parametrize("shape", [(64,), (7, 129), (3, 5, 1025)])
def test_the_fused_activation_quant_is_bitwise_identical(shape):
    x = torch.randn(shape, device="cuda") * 3.0
    assert torch.equal(qat_triton.fake_quant_act_triton(x), qat.fake_quant_act(x))


@needs_triton
def test_the_fused_weight_quant_is_bitwise_identical():
    w = torch.randn(128, 256, device="cuda")
    assert torch.equal(qat_triton.fake_quant_weight_triton(w), qat.fake_quant_weight(w))


@needs_triton
def test_the_straight_through_gradient_matches_the_reference():
    """The backward is the clamp-gated straight-through estimator; a mismatch here trains
    a different model rather than the same one faster."""
    x = torch.randn(512, device="cuda", requires_grad=True)
    y = torch.randn(512, device="cuda", requires_grad=True)
    with torch.no_grad():
        y.copy_(x)
    qat_triton.fake_quant_act_triton(x).sum().backward()
    qat.fake_quant_act(y).sum().backward()
    assert torch.equal(x.grad, y.grad)
