# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The measured-precision policy and the explicit attention path, both born of one
#   profile (M2, 2026-09-05): bf16 matmuls at half the fp16 rate, sdpa upcasting to fp32
#   on MPS. Pins: the precision strings a trainer may ask for and what each resolves to
#   on a CPU; the label a run records; the probe's shape; and that the explicit attention
#   computes what sdpa computes, since one trains where the other would have.
# tests/plugin_contract/test_precision_and_attention.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core import model as model_mod
from veritate_core.plugin import hardware

# ------------------------------------------------------------------------------------
# Functions


def test_every_precision_choice_resolves_to_fp32_on_a_cpu():
    """A CPU runs no half precision: whatever is asked for, autocast stays off."""
    for want in hardware.PRECISION_CHOICES:
        assert hardware.resolve_precision(want, "cpu") is None
    assert hardware.resolve_precision("", "cpu") is None


def test_auto_is_a_choice_and_the_probe_is_shaped_for_the_log_line():
    assert "auto" in hardware.PRECISION_CHOICES and "fp16" in hardware.PRECISION_CHOICES
    dtype, rates = hardware.half_precision_probe("cpu")
    assert dtype is None and rates == {}
    assert hardware.half_precision_probe("cpu") == (dtype, rates)          # cached, not re-run


def test_precision_labels_name_what_the_run_computes_in():
    assert hardware.precision_label(None) == "fp32"
    assert hardware.precision_label(torch.float16) == "fp16"
    assert hardware.precision_label(torch.bfloat16) == "bf16"


def test_explicit_attention_computes_what_sdpa_computes():
    """The explicit form is sdpa without the fused kernel: same numbers, fp32 tolerance."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 17, 8) for _ in range(3))
    want = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    got = model_mod.explicit_attention(q, k, v)
    assert torch.allclose(got, want, atol=1e-5)


def test_the_fused_path_stays_for_causal_fp32_and_devices_with_a_kernel():
    """Text runs and fp32 runs are untouched: only non-causal half precision on a device
    named in EXPLICIT_ATTENTION_DEVICES takes the explicit form."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 9, 4) for _ in range(3))
    assert "mps" in model_mod.EXPLICIT_ATTENTION_DEVICES and "cpu" not in model_mod.EXPLICIT_ATTENTION_DEVICES
    causal = model_mod.attention(q, k, v, True)
    assert torch.allclose(causal, torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6)
    full = model_mod.attention(q, k, v, False)
    assert torch.allclose(full, torch.nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-6)


def test_a_bidirectional_model_forwards_the_same_through_both_paths(monkeypatch):
    """The picture model's forward with the explicit path forced on equals the fused one."""
    torch.manual_seed(0)
    m = model_mod.Veritate(256, 32, 2, 64, 4, 24, causal=False).eval()
    toks = torch.randint(0, 255, (2, 24))
    with torch.no_grad():
        fused = m(toks)[0]
        monkeypatch.setattr(model_mod, "attention", lambda q, k, v, causal: model_mod.explicit_attention(q, k, v))
        explicit = m(toks)[0]
    assert torch.allclose(fused, explicit, atol=1e-4)
