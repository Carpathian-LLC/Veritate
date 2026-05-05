# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Quantization-aware training (QAT) helpers shared by the canonical Veritate
#   model and any plugin that wants to wire QAT into a wrapper.
# - All ops are model-, size-, and precision-agnostic. They round and clip to
#   match the C engine's quantization scheme exactly:
#     weights (int8):    per-tensor symmetric maxabs INT8
#     weights (int4):    per-tensor symmetric maxabs INT4 (16 levels)
#     weights (ternary): BitNet b1.58 — per-tensor mean-abs scale, levels {-1,0,+1}
#     activations:       scale-32 INT8 (the residual stream's fixed scale)
#     RMSNorm weights:   scale-64 INT8
#   The straight-through estimator passes gradients unchanged so the optimizer
#   continues to operate as if the rounding were absent.
# - Ternary and INT4 are L3-fit accelerators. Ternary at 1.58 bits/param means
#   a 200M dense model fits 40 MB; a 1B 4-way MoE has 50 MB active. Engine
#   kernels for these live under documentation/kernels/.
# veritate/qat.py
# ------------------------------------------------------------------------------------

import torch

INT8_MAX        = 127
INT4_MAX        = 7
ACT_INT8_SCALE  = 32.0
LN_FIXED_SCALE  = 64.0

QUANT_MODE_INT8    = "int8"
QUANT_MODE_INT4    = "int4"
QUANT_MODE_TERNARY = "ternary"
QUANT_MODES        = (QUANT_MODE_INT8, QUANT_MODE_INT4, QUANT_MODE_TERNARY)


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def round_ste(x):
    return _RoundSTE.apply(x)


def fake_quant_weight(w):
    """Per-tensor symmetric maxabs INT8. Default Veritate quant scheme."""
    max_abs = w.detach().abs().amax().clamp_min(1e-8)
    scale   = max_abs / INT8_MAX
    q       = round_ste(w / scale).clamp(-INT8_MAX, INT8_MAX)
    return q * scale


def fake_quant_weight_int4(w):
    """Per-tensor symmetric maxabs INT4. 4 bits/param, 2x density vs INT8.
    Levels: {-7..+7}. Engine packs 2 weights per byte."""
    max_abs = w.detach().abs().amax().clamp_min(1e-8)
    scale   = max_abs / INT4_MAX
    q       = round_ste(w / scale).clamp(-INT4_MAX, INT4_MAX)
    return q * scale


def fake_quant_weight_ternary(w):
    """BitNet b1.58 ternary quantization. Levels {-1, 0, +1} with per-tensor
    mean-abs scale. ~1.58 bits/param (log2(3)). 5x density vs INT8.
    Engine packs 5 trits per byte (3^5 = 243 < 256)."""
    gamma = w.detach().abs().mean().clamp_min(1e-8)
    q     = round_ste(w / gamma).clamp(-1.0, 1.0)
    return q * gamma


def fake_quant_weight_mode(w, mode):
    """Dispatch by quant_mode string. Used by QuantLinear so a single layer
    can be flipped between schemes during ablations."""
    if mode == QUANT_MODE_INT8:
        return fake_quant_weight(w)
    if mode == QUANT_MODE_INT4:
        return fake_quant_weight_int4(w)
    if mode == QUANT_MODE_TERNARY:
        return fake_quant_weight_ternary(w)
    raise ValueError(f"unknown quant_mode: {mode!r}; expected one of {QUANT_MODES}")


def fake_quant_act(x, scale=ACT_INT8_SCALE):
    s = float(scale)
    q = round_ste(x * s).clamp(-INT8_MAX, INT8_MAX)
    return q / s


def fake_quant_ln_weight(w, scale=LN_FIXED_SCALE):
    s = float(scale)
    q = round_ste(w * s).clamp(-INT8_MAX, INT8_MAX)
    return q / s


def set_qat(module, value):
    """Recursively flip the .qat flag on every submodule that has one."""
    v = bool(value)
    for m in module.modules():
        if hasattr(m, "qat"):
            m.qat = v
    return module


def set_quant_mode(module, mode):
    """Recursively set .quant_mode on every submodule that has one. Activation
    and RMSNorm scales stay INT8 — only the weight quantization changes."""
    if mode not in QUANT_MODES:
        raise ValueError(f"unknown quant_mode: {mode!r}; expected one of {QUANT_MODES}")
    for m in module.modules():
        if hasattr(m, "quant_mode"):
            m.quant_mode = mode
    return module
