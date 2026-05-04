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
#     weights:        per-tensor symmetric maxabs INT8
#     activations:    scale-32 INT8 (the residual stream's fixed scale)
#     RMSNorm weights: scale-64 INT8
#   The straight-through estimator passes gradients unchanged so the optimizer
#   continues to operate as if the rounding were absent.
# veritate/qat.py
# ------------------------------------------------------------------------------------

import torch

INT8_MAX        = 127
ACT_INT8_SCALE  = 32.0
LN_FIXED_SCALE  = 64.0


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
    max_abs = w.detach().abs().amax().clamp_min(1e-8)
    scale   = max_abs / INT8_MAX
    q       = round_ste(w / scale).clamp(-INT8_MAX, INT8_MAX)
    return q * scale


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
