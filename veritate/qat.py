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


def set_engine_faithful(module, value):
    """Opt-in: when True together with qat=True, attention fake_quants q/k/v/out
    so the PyTorch forward matches the C engine's INT8 attention path. The
    M1 sister agent traced the engine fluency gap to PyTorch leaving
    SDPA's q/k/v in fp32 while the engine stores them as INT8; this flag
    closes that divergence at training time so the trained checkpoint is
    faithful to the deployment."""
    v = bool(value)
    for m in module.modules():
        if hasattr(m, "engine_faithful"):
            m.engine_faithful = v
    return module


# ----------------------------------------------------------------------------
# Split-precision split-device training (invention #1, falsifier passed)
# ----------------------------------------------------------------------------
# The master weight is a bf16 nn.Parameter on CPU. Per forward, an INT8
# fake-quanted copy is shipped to the GPU. Backward returns the GPU grad to
# the CPU master via straight-through estimator. Optimizer state lives on
# CPU. Falsifier (experiments/inventions/split_precision.py) showed:
#   - converges within noise of standard QAT
#   - peak GPU VRAM drops 58% on a 25M model
#   - 1B-class training on 12 GB VRAM becomes feasible with this + grad-ckpt
# Tradeoff: wall time is slower in the unoptimized form (per-forward H2D
# weight ship). Not a problem for VRAM-bound training; address with CUDA
# streams + 8-bit Adam on CPU when wall starts dominating.
# ----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class _CrossDeviceFakeQuantCached(torch.autograd.Function):
    """Forward: dequantize a pre-shipped INT8 GPU buffer + scale to bf16 GPU.
    Backward: ship GPU grad back to CPU via STE so autograd accumulates onto
    the CPU master parameter.

    The cached q_gpu and scale_gpu are produced once per optimizer step (when
    the master is dirty) and reused across all TBPTT chunks within that step.
    See SplitLinear.forward + invalidate_split_caches().
    """
    @staticmethod
    def forward(ctx, w_cpu, q_gpu, scale_gpu):
        ctx.in_dtype = w_cpu.dtype
        return q_gpu.to(torch.bfloat16) * scale_gpu

    @staticmethod
    def backward(ctx, grad_out):
        grad_cpu = grad_out.detach().to("cpu", non_blocking=True).to(ctx.in_dtype)
        return grad_cpu, None, None


class SplitLinear(nn.Module):
    """Drop-in for nn.Linear / QuantLinear where the weight master lives on
    CPU as bf16 and only an INT8-quantized transient is shipped to GPU.

    Caching: the CPU->GPU INT8 ship happens once per optimizer step (cache miss)
    and is reused across every TBPTT chunk within that step (cache hits). After
    `opt.step()`, call `invalidate_split_caches(model)` so the next chunk's
    forward re-quantizes from the updated master.

    Optimizer state for the CPU bf16 master lives on CPU.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               dtype=torch.bfloat16, device="cpu"))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device="cpu"))
        else:
            self.bias = None
        # Carry these flags so set_qat / set_engine_faithful are no-ops here.
        self.qat        = False
        self.quant_mode = QUANT_MODE_INT8
        self.engine_faithful = False
        # GPU-side INT8 cache (None = dirty/missing)
        self._q_gpu     = None  # int8 [out, in] on GPU
        self._scale_gpu = None  # bf16 scalar on GPU
        self._b_gpu     = None  # bf16 [out] on GPU (mirrors bias)

    def _refresh_cache(self, gpu_device):
        w = self.weight.detach().to(torch.float32)
        max_abs = w.abs().amax().clamp_min(1e-8)
        scale = max_abs / float(INT8_MAX)
        q = (w / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        self._q_gpu     = q.to(gpu_device, non_blocking=True)
        self._scale_gpu = scale.to(gpu_device).to(torch.bfloat16)
        if self.bias is not None:
            self._b_gpu = self.bias.detach().to(gpu_device, non_blocking=True).to(torch.bfloat16)
        else:
            self._b_gpu = None

    def invalidate_cache(self):
        self._q_gpu     = None
        self._scale_gpu = None
        self._b_gpu     = None

    def forward(self, x):
        if self._q_gpu is None or self._q_gpu.device != x.device:
            self._refresh_cache(x.device)
        w_gpu = _CrossDeviceFakeQuantCached.apply(self.weight, self._q_gpu, self._scale_gpu)
        if w_gpu.dtype != x.dtype:
            w_gpu = w_gpu.to(x.dtype)
        b_gpu = None
        if self._b_gpu is not None:
            b_gpu = self._b_gpu.to(x.dtype) if self._b_gpu.dtype != x.dtype else self._b_gpu
        return F.linear(x, w_gpu, b_gpu)


def invalidate_split_caches(module):
    """Walk module and invalidate every SplitLinear's GPU cache. Call this
    once after each `opt.step()` so the next training step re-quantizes from
    the freshly-updated master."""
    n = 0
    for m in module.modules():
        if isinstance(m, SplitLinear):
            m.invalidate_cache()
            n += 1
    return n


def convert_to_split_precision(module):
    """Walk module, replace every QuantLinear with SplitLinear. Copies the
    existing weights (and biases) bf16-on-CPU so any prior state_dict load
    survives the conversion. Skips any QuantLinear whose weight is tied to
    an embedding (e.g. lm_head tied to tok_emb) — those keep their original
    QuantLinear behavior so the embedding stays on GPU and the tie holds.
    Returns the module and a count of replacements."""
    # Avoid an import cycle: veritate.model imports from veritate.qat.
    from veritate.model import QuantLinear

    # Identify embedding parameter ids so we can skip tied lm_heads.
    embedding_ids = set()
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            embedding_ids.add(id(m.weight))

    n = 0
    n_skipped_tied = 0
    def _convert(parent):
        nonlocal n, n_skipped_tied
        for child_name, child in list(parent.named_children()):
            if isinstance(child, QuantLinear):
                if id(child.weight) in embedding_ids:
                    n_skipped_tied += 1
                    continue
                new = SplitLinear(child.in_features, child.out_features,
                                  bias=(child.bias is not None))
                with torch.no_grad():
                    new.weight.copy_(child.weight.detach().to(torch.bfloat16).to("cpu"))
                    if child.bias is not None:
                        new.bias.copy_(child.bias.detach().to(torch.bfloat16).to("cpu"))
                setattr(parent, child_name, new)
                n += 1
            else:
                _convert(child)

    _convert(module)
    return module, n


def split_precision_param_groups(module):
    """Return (cpu_params, gpu_params) lists for the two-optimizer pattern.
    Caller builds one AdamW per group, on its own device."""
    cpu_params, gpu_params = [], []
    for p in module.parameters():
        if p.device.type == "cpu":
            cpu_params.append(p)
        else:
            gpu_params.append(p)
    return cpu_params, gpu_params


def set_quant_mode(module, mode):
    """Recursively set .quant_mode on every submodule that has one. Activation
    and RMSNorm scales stay INT8 — only the weight quantization changes."""
    if mode not in QUANT_MODES:
        raise ValueError(f"unknown quant_mode: {mode!r}; expected one of {QUANT_MODES}")
    for m in module.modules():
        if hasattr(m, "quant_mode"):
            m.quant_mode = mode
    return module
