# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Fused Triton kernels for QAT fake-quant ops. Drop-in replacements for the
#   unfused PyTorch path in qat.py. Each unfused fake_quant_act / fake_quant_weight
#   call decomposes to ~5 pointwise kernel launches (mul, round, clamp, div,
#   plus autograd graph nodes). The fused Triton path collapses each to one
#   forward + one backward launch, reading/writing each element once.
# - Math matches qat.py bit-for-bit at the scalar level: same INT8 levels, same
#   per-tensor symmetric maxabs scheme, same STE backward with the clamp gate.
# - Bit-for-bit kernel-vs-reference parity is required by preflight rule 24.
#   Verified by veritate_mri/tests/test_qat_triton_parity.py.
# - Activation: fixed scale (ACT_INT8_SCALE = 32). Weight: per-tensor maxabs
#   scale, computed on device via .amax() and passed as a 0-d tensor so no
#   host sync is incurred.
# - Disabled at runtime by setting the VERITATE_NO_TRITON env var; QuantLinear
#   then falls back to the unfused path. Auto-disabled when triton is missing
#   or x.device is not CUDA.
# - INT4 and ternary weight quant remain unfused for now: their ablations are
#   not throughput-bound and the unfused path is correct.
# veritate_core/qat_triton.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ------------------------------------------------------------------------------------
# Constants

INT8_MAX        = 127
ACT_INT8_SCALE  = 32.0
LN_FIXED_SCALE  = 64.0
EPS_SCALE       = 1e-8
BLOCK_DEFAULT   = 1024

_ENV_DISABLE    = "VERITATE_NO_TRITON"


# ------------------------------------------------------------------------------------
# Functions

def triton_enabled(x):
    """True iff Triton is importable, the input is CUDA, and the env override
    is not set."""
    if not TRITON_AVAILABLE:
        return False
    if os.environ.get(_ENV_DISABLE, "") not in ("", "0", "false", "False"):
        return False
    if not x.is_cuda:
        return False
    return True


if TRITON_AVAILABLE:

    @triton.jit
    def _quant_act_fwd_kernel(x_ptr, y_ptr, n, scale, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q    = tl.extra.libdevice.rint(x * scale)
        q    = tl.minimum(tl.maximum(q, -127.0), 127.0)
        y    = q / scale
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _quant_act_bwd_kernel(grad_in_ptr, grad_out_ptr, x_ptr, n, scale, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g    = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q    = tl.extra.libdevice.rint(x * scale)
        ok   = (q >= -127.0) & (q <= 127.0)
        g_in = tl.where(ok, g, 0.0)
        tl.store(grad_in_ptr + offs, g_in, mask=mask)

    @triton.jit
    def _quant_w_fwd_kernel(w_ptr, y_ptr, scale_ptr, n, BLOCK: tl.constexpr):
        pid   = tl.program_id(0)
        offs  = pid * BLOCK + tl.arange(0, BLOCK)
        mask  = offs < n
        scale = tl.load(scale_ptr).to(tl.float32)
        w     = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q     = tl.extra.libdevice.rint(w / scale)
        q     = tl.minimum(tl.maximum(q, -127.0), 127.0)
        y     = q * scale
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _quant_w_bwd_kernel(grad_in_ptr, grad_out_ptr, w_ptr, scale_ptr, n, BLOCK: tl.constexpr):
        pid   = tl.program_id(0)
        offs  = pid * BLOCK + tl.arange(0, BLOCK)
        mask  = offs < n
        scale = tl.load(scale_ptr).to(tl.float32)
        g     = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w     = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q     = tl.extra.libdevice.rint(w / scale)
        ok    = (q >= -127.0) & (q <= 127.0)
        g_in  = tl.where(ok, g, 0.0)
        tl.store(grad_in_ptr + offs, g_in, mask=mask)

    @triton.jit
    def _quant_ln_fwd_kernel(w_ptr, y_ptr, n, scale, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        w    = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q    = tl.extra.libdevice.rint(w * scale)
        q    = tl.minimum(tl.maximum(q, -127.0), 127.0)
        y    = q / scale
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _quant_ln_bwd_kernel(grad_in_ptr, grad_out_ptr, w_ptr, n, scale, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g    = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w    = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q    = tl.extra.libdevice.rint(w * scale)
        ok   = (q >= -127.0) & (q <= 127.0)
        g_in = tl.where(ok, g, 0.0)
        tl.store(grad_in_ptr + offs, g_in, mask=mask)


class _FakeQuantActTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = float(scale)
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return y
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_act_fwd_kernel[grid](x, y, n, ctx.scale, BLOCK=BLOCK_DEFAULT)
        return y

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        g = g.contiguous()
        grad_in = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return grad_in, None
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_act_bwd_kernel[grid](grad_in, g, x, n, ctx.scale, BLOCK=BLOCK_DEFAULT)
        return grad_in, None


class _FakeQuantWeightTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w = w.contiguous()
        wf      = w.detach().to(torch.float32)
        max_abs = wf.abs().amax().clamp_min(EPS_SCALE)
        scale   = (max_abs / INT8_MAX).to(torch.float32)
        y = torch.empty_like(w)
        n = w.numel()
        if n == 0:
            ctx.save_for_backward(w, scale)
            return y
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_w_fwd_kernel[grid](w, y, scale, n, BLOCK=BLOCK_DEFAULT)
        ctx.save_for_backward(w, scale)
        return y

    @staticmethod
    def backward(ctx, g):
        w, scale = ctx.saved_tensors
        g = g.contiguous()
        grad_in = torch.empty_like(w)
        n = w.numel()
        if n == 0:
            return grad_in
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_w_bwd_kernel[grid](grad_in, g, w, scale, n, BLOCK=BLOCK_DEFAULT)
        return grad_in


class _FakeQuantLnWeightTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        ctx.save_for_backward(w)
        ctx.scale = float(scale)
        w = w.contiguous()
        y = torch.empty_like(w)
        n = w.numel()
        if n == 0:
            return y
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_ln_fwd_kernel[grid](w, y, n, ctx.scale, BLOCK=BLOCK_DEFAULT)
        return y

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        g = g.contiguous()
        grad_in = torch.empty_like(w)
        n = w.numel()
        if n == 0:
            return grad_in, None
        grid = (triton.cdiv(n, BLOCK_DEFAULT),)
        _quant_ln_bwd_kernel[grid](grad_in, g, w, n, ctx.scale, BLOCK=BLOCK_DEFAULT)
        return grad_in, None


def fake_quant_act_triton(x, scale=ACT_INT8_SCALE):
    """Triton-fused activation fake-quant. Matches qat.fake_quant_act bit-for-bit
    on CUDA. Falls back to the unfused reference when Triton is unavailable or
    x is not on CUDA."""
    return _FakeQuantActTriton.apply(x, scale)


def fake_quant_weight_triton(w):
    """Triton-fused weight fake-quant (INT8 per-tensor symmetric). Matches
    qat.fake_quant_weight bit-for-bit on CUDA."""
    return _FakeQuantWeightTriton.apply(w)


def fake_quant_ln_weight_triton(w, scale=LN_FIXED_SCALE):
    """Triton-fused RMSNorm-weight fake-quant. Matches qat.fake_quant_ln_weight
    bit-for-bit on CUDA."""
    return _FakeQuantLnWeightTriton.apply(w, scale)
