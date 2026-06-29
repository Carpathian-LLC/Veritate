# qat triton kernels

Fused Triton kernels for QAT fake-quant ops. Used at training time only;
deployment uses the C engine's INT8 kernels.

## what

Three fake-quant ops have Triton-fused forward + backward kernels:

| op                       | unfused path                                                  | fused path                          |
|--------------------------|----------------------------------------------------------------|--------------------------------------|
| `fake_quant_act(x)`      | mul + round + clamp + div (4 launches, autograd graph)         | one Triton launch                   |
| `fake_quant_weight(w)`   | amax + div + round + clamp + mul (5 launches, autograd graph)  | amax + one Triton launch            |
| `fake_quant_ln_weight(w)`| mul + round + clamp + div (4 launches, autograd graph)         | one Triton launch                   |

INT4 and ternary weight modes stay on the unfused path. Their ablations are not
throughput-bound.

## numerics

All math runs in fp32 inside the kernel and inside the unfused reference.

The clamp mask in backward is computed on the post-round integer level, not the
pre-round float. This matches PyTorch's `clamp` autograd which sees the rounded
tensor as its input.

## activation

Triton path is selected when:
- `triton` import succeeds (true on Windows with `triton-windows` 3.7+, Linux
  with stock triton, mac falls back to unfused).
- input tensor is CUDA.
- env var `VERITATE_NO_TRITON` is unset / "0" / "false".

Otherwise the unfused reference runs. Both paths are mathematically equivalent.

## perf

Measured on RTX 5070 (Blackwell, sm_120), bf16, B=32 T=1024 C=768:

| op pattern                                | unfused | triton  | speedup |
|-------------------------------------------|---------|---------|---------|
| `linear(fake_quant_act(x), fake_quant_w(w))` fwd     | 1.37 ms | 0.80 ms | 1.7x    |
| same + `.sum().backward()`                | 3.65 ms | 2.41 ms | 1.5x    |

End-to-end training-step speedup is bounded by the Python autograd overhead
that remains around each fused op. A larger win requires a single Triton kernel
that fuses `quant_act + quant_w + matmul + dequant` (and the symmetric backward);
that work is tracked under "fused INT8 linear" in ideas.

## files

| file                                  | role                                |
|---------------------------------------|--------------------------------------|
| `veritate_core/qat.py`                | reference + Triton dispatch          |
| `veritate_core/qat_triton.py`         | Triton kernels + autograd Functions  |
