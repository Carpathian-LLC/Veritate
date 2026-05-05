---
title: Skipping the dead half of the FFN
date: 2026-05-05
tags: [kernel, ffn, sparsity, gelu, avx-512]
summary: Post-GELU activations are mostly near zero. A sparse-aware ffn_down kernel skips the zeros, gives 1.24x decode speedup, and (surprise) improves perplexity by 2.6%.
---

GELU squashes negative inputs toward zero and passes positive inputs through almost unchanged. After a trained FFN's `up` projection, a large fraction of post-GELU activations are at or very near zero. Multiplying those rows of the activation against the `down` weight matrix is wasted work: the contribution is zero (or near-zero) by construction.

The sparse path scans the post-GELU activation, zeroes anything below a small threshold, and dispatches to a sparse matmul that touches only the surviving columns. It falls through to the dense path when not enough rows are sparse for the scan-plus-dispatch overhead to be worth it.

## the dispatcher

```
if (n_nonzero * 2 < V_FFN) -> sparse
else                      -> dense (matmul_int8_vnni_prep)
```

The factor of 2 covers the cost of the pre-scan plus the worse memory access pattern of the sparse path. Below half the FFN width nonzero, sparse wins; above, dense wins. The dispatcher decides per token, every token.

Bit-identical int32 output to the dense path by construction: zeros contribute zero whether the kernel multiplies them or skips them.

## the threshold finding

The interesting result is not the speedup. It is what happened to perplexity when we raised the threshold from 0 to 4.

| threshold | decode p50 | sparse fires | ppl |
|---|---|---|---|
| 0 | 0.953 ms | 33% | 19.350 |
| 4 (default) | 0.769 ms | 100% | **18.842** (-2.6%) |
| 6 | - | - | 60.3 (cliff) |
| 8 | 0.547 ms | - | 61.5 |

Quality **improved** when we hard-zeroed low-magnitude activations. The interpretation: this is a QAT-quantized model, and quantization adds drift to the residual stream over depth. Low-magnitude post-GELU values are mostly that drift, not real signal. Clamping them to zero is a free denoiser.

Above threshold 6, real signal starts getting clipped and perplexity falls off a cliff. So 4 is the sweet spot for this checkpoint and ships as the default.

## numbers

- Validated on 51,000 byte tokens of TinyStories val.
- 1.24x decode speedup (0.953 -> 0.769 ms p50) on the 9800X3D, QAT 80M model.
- 2.6% perplexity reduction.
- Compile-time flag `VERITATE_GELU_ZERO_THRESH` controls the threshold; default is 4 in `build.bat`.
- Lives in `engine/kernels/x86_64/transformer_avx512.c::matmul_int8_sparse_decode` and `ffn_down_decode`.

## caveat

The threshold is checkpoint-specific. A different model trained with a different quantization recipe will have a different drift profile. Re-validate before changing the default.

## takeaway

Sparsity in the post-GELU path is real and exploitable. The denoising side effect was a surprise: it suggests there is room to fold the threshold into the training loop rather than the kernel, so the model learns to put real signal above the threshold and noise below.
