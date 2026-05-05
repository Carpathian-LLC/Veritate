---
title: kernels architecture
date: 2026-05-05
tags: [kernels, architecture, int8, dispatch]
summary: How Veritate is built and why. Layering, INT8 rationale, runtime dispatch, matmul kernel, ablation contract.
---

> Source: `documentation/kernels/architecture.md` (mirrored copy; the file at that path remains the canonical contract).

# Architecture

How Veritate is built, and why.

# ------------------------------------------------------------------------------------
# The big picture
# ------------------------------------------------------------------------------------

Veritate is three layers:

```
┌─────────────────────────────────────────────────────────────┐
│  main.c — entry point, timing, CLI                          │
├─────────────────────────────────────────────────────────────┤
│  dispatch.c — detect CPU, fill function-pointer table       │
│  tensor.c   — alloc, init, quantize INT8 tensors            │
│  model.c    — forward pass: composes kernels into a layer   │
├─────────────────────────────────────────────────────────────┤
│  KERNELS (one per backend, picked at startup):              │
│    scalar/matmul.c          plain C, runs anywhere          │
│    x86_64/matmul_avx2.c     AVX2 intrinsics, INT8 dot       │
│    x86_64/matmul_vnni.c     AVX-512 + VPDPBUSD              │
│    arm64/matmul_neon_sdot.c NEON + SDOT, Apple Silicon path │
│    arm64/matmul_neon.c      NEON without SDOT (Pi 4 class)  │
└─────────────────────────────────────────────────────────────┘
```

The high-level layer never knows which kernel ran. The dispatch table is filled at
startup based on `cpuid` results, and every call site invokes through a function pointer.

# ------------------------------------------------------------------------------------
# Why INT8
# ------------------------------------------------------------------------------------

Modern weights compress to INT8 with negligible accuracy loss for inference. INT8 buys us:

- **4× memory bandwidth** vs float32. The bottleneck for inference is RAM, not compute.
- **4× cache footprint** improvement. A 7B model in INT8 fits in 7 GB; in fp32 it's 28 GB.
- **Native SIMD throughput**: VPDPBUSD does 64 INT8 multiply-accumulates per instruction.
- **Analog compatibility**: real analog hardware (Mythic, IBM) operates at ~8-bit precision
  via Ohm's law on flash cells. Code written in INT8 ports natively.

Block quantization (Q8_0): for every 32 INT8 values, we store one fp16 scale. This is the
exact format used by GGUF / llama.cpp.

# ------------------------------------------------------------------------------------
# Why runtime dispatch
# ------------------------------------------------------------------------------------

Compile once, run on any x86 CPU from a 2013 Haswell to a 2025 Zen 5. The runtime detects:

- `sse4_2` → must-have baseline (2008+)
- `avx2`   → 256-bit vectors (2013+)
- `avx512f` → 512-bit vectors (2017+, server / workstation)
- `avx512_vnni` → INT8 dot product instruction (2019+, Ice Lake / Zen 4+)

It picks the highest tier available and patches function pointers at startup. Cost at call
site: one indirect jump, predicted by the branch predictor after the first call. Effectively
zero overhead.

This is the same pattern used by FFmpeg, OpenBLAS, and llama.cpp.

# ------------------------------------------------------------------------------------
# Memory layout
# ------------------------------------------------------------------------------------

Weights are stored on disk as a flat binary blob, `mmap()`ed at startup. Layout:

```
[ header (16 bytes) ]
[ weight block 0: int8[N], scale[N/32] ]
[ weight block 1: int8[M], scale[M/32] ]
...
```

`mmap` means zero-copy load. The OS pages weights in on demand. Cache-friendly because
the access pattern in matmul is linear within a block.

# ------------------------------------------------------------------------------------
# The matmul kernel — the heart of Veritate
# ------------------------------------------------------------------------------------

99% of inference time is matrix multiplication. Everything else (softmax, layernorm, GELU)
is rounding error. Veritate spends its complexity budget on one kernel and keeps the rest
trivial.

### Scalar reference (always works)

```c
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += a[i*K + k] * b[k*N + j];
        c[i*N + j] = acc * scale;
    }
```

3 nested loops, INT8 inputs, INT32 accumulator (so we don't overflow on long dot products).

### AVX2 (current target)

For each output element, we want the dot product of two INT8 vectors of length K. AVX2:

1. `_mm256_loadu_si256` — load 32 INT8 values from each input
2. `_mm256_maddubs_epi16` — multiply pairs into INT16, sum adjacent pairs
3. `_mm256_madd_epi16` — sum INT16 pairs into INT32
4. `_mm256_add_epi32` — accumulate into running sum
5. After K iterations: horizontal-sum the INT32 lanes into one int.

That's 32 multiply-accumulates per ~3 instructions. Plus aggressive loop unrolling.

### AVX-512 VNNI (v2, where the magic is)

`VPDPBUSD` does steps 2+3+4 in **one instruction**. 64 INT8 muls per cycle. On the 9800X3D,
two FMA units per core × 8 cores ≈ ~8 TOPS aggregate INT8 throughput.

# ------------------------------------------------------------------------------------
# Sub-millisecond — the bar
# ------------------------------------------------------------------------------------

A 1024×1024×1024 INT8 matmul = 2 GOps. At 1 TOPS effective (single core, AVX2) that's 2 ms.
At 4 TOPS (multi-core AVX-512 VNNI) it's ~0.5 ms. Sub-millisecond is achievable but not free.

Every commit must benchmark and log the result in `docs/WORKBOOK.md`. Regressions revert.

# ------------------------------------------------------------------------------------
# What we're NOT doing
# ------------------------------------------------------------------------------------

- No graph executor. Forward pass is hand-coded.
- No autograd. Inference only.
- No plugin system. Kernels are statically dispatched.
- No format flexibility. INT8 only. (INT4 in v4.)
- No GPU. Veritate is CPU-first; GPU would be a sister project.
- No cross-language runtime. C and assembly only.

# ------------------------------------------------------------------------------------
# Streaming prefill — making latency disappear
# ------------------------------------------------------------------------------------

Sub-millisecond matmul is half the story. The other half is hiding compute behind the
user's input phase. From the moment a user starts typing to the moment they hit Enter
is dead time on the CPU — Veritate uses it.

Mechanism (v3+):
- On every keystroke (debounced to word boundaries), forward-pass the partial input.
- Persist the resulting KV cache, keyed by the prefix.
- On Enter, only the final token positions need processing — everything else is reused.
- Effective user-perceived latency = typing speed, not model speed.

Edge case: user edits or deletes. KV cache is structured as a tree, not a chain. On
divergence, rewind to the last common prefix and re-prefill from there.

This is not theoretical — vLLM does it server-side, llama.cpp has primitive prompt
caching, and Apple's iOS predictive text is essentially this idea. Veritate makes it
the default UX.

Implementation deferred to v3 when the autoregressive forward pass exists.

# ------------------------------------------------------------------------------------
# Causal ablation (v8)
# ------------------------------------------------------------------------------------

Ablation is applied in `model.c::forward_decode` on the post-GELU
`ffn_neurons` buffer immediately before the `ffn_down` matmul. When the model
is configured with `(ablate_layer, ablate_neuron) = (L, N)`, the engine zeros
`ffn_neurons[L][pos][N]` for the current position before the down-projection
runs. This is a one-line operation against the int8 buffer; no kernel
signatures change. Every kernel tier (scalar, AVX-512 VNNI, future ARM64
NEON) sees identical inputs, so the rule 23 bitwise-parity contract holds
without modification.

Contract:

- Ablation is a forward-pass parameter, not a model field. It does not
  appear in `veritate.bin`. It does not change `VERITATE_MODEL_VERSION`.
- Ablation is requested per generation via `/generate?ablate_layer=L&ablate_neuron=N`
  (engine + pytorch). The engine echoes the active `(L, N)` into the TFRM v8
  frame's `ablation_layer` / `ablation_neuron` fields for UI labeling.
- Pytorch parity: `Brain.stream` exposes a forward hook that zeros the same
  `ffn_neurons[L][N]` pre-`ffn_down`, ensuring the same output deltas as the
  engine path.
- Default: `(ablate_layer, ablate_neuron) = (-1, -1)`. No-op; zero runtime
  cost.

# ------------------------------------------------------------------------------------
# What "ASIC-like" means here
# ------------------------------------------------------------------------------------

A real ASIC bakes the model architecture into silicon. Etched Sohu only runs transformers
because the transformer dataflow is wired into the chip. Veritate does the software analogue:
the model topology is compile-time. Layer count, hidden dim, head count — all `#define`s.
The compiler unrolls accordingly. The binary is the model.

The trade-off: changing the architecture requires recompiling. The win: zero runtime
overhead, no graph traversal, no shape checks.
