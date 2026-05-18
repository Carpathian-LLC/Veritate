---
title: kernels architecture
date: 2026-05-05
tags: [kernels, architecture, int8, dispatch]
summary: How Veritate is built. Three layers, INT8 weights, runtime dispatch, one matmul.
---

> Friendly summary. The canonical contract is `documentation/kernels/architecture.md`.

## three layers

| layer | files | job |
|---|---|---|
| top | `main.c`, `model.c` | entry point, forward pass, composes kernels into layers |
| middle | `dispatch.c`, `tensor.c` | detect CPU at startup, fill function-pointer table, allocate INT8 tensors |
| bottom | `kernels/<arch>/` | hand-tuned matmul + helpers per CPU |

The top layer never knows which kernel ran. Calls go through function pointers filled at startup based on `cpuid`.

## why INT8

INT8 weights cost negligible accuracy for inference and unlock four wins.

- 4× the memory bandwidth of float32. Inference is RAM-bound.
- 4× the cache footprint. A 7B model fits in 7 GB instead of 28 GB.
- Native SIMD throughput. AVX-512 VNNI does 64 INT8 multiply-accumulates per instruction.
- Analog hardware (Mythic, IBM) runs at ~8 bits via Ohm's law on flash. INT8 code ports natively.

Block quantization Q8_0 (one fp16 scale per 32 INT8 values), same format as GGUF / llama.cpp.

## why runtime dispatch

Compile once, run on any x86 from a 2013 Haswell to a 2025 Zen 5. The runtime probes for SSE 4.2, AVX2, AVX-512, and VNNI, then patches the kernel pointers to the highest tier the CPU supports. Cost per call: one indirect jump that the branch predictor learns after the first hit. Same pattern as FFmpeg, OpenBLAS, and llama.cpp.

## memory layout

Weights are a flat blob `mmap()`ed at startup. Zero-copy load, demand-paged by the OS. The matmul access pattern is linear within a block, so the caches stay happy.

```
[ header (16 bytes) ]
[ weight block 0: int8[N], scale[N/32] ]
[ weight block 1: int8[M], scale[M/32] ]
...
```

## the matmul

99% of inference time. Everything else is rounding error. Veritate spends its complexity budget on this one kernel.

| kernel | one MAC takes | where |
|---|---|---|
| scalar reference | 3 nested loops, INT8 in, INT32 accumulator | `kernels/scalar/matmul.c` |
| AVX2 | 32 MACs per ~3 instructions via `maddubs` + `madd` | `kernels/x86_64/matmul_avx2.c` |
| AVX-512 + VNNI | 64 MACs per single `vpdpbusd` | `kernels/x86_64/matmul_vnni.c` |
| NEON SDOT | 16 MACs per `sdot` | `kernels/arm64/matmul_neon_sdot.c` |
| NEON only | scalar fallback for pre-SDOT ARM | `kernels/arm64/matmul_neon.c` |

Sub-millisecond matmul on the 9800X3D. A 1024×1024×1024 INT8 matmul is 2 GOps; at ~4 effective TOPS that's ~0.5 ms. Achievable, not free. Every commit benches and logs to the workbook; regressions revert.

## what we are not doing

No graph executor. No autograd. No plugin kernel system. No format flexibility (INT8 today, INT4 in v4). No GPU. No cross-language runtime. C and assembly only. The architecture is compile-time — layer count, hidden dim, head count are `#define`s. The binary is the model.

## streaming prefill (v3)

Sub-ms matmul is half the latency story. The other half is hiding compute behind the user's typing.

- On every keystroke (debounced to word boundaries), forward-pass the partial input.
- Persist the resulting KV cache, keyed by the prefix.
- On Enter, only the final token positions need processing.
- Effective user-perceived latency = typing speed, not model speed.

Edits or deletes? KV cache is a tree. Rewind to the last common prefix and re-prefill from there. Same idea as vLLM, llama.cpp prompt caching, and iOS predictive text. Defaults on once the autoregressive forward lands in v3.

## causal ablation (v8)

Ablation is a forward-pass parameter, not a model field. The engine zeros `ffn_neurons[L][pos][N]` immediately before `ffn_down` when the request carries `(ablate_layer, ablate_neuron)`. One line of code, no kernel signatures change, every kernel tier sees identical inputs so bitwise parity holds. PyTorch mirrors with a forward hook. Default `(-1, -1)` is a no-op.

Per generation: `GET /generate?ablate_layer=L&ablate_neuron=N`. The engine echoes the active `(L, N)` into the TFRM frame for UI labeling.

## the ASIC analogy

Etched Sohu only runs transformers because the dataflow is wired into the silicon. Veritate is the software analogue. The model topology is compile-time. Compiler unrolls accordingly. Trade-off: changing the architecture requires recompiling. Win: zero runtime overhead, no graph traversal, no shape checks.
