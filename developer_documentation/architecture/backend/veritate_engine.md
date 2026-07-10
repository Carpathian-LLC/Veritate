# veritate_engine (C inference)

## What it is

Compiled C inference engine at [veritate_engine/v1/](../../../veritate_engine/v1/). Loads `.bin` model files (versions v3 through v13) and serves fast byte-level generation. Used by the dashboard when the C backend is selected; co-exists with the PyTorch inference brain.

## How it works

- **Source** at [veritate_engine/v1/src/](../../../veritate_engine/v1/src/). Header [veritate.h](../../../veritate_engine/v1/src/veritate.h) defines the public API.
- **Kernels** at [veritate_engine/v1/kernels/](../../../veritate_engine/v1/kernels/) — INT8, INT4, ternary matmul + transformer ops with SIMD specializations.
- **Dispatch** at [veritate_engine/v1/src/dispatch.c](../../../veritate_engine/v1/src/dispatch.c) — runtime CPU feature detection (AVX2, AVX-512 VNNI on x86; SDOT, I8MM on ARM64). Function pointers selected once at engine load.
- **Bin loader** at [veritate_engine/v1/src/model.c](../../../veritate_engine/v1/src/model.c) — parses the header, dispatches per quantization mode (INT8, INT4, ternary).
- **Binaries** at [veritate_engine/v1/bin/](../../../veritate_engine/v1/bin/) per platform (`macos/arm64`, `macos/x86_64`, `linux/x86_64`, etc.).

## Format versions

| Version | Adds                                                  |
| ------- | ----------------------------------------------------- |
| v3      | Baseline INT8                                         |
| v4      | INT4                                                  |
| v5      | Per-column INT8 scales                                |
| v6      | Mixture-of-Depths gate                                |
| v8      | RMSNorm                                               |
| v9      | `act_boost` residual scale                            |
| v10     | Ternary baseline                                      |
| v11     | QAT mode flag, MoE (top-1 routing only)               |
| v12     | MTP byte-0 transform, RMSNorm scale-64, untied lm_head|
| v13     | hybrid trunk (local attn + gla recurrent global slots), fp32/fp16, own forward path in [src/hybrid.c](../../../veritate_engine/v1/src/hybrid.c) |

Subprocess spawned via `app.config["C_SUBPROCESS"]` on demand. Routes control it via [engine_routes.py](../../../veritate_mri/routes/engine_routes.py). Format versions are declared in [engine_versions.json](../../../veritate_engine/v1/engine_versions.json).

The `v2/` sibling directory is an empty scratchpad reserved for future hot-path-changing experiments; v1 is the sole production engine.

## Dependencies

- [training/export.py](export.md) — produces the `.bin` files this engine consumes.
- [training/build_runner.py](build_runner.md) — orchestrates rebuilds.
- [routes/engine_routes.py](../../../veritate_mri/routes/engine_routes.py) — start/stop, status.

## Pitfalls

- Engine binary lock: while the C subprocess is alive, the binary file is open. The pre-build hook in [app.py:197](../../../veritate_mri/app.py#L197) closes it before rebuilds.
- Bin version compatibility is forward-only (newer engine reads older bins). Loading a newer bin with an older engine fails the magic+version check.
- CPU feature mismatch (e.g., a binary built on AVX-512 host running on a non-VNNI CPU) takes the scalar path silently.
- The dense (v3-v12) kernels are specialized to head_dim 64; `model_load` refuses dense bins with any other head_dim (before the guard they silently corrupted heap through `score_dot_v`'s fixed 64-byte output). v13 is head_dim-generic.
- The chat_traced stdin protocol carries embedded prompt newlines as `0x01` (escaped in [c_engine.py](../../../veritate_mri/inference/backends/c_engine.py), unescaped in `chat_traced_loop`) so chat templates keep their trained framing across the line-based pipe.
- Metal (GPU) path: probe-only today. `metal_dispatch.m` compiles and correctly detects the device (e.g. "Apple M3 Ultra, 222 GB working set"), but the `.metal` shaders need `xcrun metal` (full Xcode, not just Command Line Tools) to compile a `default.metallib`; without it `verify-metal` fails every shape with "library not found." Beyond the tooling gate, the ROI is poor for this engine's workload: decode is single-token O(1) state (m=1 matvecs), where GPU per-dispatch launch+sync overhead (~0.1ms x ~100 matvecs/forward) exceeds CPU NEON compute (~3ms/forward), so Metal would likely make decode SLOWER. GPU only pays off for the batched matmul in long-prompt prefill. The intended "fast on CPU and GPU" split is C engine on CPU (int8 <2 ms/byte) + PyTorch on MPS for the full-MRI path (see inference_brain.md), not Metal-in-the-C-engine. Mechanism to revisit: install Xcode, `verify-metal` to validate the int8 kernel bit-for-bit, then offload prefill matmuls only.
- Stream lock deadlock (fixed): `CTracedSubprocess.stream` holds `self.lock` for a whole generation to serialize the one stateful subprocess. If an SSE client vanishes while its generator is suspended at a `yield` inside the lock, that generator is never closed and the lock is never released, so every later `/generate` (C backend) emits its `meta` frame and then blocks on the lock forever, presenting as an indefinite hang. A stream that cannot take the lock within `STREAM_LOCK_TIMEOUT_S` (30s) now declares the holder dead, kills+respawns the subprocess, bumps `self._epoch`, and swaps in a fresh lock; the abandoned generator's cleanup is epoch-guarded so it cannot drain frames off the fresh proc. Immediate recovery without a fix is a C-backend unload+load (fresh object, fresh lock).
