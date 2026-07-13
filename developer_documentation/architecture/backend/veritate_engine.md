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
- Line-capacity rule (line-based protocol): every serving loop reads one protocol line per `fgets`, so a line longer than the read buffer would leave residue in stdin and permanently desync the persistent subprocess (the next request reads the residue/header as its prompt). Two guards: (1) `c_engine.py` tail-clamps the prompt payload to `min(seq+2, seq - min(max_new, seq//2))` bytes — `fgets(prompt_line, seq+4)` stores at most `seq+3` chars incl the `'\n'`, so `seq+2` payload + `'\n'` fits with no residue, and since the generation budget is `seq - prompt_len` with no window slide, the clamp also reserves room (capped at `seq//2`) so a near-window prompt still gets its `max_new` reply; clamping the tail keeps the newest context. (2) `fgets_drain` in `main.c` wraps every serving-loop `fgets` (header + prompt in `chat_traced`, and the `chat`/`chat_greedy`/`chat_spec` line reads) and discards any bytes past the buffer up to the next `'\n'`, so even an unclamped over-long line cannot shift the protocol.
- Fast serving (non-traced): the `chat_traced` stdin header takes an optional trailing `trace` flag (default 1, so a legacy 9-field header is unchanged). With `trace=0` the loop passes `NULL` trace to `forward`/`forward_decode` (skips the per-layer logit-lens matvec and all recording), skips the DLA scans, attention quantization, decisiveness/confidence, and candidate DLA, and emits a 16-byte `FFRM` frame (marker + pos + real_len + byte + argmax, no payload) instead of the full `TFRM` frame. Sampling is the identical `sample_token_ext` call, so output bytes match the traced path for the same request (verified greedy byte-for-byte vs both the traced path and the pre-change binary). The chat serves `trace=0` (coarse `fast_byte` events); the `/generate` MRI view serves `trace=1` (full frames). Measured 1.26x per-byte on M3 Ultra (chat80m: 3.00 -> 2.38 ms/byte); the gap widens on bandwidth-bound CPUs because the traced frame streams ~300 KB/byte vs 16 B.
- Metal (GPU) path: probe-only today. `metal_dispatch.m` compiles and correctly detects the device (e.g. "Apple M3 Ultra, 222 GB working set"), but the `.metal` shaders need `xcrun metal` (full Xcode, not just Command Line Tools) to compile a `default.metallib`; without it `verify-metal` fails every shape with "library not found." Beyond the tooling gate, the ROI is poor for this engine's workload: decode is single-token O(1) state (m=1 matvecs), where GPU per-dispatch launch+sync overhead (~0.1ms x ~100 matvecs/forward) exceeds CPU NEON compute (~3ms/forward), so Metal would likely make decode SLOWER. GPU only pays off for the batched matmul in long-prompt prefill. The intended "fast on CPU and GPU" split is C engine on CPU (int8 <2 ms/byte) + PyTorch on MPS for the full-MRI path (see inference_brain.md), not Metal-in-the-C-engine. Mechanism to revisit: install Xcode, `verify-metal` to validate the int8 kernel bit-for-bit, then offload prefill matmuls only.
- Stream lock deadlock (fixed): `CTracedSubprocess.stream` holds `self.lock` for a whole generation to serialize the one stateful subprocess. If an SSE client vanishes while its generator is suspended at a `yield` inside the lock, that generator is never closed and the lock is never released, so every later `/generate` (C backend) emits its `meta` frame and then blocks on the lock forever, presenting as an indefinite hang. A stream that cannot take the lock within `STREAM_LOCK_TIMEOUT_S` (30s) now declares the holder dead, kills+respawns the subprocess, bumps `self._epoch`, and swaps in a fresh lock; the abandoned generator's cleanup is epoch-guarded so it cannot drain frames off the fresh proc. Immediate recovery without a fix is a C-backend unload+load (fresh object, fresh lock).
