# veritate_engine (C inference)

## What it is

Compiled C inference engine at [veritate_engine/v1/](../../../veritate_engine/v1/). Loads `.bin` model files (versions v3 through v12) and serves fast byte-level generation. Used by the dashboard when the C backend is selected; co-exists with the PyTorch inference brain.

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
