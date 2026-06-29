# build_runner

## What it is

Orchestrates rebuilds of the C engine at [veritate_mri/training/build_runner.py](../../../veritate_mri/training/build_runner.py).

## How it works

`set_pre_build_hook(fn)` registers a function called before each build. [app.py:197](../../../veritate_mri/app.py#L197) wires this to `_close_c_for_rebuild`, which closes the C engine subprocess so the build doesn't fail with a binary-lock error.

Builds invoke the platform-specific build script under [veritate_engine/v1/build/](../../../veritate_engine/v1/build/). Output binaries land in [veritate_engine/v1/bin/](../../../veritate_engine/v1/bin/) per platform (`macos/arm64`, `macos/x86_64`, `linux/x86_64`, etc.).

## Dependencies

- [engine_routes.py](../../../veritate_mri/routes/engine_routes.py) — exposes build trigger to the dashboard.
- `app.config["C_SUBPROCESS"]` — held while the engine is running; the pre-build hook clears it.

## Pitfalls

- Concurrent builds for the same platform are not protected. The dashboard UI disables the build button while one is in flight; programmatic callers should respect a similar gate.
- The build relies on a working C toolchain (clang on macOS, gcc on Linux). Missing toolchain shows as a generic exit code in the log ring — check the Logs tab.
