# Engine versioning

The C inference engine (`veritate.exe` and historical builds) is versioned
with simple semver. The current build is **v2.0.0**. The previous build,
historically tagged `v3.4.5`, has been renumbered to **v1.0.0** under this
simplified scheme.

## Why renumber

The earlier 3.4.5 number conflated kernel changes, MRI app changes, and
weight-format changes into one tag. v1/v2 here track the engine binary
only, so a kernel-only bump cleanly increments without dragging unrelated
work into the version number.

## Manifest

`veritate_engine/engine_versions.json` is the source of truth. Each entry maps an
engine binary filename under `veritate_engine/bin/<os>/<arch>/` to a semver string
and a human-readable label.

```json
{
  "current": "v2.0.0",
  "engines": [
    {"version": "v2.0.0", "exe": "veritate.exe",        "label": "current", "perf_ms_per_byte": 4.0},
    {"version": "v1.0.0", "exe": "veritate_v1.0.0.exe", "label": "legacy",  "perf_ms_per_byte": 7.0}
  ]
}
```

The MRI server reads this manifest at startup, defaults the C backend to
the highest-version entry whose exe file exists, and exposes the active
version in the top-right meta strip of the UI.

## When to bump

Bump the patch number for kernel tweaks, the minor for new kernel
families, the major for protocol or weight-format changes (anything that
breaks compatibility with older `.bin` files or older trace formats).

## Workflow when shipping a new engine

1. `build.bat` writes `veritate.exe` (the live build).
2. If you want to keep the previous build for A/B comparison, copy the
   old `veritate.exe` out to `veritate_v<old>.exe` BEFORE rebuilding.
3. Bump the `version` of the `veritate.exe` entry in
   `veritate_engine/engine_versions.json` and add a new entry for the archived exe.
4. Restart the MRI server. The dropdown auto-defaults to the new version.

The user-facing UI does not expose engine selection. The engine is
abstracted to "always the newest version with an existing binary." If
you need to run an older engine for benchmarking, pass
`--c-exe <path>` to `run_serve.py`.

## Performance reference

Numbers from the dev box (Ryzen 9800X3D, AVX-512 + VNNI). Per-byte
decode wall time on the 80M model:

| version | ms/byte | notes                                           |
|---------|---------|-------------------------------------------------|
| v1.0.0  | 7.0     | legacy v3.4.5 build                             |
| v2.0.0  | 4.0     | current build                                   |

Treat these as informational anchors; the workbook holds live bench
numbers from `veritate.exe bench 50 200`.
