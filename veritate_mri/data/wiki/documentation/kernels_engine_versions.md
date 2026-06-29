---
title: engine versioning
date: 2026-05-05
tags: [engine, versions, semver, manifest]
summary: How the C engine binary is versioned and how the dashboard picks which one to use.
---

> Friendly summary. The canonical contract is `developer_documentation/kernels/engine_versions.md`.

The C engine (`veritate.exe`) uses simple semver. The current build is **v2.0.0**. The earlier build, historically tagged v3.4.5, has been renumbered to **v1.0.0** under this scheme.

## why renumber

The earlier 3.4.5 number conflated kernel changes, dashboard changes, and weight-format changes into one tag. v1 / v2 here track only the engine binary, so a kernel-only bump increments cleanly without dragging unrelated work along.

## the manifest

`veritate_engine/v1/engine_versions.json` is the source of truth. It maps an exe filename in `$LOCALAPPDATA/veritate/` to a semver string and a label.

```json
{
  "current": "v2.0.0",
  "engines": [
    {"version": "v2.0.0", "exe": "veritate.exe",        "label": "current", "perf_ms_per_byte": 4.0},
    {"version": "v1.0.0", "exe": "veritate_v1.0.0.exe", "label": "legacy",  "perf_ms_per_byte": 7.0}
  ]
}
```

The MRI server reads this at startup, picks the highest-version entry whose exe exists, and shows the active version in the top-right meta strip of the dashboard.

## when to bump

| change | bump |
|---|---|
| kernel tweak | patch |
| new kernel family | minor |
| protocol or weight-format break | major |

## shipping a new engine

1. `build.bat` writes `veritate.exe` (the live build).
2. To keep the previous build for A/B comparison, copy it to `veritate_v<old>.exe` **before** rebuilding.
3. Bump the `version` of the `veritate.exe` entry in `veritate_engine/v1/engine_versions.json` and add a new entry for the archived exe.
4. Restart the MRI server. The dropdown auto-defaults to the new version.

The user-facing UI doesn't expose engine selection. The engine is abstracted to "always the newest version with an existing binary." To run an older engine for benchmarking, pass `--c-exe <path>` to `run_serve.py`.

## reference perf

Numbers from the dev box (Ryzen 9800X3D, AVX-512 + VNNI). Per-byte decode wall time on the 80M model.

| version | ms/byte | notes |
|---|---|---|
| v1.0.0 | 7.0 | legacy v3.4.5 build |
| v2.0.0 | 4.0 | current build |

Treat these as informational anchors. The workbook holds live bench numbers from `veritate.exe bench 50 200`.
