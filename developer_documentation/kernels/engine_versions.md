# Engine versioning

## what it is

The C inference engine is versioned with simple semver, tracking the engine binary only so a kernel-only change bumps cleanly without dragging MRI app or weight-format work into the number. The current build is **v2.0.0**.

## manifest

`veritate_engine/v1/engine_versions.json` is the source of truth. `current` names the active semver; each entry in `engines` describes one binary.

| field | meaning |
|---|---|
| `version` | semver string |
| `exe` | binary filename under `veritate_engine/v1/bin/<os>/<arch>/` |
| `path` | absolute path to that binary; `/c-engines` resolves and filters on it |
| `label` | short human label shown in the UI |
| `perf_ms_per_byte` | informational decode anchor |
| `notes` | kernels compiled in, trace protocol version, `.bin` format versions loadable |

Binary name is per-OS (`readers/paths.py::BINARY_NAME_BY_OS`): `veritate.exe` on Windows, `veritate` on Linux and macOS. Only `v1` is built (`ENGINE_PRIMARY`).

## consumers

- `veritate_mri/readers/engine.py`: `manifest()`, `engines()`, `by_path(abs_path)`. Returns an empty registry rather than raising when the file is missing or malformed.
- `/c-engines` (`routes/engine_routes.py`): lists manifest entries whose `path` is an existing file, annotating `is_current`, `mtime`, `size`.
- `/c-config` (POST `{exe, model}`): closes the running subprocess and respawns `CTracedSubprocess` against the selected binary. This is how a specific engine build is selected at runtime.
- `veritate_mri/tools/perf_trace.py`: defaults `--exe` to the manifest's current entry.

## when to bump

Patch for kernel tweaks, minor for a new kernel family, major for protocol or weight-format changes (anything that breaks compatibility with older `.bin` files or older trace formats).

## shipping a new engine

1. `build.sh` (macOS/Linux) or `build.bat` (Windows) writes the live binary.
2. To keep the previous build for A/B, copy it aside under a versioned filename before rebuilding.
3. Bump the live entry's `version` in `engine_versions.json` and add an entry for the archived binary.
4. Restart the MRI server; the engine defaults to the highest-version entry whose binary exists.

## pitfalls

- An entry with no `path` key never appears in `/c-engines`, because the route filters on `os.path.isfile(entry["path"])`. `exe` alone is not enough.
- `perf_ms_per_byte` values in the manifest are anchors, not measurements of the current box. Live numbers come from `veritate bench 50 200`.
