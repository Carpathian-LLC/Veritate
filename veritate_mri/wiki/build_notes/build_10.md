---
build: 10
date: 2026-07-15
title: Settings UI overhaul, auto-install of missing deps, Windows regression fixes, Auto Optimize v2
---

## What changed

### Settings tab, cleaned up
- **Semantic grouping.** Related controls now live together instead of scattered
  across the page:
  - **Engine** section holds *Veritate Engine*, *PyTorch backend*, and *Compute
    device (training override)* as three cards in one row -- the three knobs that
    decide "how does my machine actually run a model".
  - **Analytics** collapsed from three columns to two: *Detect system* stacked
    with *HUD overlay* (the "monitor your system" pair) on the left, telemetry
    opt-ins on the right.
  - **AI assist** and **Teacher Model** now sit side-by-side -- both are
    external-model configs, so they belong together.
- **Heartbeat is a one-liner.** The always-on presence ping shrank from a full
  card to a slim status strip: device name, last send, restarts, send-now
  button. The verbose "what is heartbeat" copy was pushing everything below the
  fold for something you almost never touch.
- **Advanced (bottom of tab).** *MRI telemetry*, *Power save*, *Tutorial*,
  *Extensions*, and *API access* moved into a single collapsible **Advanced**
  section at the bottom of the tab. All five cards share the same neutral
  border/label style -- no more colored left-borders on Power save and Tutorial
  making them look like unfinished mocks.
- **"Training" header removed.** There was only one control under it (Compute
  device override) so the header was pure decoration. The control moved into
  the Engine row.
- **Extensions defaults to off.** The legacy `experimental=true` → `extensions=true`
  auto-migration in `runtime/settings.py` was removed; the toggle now honors
  the `DEFAULTS["extensions"] = False` value on fresh installs and no longer
  silently inherits enablement from the pre-rename setting.

### Missing deps install themselves on server restart
- The launcher and the runtime `deps` module cooperate on first boot and on
  every subsequent restart to bring the interpreter up to spec **without a
  terminal**. If `torch`, `psutil`, `pynvml`, `bitsandbytes`, or the temperature
  helpers are missing, the dashboard installs them via the existing escalation
  ladder (`pip --user` → `pip` → Windows UAC / `sudo -n`) and continues booting.
- **NVIDIA repair path.** If the box has an NVIDIA GPU but got the CPU torch
  wheel by accident (the classic PyPI-fallback mis-install), `install_torch()`
  now re-pins to `download.pytorch.org/whl/cu128` with `--force-reinstall` and
  the dashboard prompts a soft-reload so the running interpreter picks up the
  new wheel.
- **No hangs on `sudo`.** POSIX escalation uses `sudo -n`, which fails fast
  when a password would be required, so the boot thread never blocks on
  interactive input.
- See [deps.md](../../developer_documentation/architecture/backend/deps.md) for
  the full escalation ladder, torch-build detection, and the `status_snapshot`
  contract the Detect Hardware modal consumes.

### Windows regression fixes
- **Engine build.** The Windows C-engine build path recovered from the "MSVC
  couldn't find `<intrin.h>` header" and stale-object regressions -- see commit
  `95fcc60`. The build now checks for and reports missing MSVC / Windows SDK
  toolchain components with actionable messages instead of an opaque `cl.exe`
  exit code.
- **Broken package detection.** `pynvml` and `bitsandbytes` occasionally shipped
  with mismatched CUDA versions on Windows; the deps module now detects the
  mismatch and reinstalls against the running torch's CUDA build.
- **Better cross-platform error surfacing.** Errors that used to die silently in
  the launcher log now bubble to the dashboard's error toast with a short
  actionable message ("torch wheel is CPU but an NVIDIA GPU is present -- click
  Auto Optimize to repair").

### Auto Optimize v2
- **Remembers your settings.** The Auto Optimize modal now persists the
  selected training profile, compute-device override, and PyTorch load mode to
  `mri_settings.json`, so re-opening the modal (or a fresh dashboard boot)
  starts from your last-known-good configuration instead of the analytic
  defaults.
- **Broader hardware coverage.** In addition to the existing training-memory
  and throughput probe, Auto Optimize now benchmarks:
  - **CPU** -- sustained throughput on a synthetic batched matmul, both
    single-threaded and across all logical cores.
  - **GPU** -- the existing forward+backward memory + tokens-per-second probe
    with the CUDA/MPS/CPU device that would be selected for a real run.
  - **RAM** -- fill-and-copy bandwidth benchmark, plus a peak-allocation probe
    that reports the largest contiguous block the OS will give the interpreter.
  - **Disk write speed** -- sequential and random write benchmarks against the
    corpus directory, cached separately per drive. Future builds will use disk
    as a context store for RAG/agent retrieval; the benchmark is being
    collected now so the eventual tiering logic has real numbers to plan
    against.
- Results land in `data/system_specs.json` and are surfaced in the *Detect
  system* card under Analytics.

## What you must do

- Restart the dashboard **once** to load the new settings schema and the
  auto-installer. If you were on the `experimental` toggle previously, verify
  Extensions is off (Settings → Advanced → Extensions) and re-enable if you
  actually want it.
- If Auto Optimize hasn't run since build 9, open it once from the Training
  tab -- the new CPU / GPU / RAM / disk benchmarks populate the specs file so
  the memory planner and future retrieval features have baselines to work
  from.
- Windows users with an NVIDIA GPU: if the dashboard prompts a torch repair on
  boot, accept the UAC prompt. The reinstall replaces a stale CPU wheel with
  the `cu128` build and requires a single soft-reload afterward.

## Versions

| component | version |
|---|---|
| build | 10 |
| engine | v1.3.1 |
| mri | v1.4.0 |
| format | v1.5.0 |
| trainers | v1.1.1 |
