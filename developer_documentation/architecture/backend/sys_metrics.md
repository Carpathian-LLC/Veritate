# sys_metrics

## What it is

Hardware detection and runtime metrics at [veritate_mri/runtime/sys_metrics.py](../../../veritate_mri/runtime/sys_metrics.py). Detects CPU, GPU, RAM at startup; serves periodic snapshots for the HUD and heartbeat.

## How it works

`detect_specs()` enumerates CPU model + cores, RAM total, every GPU (vendor, name, VRAM, integrated bit), and OS version. Cached in memory and written to `data/system_specs.json` on `POST /sys/detect`.

`snapshot()` returns a lightweight snapshot for the HUD: per-component usage and temperatures. Sampled at a moderate cadence; cached briefly.

Per-platform detail:

- macOS — `ioreg` + `sysctl` for hardware, `powermetrics` if available for GPU.
- Linux — CPU/RAM from `/proc`; GPU adapters enumerated from `/sys/class/drm/card*` (vendor id, AMD `gpu_busy_percent` load). Model names come from `lspci -D` matched to each card by PCI slot (`Intel UHD Graphics 630`, not `card1`); NVIDIA load/VRAM/temp overlaid from `nvidia-smi`.
- Windows — WMI queries via subprocess.

`warm()` is called at dashboard startup ([app.py:203](../../../veritate_mri/app.py#L203)) unless minimal mode, pre-populating the cache so the first HUD render is instant.

## Dependencies

- [routes/sys_routes.py](../../../veritate_mri/routes/sys_routes.py) — exposes `/sys/specs`, `/sys/detect`, `/sys/snapshot`.
- [runtime/heartbeat.py](../../../veritate_mri/runtime/heartbeat.py) — calls `snapshot()` for the analytics-tier hw block.
- Frontend [hud.md](../frontend/hud.md) — primary consumer of `snapshot()`.

## Pitfalls

- Per-platform detection can fail on unusual configurations (custom kernels, missing tools). Failures fall back to `available=false` rather than raising.
- GPU temperatures are best-effort; absent fields render as "—" in the HUD rather than 0.
- GPU utilization is per-vendor: AMD (`gpu_busy_percent`) and NVIDIA (`nvidia-smi`) report `load_pct`; Intel integrated GPUs on Linux and non-NVIDIA adapters on Windows have no cheap counter, so `load_pct` stays `null` by design. The HUD disables the load meter for those adapters instead of showing a dead 0% bar. Each metric (load, VRAM, temp) is independent: an adapter reports whichever it can read.
- `warm()` blocks for a second or two on cold cache. Skipped in `VERITATE_MINIMAL=1` mode to keep startup fast.
