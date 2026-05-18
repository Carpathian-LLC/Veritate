# HUD metrics

The HUD's live counters come from one HTTP poll.

- **Endpoint:** `GET /sys_metrics` → returns `sys_metrics.snapshot()` from [veritate_mri/runtime/sys_metrics.py](../../veritate_mri/runtime/sys_metrics.py).
- **Frontend:** `index.js` polls `/sys_metrics` on tick; render code near `cpu_temp_c` / `temp_c` references in [veritate_mri/web/index.js](../../veritate_mri/web/index.js).

## Snapshot shape

```
{
  "available": bool,                  // false → psutil missing
  "cpu_pct":         float | null,    // process %, per-core normalized
  "cpu_count":       int,
  "cpu_temp_c":      float | null,    // package temp in Celsius
  "rss_bytes":       int | null,
  "sys_mem_total":   int,             // installed RAM (DIMMs), not OS-reported
  "sys_mem_used":    int,
  "sys_mem_available": int,
  "gpus": [
    {"name": str, "vendor": str, "integrated": bool,
     "load_pct": float|null, "vram_used": int|null,
     "vram_total": int|null, "temp_c": float|null}
  ],
  "ts": float
}
```

## Sources by field × OS

| Field | Linux | macOS | Windows |
|---|---|---|---|
| `cpu_pct`, `rss`, mem | psutil | psutil | psutil |
| `cpu_temp_c` | psutil `sensors_temperatures` (coretemp / k10temp / cpu_thermal) | Apple Silicon: `macmon pipe -s 1` (JSON `temp.cpu_temp_avg`). Intel: `osx-cpu-temp -c`. Homebrew. | LibreHardwareMonitor WMI |
| GPU list | `/sys/class/drm` | `system_profiler SPDisplaysDataType -json` | `Get-CimInstance Win32_VideoController` |
| Apple SoC GPU load | n/a | `ioreg -c IOAccelerator` → `Device Utilization %` | n/a |
| Apple SoC GPU temp | n/a | `macmon pipe -s 1` (JSON `temp.gpu_temp_avg`) | n/a |
| NVIDIA load / VRAM / temp | `nvidia-smi` | `nvidia-smi` | `nvidia-smi` |
| Non-NVIDIA GPU temp | none | Apple Silicon: `macmon`. Intel: none. | LibreHardwareMonitor WMI |

## macOS CPU temperature

Apple exposes **no** first-party non-sudo API for CPU temperature. `psutil.sensors_temperatures()` returns `{}` on every macOS build (Intel and Apple Silicon).

The metrics module probes two third-party CLIs in order:

1. **`macmon`** (Apple Silicon, sudoless) — `macmon pipe -s 1 -i 200` emits one newline-delimited JSON sample with `temp.cpu_temp_avg` and `temp.gpu_temp_avg`. Install: `brew install macmon`. Single call populates both CPU and GPU; result is cached for `_LIVE_TTL`.
2. **`osx-cpu-temp -c`** (Intel only, SMC) — CPU only. Install: `brew install osx-cpu-temp`.

Neither installed → `cpu_temp_c` is `null`, the integrated GPU's `temp_c` is `null`, and the HUD shows a settings-tab notice with the install commands. `sudo powermetrics` would also produce both numbers but requires sudo per call, which is unworkable for a double-click launcher.

NVIDIA cards on Mac (rare; historic only) report their own temperature via `nvidia-smi` regardless of `macmon`.

## Caches and rates

| Probe | TTL | Notes |
|---|---|---|
| CPU temp (`_cpu_temp`) | 1s | `_LIVE_TTL` |
| Apple GPU load (`_mac_apple_gpu_load`) | 1s | `_LIVE_TTL` |
| LibreHardwareMonitor (`_lhm_sensors`) | 1s | `_LIVE_TTL` |
| Adapter list (`_adapters`) | warmed once on startup | rescanned only at `warm()` |
| Installed RAM bytes | warmed once on startup | OS-specific path |

`snapshot()` is cheap to call after `warm()`; the heavy probes (`system_profiler`, `nvidia-smi`) only run via cached paths or on demand.

## Extending

To add a new field:
1. Compute it in a small helper (`_foo()`) inside `sys_metrics.py` with platform guards.
2. Surface it from `snapshot()` (and from the `available: false` branch as `null`).
3. Read it in the HUD render in `index.js`. Treat `null` as "no data" — never `NaN`, never `0`.

Keep the field name the same across all OSes; the OS-difference belongs inside the helper, not in the consumer.
