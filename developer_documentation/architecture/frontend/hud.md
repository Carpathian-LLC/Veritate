# HUD overlay

## What it is

Always-visible system-metrics bar at the top of the dashboard: CPU usage, memory, CPU temperature, GPU temperature, per-GPU activity.

## How it works

Markup at [index.html:35–41](../../../veritate_mri/web/index.html#L35). Each bar has a label, a fill track, a numeric value, and a detail tooltip.

- Fed by periodic `/sys/snapshot` (or equivalent metrics endpoint) — payload shape comes from [sys_metrics.py](../../../veritate_mri/runtime/sys_metrics.py).
- Settings: `hud_enabled`, `hud_position` (`top` or `bottom`), `hud_detailed` (whether to show the detail tooltip).
- `temperature_unit` setting controls the displayed unit (C or F).

Per-GPU bars render into `#hudGpus`; one bar per detected GPU.

## Dependencies

- Backend [sys_metrics.py](../../../veritate_mri/runtime/sys_metrics.py).
- Settings keys: `hud_enabled`, `hud_position`, `hud_detailed`, `temperature_unit`.
- [../backend/sys_metrics.md](../backend/sys_metrics.md).

## Pitfalls

- On older Intel Macs and on Linux without `nvidia-smi`, GPU temps may be missing. The bars hide themselves when their source field is null.
- The HUD polls at a moderate cadence (a few seconds). Don't add tight loops querying the same metrics from other tabs — share the same snapshot.
