# Settings tab

## What it is

Dashboard preferences, trainer/plugin configuration, device preference, heartbeat consent, teacher (Ollama) endpoint, mesh role.

## How it works

Markup at [index.html:1266–1642](../../../veritate_mri/web/index.html#L1266). Sectioned panels for: display, runtime, trainers, teachers, mesh, advanced.

- Settings load via `GET /settings`. The whole `mri_settings.json` object is returned and used to hydrate every form field.
- Each form change POSTs the patched key to `/settings`.
- `#sysDetectBtn` triggers `POST /sys/detect` to re-detect hardware (CPU, GPU, RAM) and store the result.
- A build-notices banner reads the build number from `versions.json` (via `/versions`) and shows acknowledgement prompts for new builds.

Settings store at [settings.py](../../../veritate_mri/runtime/settings.py); see [../backend/settings.md](../backend/settings.md).

## Dependencies

- `/settings` GET and POST routes from [settings_routes.py](../../../veritate_mri/routes/settings_routes.py).
- `/sys/detect` from [sys_routes.py](../../../veritate_mri/routes/sys_routes.py).
- `/versions` from [sys_routes.py:115](../../../veritate_mri/routes/sys_routes.py#L115).

## Pitfalls

- Some settings only take effect after a dashboard restart (e.g., `pytorch_load_mode`, `mesh_role`). The UI doesn't yet flag which ones — when in doubt, restart.
- `device_name` is capped at 15 characters (validated server-side at [settings.py:132](../../../veritate_mri/runtime/settings.py#L132)).
- `analytics_advanced_enabled` and `diagnostics_logs_enabled` gate what fields the heartbeat ships; see [../backend/heartbeat.md](../backend/heartbeat.md) for the tier definitions.
