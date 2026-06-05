# settings

## What it is

Persistent dashboard settings at [veritate_mri/runtime/settings.py](../../../veritate_mri/runtime/settings.py), stored at `data/mri_settings.json` (gitignored, machine-local).

## How it works

`DEFAULTS` at [settings.py:30](../../../veritate_mri/runtime/settings.py#L30) is the single source of truth. On first run, the file is created from `DEFAULTS`. On every load, missing keys are filled from `DEFAULTS` and written back; user values are preserved.

`get()` returns the current settings as a dict. `update(patch)` applies a patch with validation.

Validation lives inline: e.g., `device_name` capped at 15 characters at [settings.py:132](../../../veritate_mri/runtime/settings.py#L132).

## Key settings

| Key                              | Purpose                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `pytorch_load_mode`              | `always` / `on_demand` / `off` — when to load the brain        |
| `pytorch_idle_unload_secs`       | Idle watcher timeout                                           |
| `heartbeat_enabled`              | Master switch for the Carpathian webhook                       |
| `heartbeat_send_errors`          | Include error detail in presence pings                         |
| `analytics_advanced_enabled`     | Include full training payload (not just `training_active`)     |
| `diagnostics_logs_enabled`       | Send the diagnostics payload alongside presence                |
| `device_preference`              | `auto` / `cpu` / `mps` / `cuda` for trainers                   |
| `device_name`                    | Display name (max 15 chars) shown on Carpathian dashboard      |
| `update_channel`                 | `development` / `stable` for self-update                       |
| `mesh_role`                      | `off` / `hub` / `node` / `both`                                |
| `teacher_provider`, `teacher_*`  | Ollama / API teacher endpoint                                  |
| `last_acknowledged_build`        | Build notices banner cutoff                                    |

## Dependencies

- [settings_routes.py](../../../veritate_mri/routes/settings_routes.py) — GET/POST `/settings` for the dashboard.
- [runtime/heartbeat.py](../../../veritate_mri/runtime/heartbeat.py) — reads consent flags.
- [training/trainer_runner.py](../../../veritate_mri/training/trainer_runner.py) — reads `device_preference`.

## Pitfalls

- Some settings only take effect after a dashboard restart (`pytorch_load_mode`, `mesh_role`, anything captured at startup).
- Adding a new setting: extend `DEFAULTS`, add validation in `_validate` if non-trivial, document the key here.
- Don't store secrets the user expects to keep private outside the machine — `mri_settings.json` is plaintext.
