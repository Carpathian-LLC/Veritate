# trainer_runner

## What it is

The subprocess manager for trainer plugins at [veritate_mri/training/trainer_runner.py](../../../veritate_mri/training/trainer_runner.py). Spawns one plugin subprocess at a time globally, persists a PID file so a restarted dashboard can re-attach to a still-running run, and exposes `state()` for the heartbeat and dashboard.

## How it works

Public API:

- [`start(plugin_id, args)`](../../../veritate_mri/training/trainer_runner.py#L349) — finds the plugin manifest, builds `python trainers/<id>/trainer.py --arg val ...`, spawns it, writes `.plugin_pid.json`, tails stdout into the in-memory log ring.
- [`state()`](../../../veritate_mri/training/trainer_runner.py#L245) — returns `{status, plugin_id, args, started_at, finished_at, exit_code}`. Statuses are `idle`, `running`, `ok`, `failed`, `stopped`.
- [`stop()`](../../../veritate_mri/training/trainer_runner.py#L372) — SIGTERM the subprocess.
- [`is_running()`](../../../veritate_mri/training/trainer_runner.py#L250) — boolean shortcut.

Globals: `_LOCK`, `_STATE`, `_PROC`. Single-instance enforcement: `start()` returns an error if `_STATE["status"] == STATUS_RUNNING`.

PID-file persistence at `.plugin_pid.json` lets the dashboard reattach after a server restart: on module load, `_recover_from_disk` ([line 205](../../../veritate_mri/training/trainer_runner.py#L205)) finds a still-alive PID by command-marker match and restores `_STATE`.

Environment variables set on every spawn:

- `VERITATE_PLUGIN_ID` — plugin knows its own ID.
- `VERITATE_DEVICE` — device preference (`cpu`, `mps`, `cuda`, or unset for auto).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — GPU memory hygiene.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. — capped at physical cores (max 16).

## Dependencies

- Plugins under [trainers/](../../../trainers/) — see [trainer_plugins.md](trainer_plugins.md).
- [readers/trainers.py](../../../veritate_mri/readers/trainers.py) — plugin discovery and manifest reading.
- [runtime/heartbeat.py](../../../veritate_mri/runtime/heartbeat.py) — reads `state()` for presence pings; `record_training_event` is called from `start()` ([line 363](../../../veritate_mri/training/trainer_runner.py#L363)).

## Pitfalls

- One-at-a-time is global, not per-GPU. Two trainers can't run concurrently even on different devices; this is intentional (shared CPU + memory budget).
- Subprocesses survive dashboard exit on Windows but not always on macOS/Linux unless launched with `nohup` or detached process group. The dashboard's `start()` does not detach; killing the dashboard usually kills the subprocess.
- Direct-script trainers launched outside `start()` are invisible to `state()`. The [heartbeat fallback](heartbeat.md) at [app.py:223](../../../veritate_mri/app.py#L223) uses `train.csv` mtime to detect those.
