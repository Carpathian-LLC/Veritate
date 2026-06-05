# lifecycle

## What it is

Graceful Flask restart at [veritate_mri/runtime/lifecycle.py](../../../veritate_mri/runtime/lifecycle.py). Used by the app-sync daemon when a remote code push lands, and exposed via [lifecycle_routes.py](../../../veritate_mri/routes/lifecycle_routes.py) for the dashboard's "restart" button.

## How it works

`restart(config)` re-execs the Python process with the original command line captured at startup (`app.config["LAUNCH_CMD"]`). The new process re-runs `veritate.py`, which re-installs the venv if needed and re-launches the Flask app.

The app-sync daemon (`training/sync/app_sync.py`) calls `lifecycle.set_reload_hook(_app_sync_reload)` from [app.py](../../../veritate_mri/app.py) so that detected updates trigger a restart automatically.

## Dependencies

- [app.py](../../../veritate_mri/app.py) — captures `LAUNCH_CMD` and sets the reload hook.
- [training/sync/app_sync.py](../../../veritate_mri/training/sync/app_sync.py) — triggers restart on update.

## Pitfalls

- A restart kills in-flight HTTP connections. SSE clients reconnect; one-shot fetches in the dashboard fail with a network error and surface via `_backendErrMsg`.
- Restart does NOT kill detached training subprocesses (PPID=1 after `nohup`). It DOES kill `trainer_runner`-managed subprocesses (they're children of the dashboard).
