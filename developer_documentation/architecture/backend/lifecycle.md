# lifecycle

## What it is

Server lifecycle controls at [veritate_mri/runtime/lifecycle.py](../../../veritate_mri/runtime/lifecycle.py). Used by the app-sync daemon when a remote code push lands, and exposed via [lifecycle_routes.py](../../../veritate_mri/routes/lifecycle_routes.py) for the dashboard's restart, soft-reload, and kill buttons.

## How it works

Every action takes the Flask app config and routes through the shared `_cleanup(app_config, ...)`; the `stop_plugin` / `stop_build` flags are what separate them.

- `restart(app_config)` re-execs the Python process with the original command line captured at startup (`app.config["LAUNCH_CMD"]`), after closing the C engine subprocess and shutting down the build runner. The new process re-runs `veritate.py`, which re-installs the venv if needed and re-launches the Flask app.
- `soft_reload(app_config)` closes only the C engine subprocess, then re-execs. The build runner keeps running.
- `kill(app_config)` runs the full cleanup including stopping the training subprocess, then exits via `os._exit`. Nothing comes back without a manual relaunch.
- `restart_with_flag_toggle(app_config, add_flags=(), remove_flags=())` re-execs with the launch flags edited, backing `/sys/mode/relaunch`.

The training subprocess survives `restart` and `soft_reload`; the new server reattaches through `trainer_runner`'s PID file.

The app-sync daemon ([training/sync/app_sync.py](../../../veritate_mri/training/sync/app_sync.py)) owns the reload hook: [app.py](../../../veritate_mri/app.py) defines `_app_sync_reload()` as `lifecycle.restart(app.config)` and registers it with `app_sync.set_reload_hook(_app_sync_reload)` ([app_sync.py:883](../../../veritate_mri/training/sync/app_sync.py#L883)), so a detected update triggers a restart automatically.

## Dependencies

- [app.py](../../../veritate_mri/app.py): captures `LAUNCH_CMD` and registers the app-sync reload hook.
- [training/sync/app_sync.py](../../../veritate_mri/training/sync/app_sync.py): holds the hook and fires it on update.
- [training/trainer_runner.py](../../../veritate_mri/training/trainer_runner.py), [training/build_runner.py](../../../veritate_mri/training/build_runner.py): stopped by `_cleanup` per action.

## Pitfalls

- A restart kills in-flight HTTP connections. SSE clients reconnect; one-shot fetches in the dashboard fail with a network error and surface via `_backendErrMsg`.
- Restart does NOT kill detached training subprocesses (PPID=1 after `nohup`). It DOES kill `trainer_runner`-managed subprocesses (they're children of the dashboard).
- The reload hook lives on `app_sync`, not on this module. `lifecycle` exposes no `set_reload_hook`.
