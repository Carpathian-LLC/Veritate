# app.py — Flask app startup

## What it is

Single entry point for the dashboard Flask app at [veritate_mri/app.py](../../../veritate_mri/app.py). Parses CLI args, wires startup behaviors, registers route modules, starts background threads.

## How it works

Startup sequence at [app.py:169–327](../../../veritate_mri/app.py#L169):

1. **CLI args** ([line 170](../../../veritate_mri/app.py#L170)) — `--model`, `--step`, `--port`, `--threads`.
2. **Default model resolution** ([line 178](../../../veritate_mri/app.py#L178)) — `_resolve_pytorch_model(args.model)`.
3. **Thread budget** ([line 182](../../../veritate_mri/app.py#L182)) — physical-core auto-detect capped at 16.
4. **C-subprocess pre-build hook** ([line 197](../../../veritate_mri/app.py#L197)) — closes the C engine before a rebuild so the binary isn't locked.
5. **Minimal-mode check** ([line 199](../../../veritate_mri/app.py#L199)) — `VERITATE_MINIMAL=1` skips idle watcher, sys-warm, app-sync, eager brain load.
6. **Heartbeat training provider** ([line 205+](../../../veritate_mri/app.py#L205)) — defines `_heartbeat_training` with primary (`plugin_runner.state()`) + fallback (`train.csv` mtime scan). See [heartbeat.md](heartbeat.md).
7. **Heartbeat start** ([line 234](../../../veritate_mri/app.py#L234)) — `heartbeat_mod.start()`.
8. **Route registration** ([lines 121–141](../../../veritate_mri/app.py#L121)) — each module under [veritate_mri/routes/](../../../veritate_mri/routes/) exposes `register(app)`. See [routes.md](routes.md).
9. **App-sync reload hook** — binds `lifecycle.restart` for remote code pushes.
10. **Eager-load brain** (optional) — background thread loads PyTorch if `pytorch_load_mode="always"`.

## Exception handling

Global catch-all at [app.py:81](../../../veritate_mri/app.py#L81) catches every route exception, logs to the in-memory log ring (visible in the Logs tab), and returns JSON `{"ok": false, "error": ...}` — never HTML, so the frontend's `r.json()` never breaks.

## Dependencies

- [readers/](../../../veritate_mri/readers/) — all disk reads go through these.
- [training/trainer_runner.py](../../../veritate_mri/training/trainer_runner.py) — subprocess management.
- [runtime/heartbeat.py](../../../veritate_mri/runtime/heartbeat.py) — Carpathian webhook.
- [runtime/settings.py](../../../veritate_mri/runtime/settings.py) — config store.
- [runtime/sys_metrics.py](../../../veritate_mri/runtime/sys_metrics.py) — hardware detection.

## Pitfalls

- Code changes to `app.py` (or anything imported at startup) require a dashboard restart to take effect. The Flask `debug=True` reloader is not used in production launches.
- The order of route registration matters when two modules want to claim the same path. Currently all paths are unique.
- `VERITATE_MINIMAL=1` should be set in the environment BEFORE the Python process starts — runtime changes are ignored.
