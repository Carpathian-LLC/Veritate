# veritate_mri overview

## What it is

The Flask app and runtime that serves the dashboard. Lives at [veritate_mri/](../../../veritate_mri/). Contains routes, readers, training-side orchestration, runtime services, and the in-memory log ring.

## Subdirectories

| Directory                                                              | Purpose                                                                     |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [app.py](../../../veritate_mri/app.py)                                 | Flask entry point — see [app_py.md](app_py.md)                              |
| [routes/](../../../veritate_mri/routes/)                               | Route modules — see [routes.md](routes.md)                                  |
| [readers/](../../../veritate_mri/readers/)                             | Data layer — see [readers.md](readers.md)                                   |
| [training/](../../../veritate_mri/training/)                           | Trainer subprocess management, save, dump suite, sync, export, build runner |
| [training/sync/](../../../veritate_mri/training/sync/)                 | App-sync daemon (remote code pushes)                                        |
| [runtime/](../../../veritate_mri/runtime/)                             | Heartbeat, settings, lifecycle, sys_metrics, logs ring                      |
| [inference/](../../../veritate_mri/inference/)                         | PyTorch brain + decode strategies + agent — see [inference_brain.md](inference_brain.md) |
| [web/](../../../veritate_mri/web/)                                     | Static frontend assets (see [../frontend/README.md](../frontend/README.md)) |
| [eval/](../../../veritate_mri/eval/)                                   | Standalone evaluation helpers                                                |
| [tools/](../../../veritate_mri/tools/)                                 | One-shot CLI tools (corpus builders, etc.)                                  |
| [teacher/](../../../veritate_mri/teacher/)                             | Teacher (Ollama / API) integration                                           |

## Startup overview

[app.py:169–327](../../../veritate_mri/app.py#L169) orchestrates startup. Order matters:

1. CLI args + model discovery.
2. Pre-build hook registration (closes C engine for rebuilds).
3. Optional minimal-mode skip (no idle watcher, no sys-warm, no app-sync).
4. Heartbeat provider + start.
5. Route registration (16+ modules).
6. Mesh setup (conditional on `mesh_role`).
7. App-sync reload hook bind.
8. Optional eager brain load on background thread.

## Pitfalls

- Code changes anywhere imported at startup require a dashboard restart.
- The log ring is in-memory. Server restart wipes it; export logs to disk before restarting if needed for postmortem.
- `VERITATE_MINIMAL=1` is propagated via env vars across restarts via `LAUNCH_CMD`. Setting it at runtime via the dashboard requires a restart to take effect.
