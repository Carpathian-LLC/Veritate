# app_sync

## What it is

The platform self-updater, [veritate_mri/training/sync/app_sync.py](../../../veritate_mri/training/sync/app_sync.py). It downloads the GitHub source tarball for the active channel's branch and overwrites tracked source in place. No git binary required. `sys_routes` exposes it as `GET /app/update_status`, `POST /app/update_check`, `POST /app/update_pull`, `GET /app/local_edits`, `POST /app/update_channel`.

## How it works

- Channels map to branches: `stable` -> `main`, `experimental` -> `experimental`, `development` -> `dev` (`CHANNEL_BRANCHES`). A `.git` checkout's own branch takes precedence over the channel setting (`_active_branch`).
- `check_update()` prefers the GitHub compare API against local HEAD, so a commit pushed from this machine leaves `behind` at 0. The tarball ETag is the fallback for installs without `.git`.
- `pull_update(reload, force, ignore_training)` runs in one order: refuse if a trainer is running, download the tarball, `_scan_incoming()` extracts to a temp dir and hashes every file the pull would write, `local_edits(incoming=...)` gates on conflicts, `_copy_incoming()` writes them, `_write_baseline()` persists the hash map.
- `incoming` doubles as the new baseline: copied bytes are identical to the scanned bytes, so nothing is re-hashed after the write.
- `local_edits(incoming=...)` reports a baseline entry only when the pull would touch that path and the local bytes differ from what lands there. A file already matching upstream is not a conflict; a path upstream no longer ships cannot be overwritten and is dropped. Called with no `incoming` (the `GET /app/local_edits` diagnostic) it reports every drift from the last pull.
- Divergence past `max(STALE_BASELINE_MIN_FILES, len(baseline) * STALE_BASELINE_SHARE)` means the tree moved by some route other than a pull. The baseline is discarded (`has_baseline: false`, `stale_baseline: true`) and the next pull rebuilds it.
- A blocked pull returns `{ok: false, requires_force: true, edits}`. The dashboard lists `edits` in a confirm dialog and re-POSTs with `force: true`, which skips the gate.
- Top-level dirs in `DEFAULT_SKIP_DIRS` (`models/`, `plugins/`, `data/`, `experiments/`, `.git`, `.venv`, `venv`, `__pycache__`) are neither scanned nor written, so user data survives an update. The baseline lives in `data/` for that reason.
- A successful pull deletes `venv/.req_hash` so the launcher re-runs pip on next boot, then fires the registered reload hook when `reload` or the `auto_reload_on_update` setting is set.

## Dependencies

- `urllib`, `tarfile`, `shutil` only. `sync_common.sha256_file` for hashing, `runtime.net.ssl_context()` for TLS, `training.trainer_runner.is_running()` for the training gate, `runtime.settings` for the channel and auto-reload flags.

## Pitfalls

- The updater never deletes files upstream dropped. Orphans from a previous layout stay on disk; they fall out of the baseline at the next pull and stop being reported, but they are not removed.
- `_safe_extract` refuses absolute and `..` tarball entries. Keep that guard ahead of any change to extraction.
- The conflict gate costs a full download before it can refuse. A forced retry downloads again.
- Frontend counterpart is `_appUpdatePullWithGuards` in `veritate_mri/web/index.js`; the confirm text is built by `_appUpdateConfirmOverwrite` from the server's `edits` payload.
