# warm models (C-engine warm pool)

## What it is

A set of trained models kept permanently loaded as resident C-engine subprocesses so switching to one serves warm: no spawn, no reload. The pool lives at `cfg["C_WARM"]` (a `{model_name: CTracedSubprocess}` dict) and is owned by a few functions in [backends_routes.py](../../../veritate_mri/routes/backends_routes.py). The selection is persisted as the `warm_models` list in settings.

## How it works

Each warm entry is a normal [`CTracedSubprocess`](../../../veritate_mri/inference/backends/c_engine.py): an independent process driven over its own stdin/stdout, with its own per-model-dir `VERITATE_STATE_CACHE`. Multiple entries are resident at once with no shared global (no fixed port, no singleton lock), so the single-slot `cfg["C_SUBPROCESS"]` is just re-pointed at a pool entry rather than respawned.

Owning functions in [backends_routes.py](../../../veritate_mri/routes/backends_routes.py):

- `warm_spawn(cfg, name)`: spawn `name` into the pool (idempotent; revives a died entry via `CTracedSubprocess._ensure_alive`). Adopts the active single-slot `C_SUBPROCESS` when it already serves `name`, instead of spawning a duplicate. Skips with a plain log line and returns `False` when the model has no `.bin` or the engine binary is not built.
- `warm_select(cfg, name)`: point the active C slot (`C_SUBPROCESS`/`C_EXE`/`C_MODEL`) at a pool entry without spawning; revives a died entry. Returns `False` when `name` is not pinned.
- `warm_drop(cfg, name)`: close + remove a pinned model, unless it is the active `C_SUBPROCESS`, in which case it is only unpinned and left running for the normal single-slot lifecycle.
- `warm_is_pinned(cfg, sub)`: object-identity membership test.
- `warm_forget(cfg, sub)`: drop a subprocess from the pool by identity (used when it is closed through another path, e.g. the `/backends/c` unload).
- `warm_apply(cfg, names)`: reconcile the pool to `names`: spawn newly added, drop removed.
- `warm_eager_start(cfg, names)`: spawn every warm model off a daemon thread so startup is not serially blocked.
- `_warm_status(cfg)`: per-servable-model status list (see below).

**Routing.** The chat switch path `_ensure_c` ([hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py)) calls `warm_select` first. On a hit it re-points the slot; the outgoing subprocess is closed only when it is *not* warm-pinned (`warm_is_pinned`). A non-pinned model keeps today's single-slot swap. This means switching between two pinned models is instant and neither is ever killed.

**Startup.** `app.main()` ([app.py](../../../veritate_mri/app.py)) calls `warm_eager_start` for `settings.warm_models` after the app is built (skipped in `MINIMAL` mode). The pre-build hook `_close_c_for_rebuild` closes the whole pool to release the binary lock before an engine rebuild.

**Settings change.** `POST /settings` with a `warm_models` key calls `warm_apply` ([settings_routes.py](../../../veritate_mri/routes/settings_routes.py)), mirroring the `pytorch_load_mode="always"` eager-load hook: newly added models spawn, removed models close (unless active).

**Eviction.** There is no automatic C-side eviction: the idle watcher ([app.py](../../../veritate_mri/app.py) `_pytorch_idle_watcher`) only unloads the PyTorch brain, never a C subprocess, so a pinned model is never idle-unloaded.

## Status payload

`_warm_status(cfg)` adds a `warm` array to the `c` block of `GET /backends` (`_backends_status_payload`): one entry per model with a servable `.bin`, `{name, bin_bytes, pinned, resident, active}`. `resident` reflects a live pool subprocess (`_sub_alive`), `active` is the one currently in `C_SUBPROCESS`. The Warm-models settings panel renders from this list (see [../frontend/warm_models_panel.md](../frontend/warm_models_panel.md)).

## Dependencies

- [inference/backends/c_engine.py](../../../veritate_mri/inference/backends/c_engine.py): `CTracedSubprocess`, `_ensure_alive`, `close`.
- [runtime/settings.py](../../../veritate_mri/runtime/settings.py): the `warm_models` list (`DEFAULTS`, validated in `_validate`).
- [readers/](../../../veritate_mri/readers/): `bin.exists`, `paths.bin_path`, `paths.engine_binary_path`, `models.list_models`.

## Pitfalls

- Warm models each hold their `.bin` in memory continuously. The panel sums selected sizes against available RAM and soft-warns on overflow; it does not hard-block (operator's call).
- A rebuild of the engine closes the pool (binary lock); the models re-warm on the next server restart from the persisted `warm_models` list, not automatically after the build.
- Selecting the active model is a no-op (the `C_MODEL` equality check in `_ensure_c` returns early before touching the pool).
- The manual `/backends/c` unload closes the active subprocess and `warm_forget`s it so no dead pool reference lingers; a still-pinned model re-warms on the next settings apply or restart.
- **Model discovery requires a checkpoint file, not just a .bin.** The `/pytorch-models` endpoint (which populates the frontend chat/generation model dropdowns and the `<select id="ch-modelSel">` picker) filters via `_model_rows()` in [models_routes.py](../../../veritate_mri/routes/models_routes.py), which calls `checkpoints.latest_step(name)` and skips any model that returns None, that is any model whose `checkpoints/` subdir contains no `step_<N>.pt` file. A model can be fully servable (bin present, warm-pool resident, `/backends/c` reports `loaded=true`, `/backends` warm list includes it) yet invisible to the picker. This bites hard on lean deployments that ship only `veritate.bin` + `config.json` + `neuron_memory.json` to a serving box to avoid transferring 2 GB `.pt` files. Fix: include at least a zero-byte placeholder at `models/<name>/checkpoints/step_<N>.pt` (the regex is `^step_(\d+)\.pt$`, content is not inspected), or ship the real checkpoint.
