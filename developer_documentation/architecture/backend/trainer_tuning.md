# trainer_tuning

## What it is

Machine-local per-trainer setting overrides at [veritate_mri/readers/trainer_tuning.py](../../../veritate_mri/readers/trainer_tuning.py). Auto-optimize results and last-used launch args land here instead of `trainer_sizes.json`, so a machine's benchmarked settings survive an app update and never leak to other machines. The store is keyed by `plugin_id`, each entry holding `{args, measured}`.

## How it works

- Backing file: `REPO_ROOT/data/trainer_tuning.json` ([TUNING_PATH](../../../veritate_mri/readers/trainer_tuning.py#L26)), gitignored, alongside the other machine-local stores (`mri_settings.json`, `system_specs.json`). Writes are atomic (`.tmp` + `os.replace`).
- [`args_for(plugin_id)`](../../../veritate_mri/readers/trainer_tuning.py#L50): returns the stored `args` dict for a trainer, or `{}`.
- [`save(plugin_id, args, measured=None)`](../../../veritate_mri/readers/trainer_tuning.py#L56): merges `args` into the trainer's entry (last write wins per key), optionally records the `measured` benchmark, and returns True when the entry changed.
- [`trainers.scan()`](../../../veritate_mri/readers/trainers.py#L205) overlays the stored `args` onto each record's `manifest.defaults` via `_overlay_tuning()`, replacing only keys already present in the defaults. The on-disk manifest is never touched.
- Writers: [`trainers.update_defaults`](trainer_plugins.md) (on every launch) and the dashboard's Auto-optimize apply (`POST /trainers/tune_defaults`, which also passes `measured`).

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py) for `REPO_ROOT`.
- Consumed by [readers/trainers.py](trainer_plugins.md) (`scan`, `update_defaults`).

## Pitfalls

- `trainers/` is an upstream-synced checkout, so per-machine tuning MUST live here, not in `manifest.json`: a manifest edit would be overwritten by the next sync and would ship to every machine. This store is the only correct home for benchmarked-per-box settings.
- Only keys already present in a trainer's manifest defaults are overlaid; a tuning key with no matching default is stored but never surfaces on the form.
