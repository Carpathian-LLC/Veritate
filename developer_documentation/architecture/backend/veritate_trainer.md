# veritate_trainer

## What it is

THE trainer. One file, every size: [veritate_mri/training/veritate_trainer.py](../../../veritate_mri/training/veritate_trainer.py). It owns the training loop, the LR schedule, the data loader, the chunked forward, the resume logic, the QAT and 8-bit-Adam plumbing, and the `config.json` bootstrap.

It ships with the platform, so it arrives by app update like any other file and is edited in this repo like any other file.

## History

Two consolidations produced the current shape.

- **2026-07-27** — the 19 per-size trainer folders (`veritate_10m` … `veritate_1t`) were deleted. All 19 `trainer.py` files were byte-identical apart from a `PLUGIN_ID` string, and their manifests held only per-size numbers. Sizes became data.
- **2026-07-29** — the separate `Veritate-Trainers` repo was retired and the trainer moved into `veritate_mri/training/`, renamed from `vanilla_trainer.py`. A single file needs no distribution channel of its own, and living outside `veritate_mri/` had two concrete costs: the app updater skips `trainers/`, so a trainer fix could not ride an update (fortis's copy drifted 83 lines stale and was missing the `sys.path` bootstrap it needs as a standalone entry point), and edits were invisible to `git status` in this repo. `trainers_sync.py`, `.sync_state.json`, the `/trainers/git/*` routes and the Settings "Trainers" panel went with it.

Do not recreate per-size trainers. Do not add `trainers/<name>/` directories.

## Sizes are data

Shapes and tuned defaults live in [veritate_mri/data/trainer_sizes.json](../../../veritate_mri/data/trainer_sizes.json) — 34 sizes from 5m to 1t. Three layers resolve in order: `shared_defaults` → that size's `defaults` → whatever the user sets in the Training tab.

`_sizes_path()` in the trainer and `native_sizes_path()` in [readers/trainers.py](../../../veritate_mri/readers/trainers.py) resolve the same file, so the set the dashboard offers can never drift from the set the trainer accepts. The trainer keeps a beside-the-file fallback for older checkouts synced from the retired repo, where the table shipped next to the trainer.

**Never hardcode a tunable in this file** (preflight rule 11). If a user might reasonably want to change a number, it belongs in `trainer_sizes.json` with a control on the Training tab. A literal in the trainer is a bug.

## How it is launched

Always through the dashboard: `POST /trainers/run` with `{id: "native/trainer", args: {...}}`. `readers/trainers.py` exposes it as a synthetic trainer record whose `path` is `NATIVE_TRAINER_PATH`; `trainer_runner` builds argv from that path like any plugin. It is also runnable standalone — it puts the repo root on `sys.path` itself (`veritate_mri/training` → `veritate_mri` → repo root) rather than relying on a caller.

## Resume: two traps

- **The shape is NOT restored.** `apply_resume_overrides()` reads `cfg["training_args"]`. Configs written by the old `native_trainer.py` are FLAT and have no such key, so nothing is restored and `--size` falls back to `default_size` (`200m`). Always pass `--size` explicitly on a continue. Shape mismatches do fail loudly — `torch.load_state_dict` raises on a size mismatch even under `strict=False`, which only governs missing and unexpected keys — but the run dies at startup instead of resuming.
- **Anything you want changed must be passed explicitly.** Every value present in `training_args` is restored unless its flag is on `argv`. A resume that means to switch corpus, eval cadence or checkpoint cadence has to say so.

## Validation

`resolve_val_path()` returns `(val_path, warning)`. `multicorpus.resolve_and_weight()` sorts **weight-descending**, so validation follows the HEAVIEST corpus in a mix, not the first stem written in the spec.

A mix whose top corpus ships no `_val.bin` trains blind, and the eval loop is gated on a truthy val path, so before 2026-07-29 it no-oped in total silence: `core_50m` ran 1.65M steps over 27 hours on `the_pile` with `val_bin: ""` and never said a word. The trainer now prints a WARNING naming the consequence. Pinned by [tests/training/test_val_path_resolution.py](../../../tests/training/test_val_path_resolution.py).

## Verify every budget from the live log

`tok_per_s / step_rate` must equal the intended `batch × seq × n_chunks` before you trust a step budget. Two reasons this is not paranoia:

- The retired `native_trainer.py` declared `--n_chunks` and `state_carry` in argparse and **never read them**, which under-budgeted a run 16x. This trainer does honor `n_chunks` (measured 196,660 tok/step on mirach).
- The bench is unreliable in both directions on absolute throughput — it over-predicted 53% on mirach and under-predicts on fortis (56.0k predicted vs 70.2k measured). Use it for VRAM fit and relative shape only.

## Note on `trainers/`

The directory still exists but holds only `corpus/` — the `.bin` training data, in the updater's `DEFAULT_SKIP_DIRS` so it survives app updates. It contains no code. Moving it to `data/corpus/` is the last step of this consolidation.

## Dependencies

- [veritate_core/plugin/](../../../veritate_core/plugin/): `save`, `paths`, `model`, `qat`, `multicorpus`, `bench`, `hardware`, and feature-detected `optim`, `model_patched`, `model_recurrent`, `model_memory`.
- [training/save.py](save.md): CSV append, checkpoint save, `config.json` provenance.
- [readers/trainers.py](trainer_plugins.md): the catalog record, size table, and launch path.
