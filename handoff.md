# handoff

Rolling state summary. Newest facts win. Keep it short.

## boxes

| box | hardware | repo | dashboard |
| --- | --- | --- | --- |
| mirach | M3 Ultra, 256 GB unified | `/Users/mirach-00-usc1/Development/Veritate` | :8001 |
| fortis | Windows, RTX 5070 **12 GB VRAM** | `C:\GitHub\Veritate` (NOT `C:\Users\malka\Veritate`, that is a stale decoy) | :8001, reachable via `ssh fortis` |

fortis is reached with `ssh fortis`; it is a Windows box, so use `powershell -NoProfile -EncodedCommand <base64 utf-16le>` to avoid quoting hell. Its dashboard must be launched as an INTERACTIVE scheduled task (`schtasks /it`) or the browser window will not appear on the physical desktop and the process dies when the SSH session closes.

## live as of 2026-07-29

- **fortis: `core_50m`** — the one model. **50.64M params**, `trunk=dense`, seq 2048, batch 2, **AdamW** (NOT muon — `native_trainer.py` only ever had `torch.optim.AdamW`; an earlier version of this file said muon and was wrong), bf16, wsd, base_lr 6e-4. `total_steps 11,250,585 = 46.08 GB of bytes = 200 tokens/param`. Context is 2 KB.
  - **RESTARTED 2026-07-29 at step 1,650,000 (29.3 tok/param) onto a prose mix.** The first 1.65M steps ran on 100% `the_pile`, whose own catalog entry sets `recommended_min_params: 500000000` — a 50.6M model was spending its budget on ArXiv LaTeX, patents, and Latin-1 mojibake. Remaining 39.33 GB now draws 95% prose/chat:
    `fineweb_edu:0.30,slimpajama627b:0.20,openwebtext10g:0.20,chat_5gb:0.15,veritate_v1:0.05,wikitext103:0.05,the_pile:0.05`
  - **Why it had no val for 27 hours:** `the_pile` is the ONLY corpus on fortis with no `_val.bin`, and validation is gated on a truthy val path, so the entire eval loop silently no-oped. `resolve_and_weight` sorts **weight-descending**, so val follows the HEAVIEST corpus, not the first stem written — `fineweb_edu` holds the top weight specifically so val resolves. Fixed in the trainer: `resolve_val_path()` now returns a loud WARNING instead of silence (6 tests in `tests/training/test_val_path_resolution.py`).
  - **Resume does NOT restore the shape.** `apply_resume_overrides` reads `cfg["training_args"]`; core_50m's config.json is FLAT and has no such key, so nothing is restored and `--size` falls back to `default_size: 200m`. First relaunch built a 203.7M model and died on a shape mismatch. **Always pass `--size` explicitly on resume.** (It failed safely — torch raises on shape mismatch even under `strict=False`, which only governs missing/unexpected keys.)
  - **Batch is NOT a throughput lever here.** Measured ramp at seq 2048: batch 1 = 49.0k, 2 = 56.0k, 4 = 54.0k, 8 = 54.1k, 12 = 54.8k, 16 = 56.0k tok/s. Flat across 8x VRAM (1.14 -> 8.02 GB). The 5070 is compute-bound at batch 2, so batch 2 was correct all along.
  - **`--n_chunks` and `state_carry` DO NOTHING in `native_trainer.py`** — declared in argparse, never read. A first launch was budgeted 16x short because of it. `vanilla_trainer.py` DOES honor n_chunks (measured 196,660 tok/step on mirach). Verify every run with `tok_per_s / step_rate` before trusting the config.
  - The bench is unreliable in BOTH directions: over-predicted 53% on mirach, under-predicts here (56.0k predicted vs 70.2k measured). Use it for VRAM fit and relative shape only; live-log arithmetic is the only trustworthy absolute.
- **fortis: `core_200m`** — STOPPED at step 86,975 of 164,388. Resumable from ckpt 86,500. Headed for 4.4 tokens/param, i.e. the same dead end as wren. Kept on the user's instruction.
- **mirach: `wren_sft`** — vocabulary continuation, step ~66,875 of 146,014 at 12,830 tok/s, ~14 d left. batch 48 x seq 1024 x n_chunks 4 = 196,608 tok/step (verified from the log). Targets 20 tok/param. Post-SFT eval measured **43.9% form obedience on 280 items** (up from 24.3%, p=1.0e-6).

Deleted this session: `chat_200m`, `chatty_200m`, `wren` (both boxes), `chin200m`. One model per box is the standing goal — no experiment forks, no A/B arms, no version suffixes.

## trainers repo RETIRED 2026-07-29

The separate `Veritate-Trainers` repo is gone. The trainer ships with the platform.

- `trainers/common/vanilla_trainer.py` -> **`veritate_mri/training/veritate_trainer.py`** (tracked platform code, arrives by app update)
- `trainers/common/trainer_sizes.json` -> **`veritate_mri/data/trainer_sizes.json`** (tracked at last; it was untracked in BOTH repos with no fallback in `load_sizes_doc()`, and `load_native_sizes()` runs at import time, so losing that one file killed the Training tab)
- Deleted: `trainers/common/`, `trainers/.sync_state.json`, `trainers/readme.md` (described the retired per-folder `trainers/<name>/trainer.py` model), `veritate_mri/training/sync/trainers_sync.py`, the `/trainers/git/{status,sync,check,files}` routes, the Settings "Trainers" sources panel (15 lines HTML + 513 lines JS), its tutorial step, and `developer_documentation/architecture/backend/native_trainer.md`.
- Kept: `trainers/corpus/` (67 bins / 71 GB on mirach, 75 bins on fortis) and `trainers/.gitignore`, which is what stops those bins ever entering git.
- `_activeTrainingName()` was preserved out of the deleted JS block — it is used elsewhere (index.js:12506) and is not sync-specific.
- Nothing under `trainers/` was ever git-tracked, so these deletions do NOT show up in `git status`.

Ruff clean, 997 tests pass. Moving the trainer into `veritate_mri/` put it under ruff for the first time; `pyproject.toml` now carries `"veritate_mri/training/veritate_trainer.py" = ["E402", "T20"]` (deliberate `sys.path` bootstrap, and stdout IS the run log), and 10 genuine never-linted issues were fixed rather than ignored.

Component doc: `developer_documentation/architecture/backend/veritate_trainer.md`.

## corpus moved to data/corpus/ 2026-07-29

`paths.CORPUS_ROOT` is now `data/corpus/` (repo-root `data/`, which IS in `DEFAULT_SKIP_DIRS`; note `veritate_mri/data/` is NOT, since only top-level names match — that is exactly why platform data belongs there and bulk local data belongs at the root). Downloads, builders and the mix planner all write there.

- `paths.LEGACY_CORPUS_ROOT` (`trainers/corpus/`) is still **read**. `paths.corpus_search_dirs()` returns both, canonical first, and anything that lists or globs corpora must walk both or a legacy install goes blind to its own data.
- `paths.corpus_train_path()` / `corpus_val_path()` resolve an EXISTING file wherever it lives, and fall back to the canonical root for a stem that does not exist yet — so a new download lands in `data/corpus/` while an installed legacy corpus still resolves and still uninstalls.
- `corpus_sync._train_path` / `_val_path` delegate to those, so install-state detection sees legacy corpora instead of reporting them missing.
- `_free_disk_bytes` now walks up to the nearest existing ancestor: `data/corpus/` may not exist on a fresh install, and `disk_usage` on a missing path raises, which would have silently disabled the pre-download disk check.
- `veritate_mri/data/corpus/` is unrelated and unchanged — it is the staging area for Veritate-native corpora the Settings library copies into the working corpus dir.
- Bins were **copied, not moved**: 67 bins / 71 GB now in both places, file lists identical. `trainers/corpus/` stays until no run holds the bins open (fortis ~6.5 d, mirach ~14 d), then delete it. Windows refuses to rename a file the trainer has mapped, which is the only reason this is two steps.
- 9 tests in `tests/mri/test_corpus_roots.py` pin the two-root behaviour.

## typing recorder REMOVED 2026-07-29

The Settings "Typing recorder" panel and its sample store are gone: `runtime/typing_samples.py`, the `/typing/samples` and `/typing/samples/<name>` routes, `tests/mri/test_typing_samples.py`, `developer_documentation/architecture/backend/typing_samples.md`, 383 lines of JS, 30 lines of HTML, 27 lines of orphaned CSS, and `data/typing_samples/` (7 recorded sessions, backed up to the session scratchpad since recorded typing cannot be regenerated).

**The live draft/prefetch feature was NOT removed.** `_recordTypingGap`, `_typingGaps`, `_typingMedianMs`, `_pauseBaseMs`, `_trailingWord` and `speculative_pause_ms` all stay — the gap recorder was fed by BOTH the composer and the recorder, and the composer is the shipped feature. Only the recorder side went.

## the finding that reframed everything

**Bytes are not tokens.** Models here are byte-level (vocab 256), so the trainer counts BYTES while every scaling law is quoted in word-pieces. Measured on the real corpora with a 151k-vocab tokenizer: **4.55 bytes/token for prose, 4.12 for code.**

Every model built before 2026-07-28 was sized by counting bytes as tokens and is undertrained ~4x:

| model | bytes | tokens | tok/param |
| --- | --- | --- | --- |
| chat_200m | 2.01B | 0.44B | 1.6 |
| chin200m | 5.41B | 1.19B | 4.4 |
| core_200m | 4.04B | 0.89B | 4.4 |
| wren | 5.90B | 1.30B | 4.8 |
| wren_sft | 8.18B | 1.80B | 6.7 |

Chinchilla is 20. Nothing reached 7. The symptom is **correct output shape with invented words** ("drums, tsyllables, saxophones, pedals") — the model learned the format and never read enough English to lock the lexicon. No amount of instruction tuning moves it. Full rule in `veritate_mri/data/wiki/concepts/model_sizing.md`; roster in `developer_documentation/training/model_roster.md`.

## eval instrument

`ifeval_form.json` grew **26 -> 280 items**, >=20 per rule family, 0 contamination over 257.5 MB. The old 26-item set could not resolve a 12-point move (9/26 vs 12/26 is p=0.40), so the "wren_sft 46.2%" success was **withdrawn** and reopened as IDEA 10 with a pre-registered falsifier. Any "form %" quoted before 2026-07-28 came off the retired set and is not comparable.

## open, needs the user

- **`trainer_sizes.json` is UNTRACKED IN BOTH REPOS and has no fallback.** It is the single owner of all 34 sizes and every tuned default. `native_sizes_path()` resolves it to `trainers/common/trainer_sizes.json`; the location claimed by `readers/trainers.py:52` AND by CLAUDE.md — `veritate_mri/data/trainer_sizes.json` — **does not exist**. `load_sizes_doc()` has no try/except and `load_native_sizes()` runs at import time, so if that one untracked file goes missing the Training tab stops importing. It survives today only as a local file on mirach plus a hand-copy on fortis. Commit it.
- **Deprecate the upstream trainers repo** (user's proposal 2026-07-29, agreed). Its reason for existing — distributing 19 per-size trainers — died 2026-07-27; it now ships one file. Costs of keeping it, all hit this session: trainer edits are invisible to `git status` in Veritate; the updater SKIPS `trainers/` so a platform update cannot deliver a trainer fix (fortis's copy was 83 lines stale and missing the `sys.path` bootstrap it needs as a standalone entry point); and `trainer_sizes.json` is untracked. Blocker: `trainers/` is skipped precisely because `trainers/corpus/` holds ~100 GB of bins, so either narrow the skip to `trainers/corpus` or move bins to `data/corpus/`. Then `trainer_sizes.json` becomes tracked platform data and `/trainers/git/sync` retires.
- 19 stale `veritate_*` per-size trainer dirs were deleted from fortis; `trainers/` there is now `common/` + `corpus/` only, matching canonical.

## updater — FIXED and verified in production 2026-07-29

The "behind" counter that never cleared is fixed and confirmed on fortis. `pull_update` applies a TARBALL, so `.git/HEAD` never advances and comparing against it left `behind` permanently stuck. `app_sync.py` now records `pulled_commit` (the real remote branch sha) on pull and `_compare_base()` prefers it, with `head_short` built from that sha instead of the ETag. Verified live: fortis went `behind: 7, head_short e68f62a` -> pull 596 files -> `behind: 0, head_short 80501e4, update_available: false`, cleared on the FIRST pull.

Two facts about the updater worth keeping:
- **`DEFAULT_SKIP_DIRS` resolves to `('.git', '.venv', '__pycache__', 'data', 'experiments', 'models', 'trainers', 'venv')`** — `PLUGINS_ROOT` basename is `trainers`. So models, checkpoints, corpus bins and the sizes table all survive an update. Updating a box mid-training does NOT endanger a checkpoint.
- **The updater never DELETES.** It only copies incoming files, so files removed upstream linger. `veritate_mri/training/native_trainer.py` is still on fortis after 80501e4 deleted it. Harmless here (nothing references it — `NATIVE_TRAINER_PATH` was repointed at `vanilla_trainer.py` in the same commit) but it means stale modules accumulate.

## corpus

fortis holds ~100 GB of bins including **the_pile 50 GB** and **slimpajama627b 20.5 GB**; mirach has ~25 GB of prose. Corpus size, not GPU time, was the binding constraint before those landed.
