# handoff

Rolling state summary. Newest facts win. Keep it short.

## 2026-07-29: documentation + standards rebuild

- Documentation is now exactly five root files: `claude_preflight.md` (rules, rewritten, 38 rules absorbing coding_roe/agent_roe/claude_merge), `documentation.md` (single platform reference, served by the dashboard wiki tab), `successes.md`/`failures.md`/`ideas.md` (condensed; `research.md` retired). `developer_documentation/` and `veritate_mri/data/wiki/` are deleted; stale in-code doc pointers repointed.
- Wiki tab reworked to serve `documentation.md` (readers/wiki.py single-file loader, `/wiki` + `/wiki/doc` + `/wiki/<slug>/page` routes; training-form learn-more links unchanged). Dashboard needs a reload to pick up index.js; the mirach run is a detached process and survives it.
- New tests: wiki single-file (5), readers/trainers direct (8), trainer_sizes.json schema (4), apply_resume_overrides (4). Suite 1036 passed, ruff clean.
- `veritate_mri/eval/_smoke.py` + sample-data dependency rescued from `experiments/v2/eval_harness/` before the pending experiments purge; `python -m veritate_mri.eval._smoke` passes.
- Awaiting user approval: delete `experiments/` (manifest extracted to ledgers), retire-or-keep `models_sync` + `/models/git/*`, atlas-vs-`/neuron` duplication, vestigial `trainers` key in `versions.json`, and (after the live run ends) `LEGACY_CORPUS_ROOT` + `trainers/corpus/`.

## boxes

| box | hardware | repo | dashboard |
| --- | --- | --- | --- |
| mirach | M3 Ultra, 256 GB unified | `/Users/mirach-00-usc1/Development/Veritate` | :8001 |
| fortis | Windows, RTX 5070 **12 GB VRAM** | `C:\GitHub\Veritate` (NOT `C:\Users\malka\Veritate`, that is a stale decoy) | :8001, reachable via `ssh fortis` |

fortis is reached with `ssh fortis`; it is a Windows box, so use `powershell -NoProfile -EncodedCommand <base64 utf-16le>` to avoid quoting hell. Its dashboard must be launched as an INTERACTIVE scheduled task (`schtasks /it`) or the browser window will not appear on the physical desktop and the process dies when the SSH session closes.

## live as of 2026-07-29

- **fortis: `core_50m` — STOPPED 2026-07-29 21:09 at step 1,875,000 of 11,250,585** (16.7% of budget, **33.3 tok/param**), on the user's instruction, at a checkpoint boundary and deliberately NOT restarted. `checkpoints/step_1875000.pt`, 607,762,263 bytes. Stopped with `POST /trainers/stop`; verified by process list (only the two dashboard pythons left) and a frozen `train.csv`. Validation across the restarted stretch, at 25k intervals: `1.048, 1.074, 1.006, 1.037, 0.982, 1.002, 1.036` — noisy, hovering just above 1.0, no clear descent in the last 100k steps. Resume needs `--size 50m` passed explicitly (see the shape trap below).
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
- Bins were **copied, not moved**: 67 bins / 71 GB now in both places, file lists identical. Windows refuses to rename a file the trainer has mapped, which is the only reason this is two steps.
- **mirach `trainers/corpus/` is still held open** — verified 2026-07-29 with `lsof` on the live trainer (pid 80102): 7 handles into `trainers/corpus/`, not `data/corpus/`. A running trainer resolves its bins once at launch, so it keeps the root it started with. Delete that copy when `wren_sft` ends (~14 d), not before.
- **fortis needs no corpus action at all, and this is why `LEGACY_CORPUS_ROOT` exists.** It holds 75 bins / **101 GB** under `trainers/corpus/` and has no `data/corpus/` at all. Whenever it pulls the consolidation, `corpus_search_dirs()` finds every bin where it already lies. Nothing to move, nothing to re-download.
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

Both items that stood here on 2026-07-29 are closed. `trainer_sizes.json` is tracked at `veritate_mri/data/trainer_sizes.json`, the upstream trainers repo is retired, and the whole consolidation is committed as `b2a678e`.

- **fortis is on `80501e4`, NOT the consolidation `b2a678e`.** Its `/trainers` still reports `C:\GitHub\Veritate\trainers\common\vanilla_trainer.py`, and `trainers/` there still holds `common/`, `.sync_state.json` and `readme.md` — all three deleted upstream, all three lingering because **the updater never deletes**. Deliberately left alone: mirach is where working code lives, and fortis is downstream. Pulling the update is safe whenever the user wants it (no run is active, the corpus resolves from the legacy root, and `veritate_mri/data/trainer_sizes.json` rides the update by design).

## repo cleanup 2026-07-29 (after the consolidation landed)

- **`veritate_mri/training/sync/git_runner.py` DELETED** — 55 lines, zero references anywhere in the tree. Its own header claimed it was "shared by app_sync, plugins_sync, models_sync": `plugins_sync` is `trainers_sync`, deleted with the repo retirement, and `app_sync` parses the `.git` layout directly with **no git binary at all**. Dead since the tarball updater landed.
- **`NATIVE_DEFAULT_SIZE = "85m"` removed from `readers/trainers.py`.** It was a hardcoded tunable (preflight rule 11) that silently overrode the data: `trainer_sizes.json` states the intended prefill *twice* (`default_size: 200m` and `shared_defaults.size: 200m`) and nothing read either key. The manifest now resolves `size_defaults(load_sizes_doc().get(DEFAULT_SIZE_KEY))`. **Behaviour change: a fresh Training form prefills 200m, not 85m.** `NATIVE_DEFAULT_SEQ` and `NATIVE_DEFAULT_VOCAB` went too — both unreferenced duplicates of `shared_defaults`.
- **`developer_documentation/architecture/backend/trainer_plugins.md` rewritten.** Only its bottom sections had been patched, so the doc still opened by describing `trainers/` as a synced upstream checkout and still listed a table of nineteen per-size plugins as current. The plugin *mechanism* is real and kept (`_walk` discovers `trainers/<id>/trainer.py` + `manifest.json`, or a bare `<name>.py`/`<name>.json` pair); what is gone is the claim that any exist.
- 4 genuinely broken doc links fixed: `backend/README.md` -> deleted `native_trainer.md`; `training/settings_index.md` -> deleted `trainers/veritate_80m/manifest.json`; `backend/save.md` -> deleted `trainers/veritate_200m/trainer.py`; `backend/trainer_plugins.md` -> deleted `trainers/.sync_state.json`. A link sweep over `developer_documentation/` + the wiki now reports only 3 hits, all regex false positives on code fragments.
- Stale comment in `index.js` fixed: the form->config direction is `veritate_trainer.write_config`, not `native_trainer._save_args_for_config` (a function that no longer exists under any name).
- **6 new tests, `tests/mri/test_updater_skip_depth.py`.** `DEFAULT_SKIP_DIRS` is matched against the FIRST path segment only, and the whole consolidation rests on that asymmetry: repo-root `data/corpus/` must survive an update untouched while `veritate_mri/data/trainer_sizes.json` must ride it. Same directory name, opposite outcomes, decided purely by depth — and it was pinned by nothing.

Checked and deliberately left alone: `temp/` (19 tracked files that look like scratch but are cited as provenance by `successes.md`, `failures.md` and `ideas.md`, and `build_sft_idk_corpus.py` documents `--in-dir temp/sft_gen`); `build_curriculum_corpus.py` (flagged as unimported only because it is a CLI entry point, and it is a documented builder with 10 siblings); `RESERVED_DIRS` still listing `common` (harmless, still guards a plugin dir of that name). No merge-conflict markers anywhere — the `worklog.md` note claiming unresolved markers at lines 18-31 is stale.

Ruff clean, **1002 tests pass**, 6 skipped, 8 xfailed. `veritate_mri.app` imports, `index.js` passes `node --check`, and a DOM/CSS orphan sweep found nothing left behind by either panel removal (the 18 ids JS reaches that HTML lacks are all dynamically built modals; the 2 "dead CSS ids" are hex colours).

## full suite on fortis (Windows) 2026-07-29 — 985 passed, 1 real bug, 4 phantom

First time the suite has been run on Windows in a while. `venv\Scripts\python.exe -m pytest -q` from the repo root on `b2a678e`: **5 failed, 985 passed, 28 skipped, 8 xfailed** (mac skips 6, Windows 28 — the extra skips are the torch/MPS-gated ones).

**The one real bug, a Windows-only test defect (fixed on mirach):** `tests/mri/test_app_sync_edits.py::test_local_file_matching_incoming_is_not_a_conflict`. Its `_write()` helper opened files in text mode, so on Windows every `\n` became `\r\n` and the bytes on disk stopped matching `_sha(text)` — the file always read as locally modified, so the "already identical to incoming is not a conflict" assertion could never hold. Fixed with `newline=""`, which suppresses translation on every platform (Python writes `\n` literally when `newline` is `""` or `"\n"`). **The updater itself was correct** — worth stating plainly, because the failure looks like a conflict-gate bug and is not one. The file's other tests pass on Windows for the wrong reason: they *want* a SHA mismatch, and newline translation hands them one.

A sweep found 12 more text-mode writes in byte-sensitive test files (`tests/corpus/`, `tests/export/`, `tests/engine/`). All pass today — they write JSON/JSONL that is read back through text mode, which normalizes the translation away. Left alone deliberately rather than churned.

**The 4 phantom failures are the finding that matters: the updater never deletes.** `tests/mri/test_typing_samples.py` survived on fortis while the routes it tests were removed in `b2a678e`, so it fails 4 ways against a 404. A full tree diff (fortis vs `git ls-files`) shows fortis is otherwise a faithful copy — 250 files vs 245 tracked, and the 5 extras are exactly the files upstream deleted:

| stale on fortis | deleted upstream in |
| --- | --- |
| `veritate_mri/training/native_trainer.py` | `80501e4` |
| `veritate_mri/runtime/typing_samples.py` | `b2a678e` |
| `veritate_mri/training/sync/trainers_sync.py` | `b2a678e` |
| `tests/mri/test_typing_samples.py` | `b2a678e` |
| `tests/engine/test_v13_compat.py` | earlier |

Consequences already observed: a stale test file makes a clean tree report 4 failures, and two dead modules sit importable where nothing should be able to reach them. **This is the top platform-arch item.** The fix belongs in `_copy_incoming`: a path in the baseline, absent from `incoming`, inside a non-skipped top-level dir, is a file upstream deleted and should be removed locally — with the same conflict gate applied, so a user-modified file is never silently destroyed. Until that lands, every downstream box drifts further from the commit it claims to be on.

## the update that "kept failing" was never applied 2026-07-29

fortis reported `head_short: 80501e4, behind: 1, update_available: true` with `last_check_ok: true` and a recorded etag from a check at **21:18:08**, while `pulled_commit` still read `80501e4`. A **check** had run; a **pull** had not. `POST /app/update_pull` then applied it in one shot: `head_short: b2a678e, behind: 0, update_available: false`, `veritate_trainer.py` present at 42,590 bytes, and fortis's stale 4,700-byte `veritate_mri/data/trainer_sizes.json` (an older copy from `3259282`) correctly overwritten with the real 13,223-byte table.

Nothing was broken and the hash was never missing. But a UI that leaves `update_available: true` after the user believes they updated is the same class of problem as the counter that never cleared: **the dashboard must make "checked" versus "applied" impossible to confuse.** Worth fixing alongside the delete gap.

## updater — FIXED and verified in production 2026-07-29

The "behind" counter that never cleared is fixed and confirmed on fortis. `pull_update` applies a TARBALL, so `.git/HEAD` never advances and comparing against it left `behind` permanently stuck. `app_sync.py` now records `pulled_commit` (the real remote branch sha) on pull and `_compare_base()` prefers it, with `head_short` built from that sha instead of the ETag. Verified live: fortis went `behind: 7, head_short e68f62a` -> pull 596 files -> `behind: 0, head_short 80501e4, update_available: false`, cleared on the FIRST pull.

Two facts about the updater worth keeping:
- **`DEFAULT_SKIP_DIRS` resolves to `('.git', '.venv', '__pycache__', 'data', 'experiments', 'models', 'trainers', 'venv')`** — `PLUGINS_ROOT` basename is `trainers`. Matched against the FIRST path segment only, so models, checkpoints and corpus bins survive an update while `veritate_mri/data/trainer_sizes.json` correctly arrives WITH one. Updating a box mid-training does NOT endanger a checkpoint. Pinned by `tests/mri/test_updater_skip_depth.py`.
- **The updater never DELETES.** It only copies incoming files, so files removed upstream linger. `veritate_mri/training/native_trainer.py` is still on fortis after 80501e4 deleted it. Harmless here (nothing references it — `NATIVE_TRAINER_PATH` was repointed at `veritate_trainer.py` in the same commit) but it means stale modules accumulate.

## corpus

fortis holds ~100 GB of bins including **the_pile 50 GB** and **slimpajama627b 20.5 GB**; mirach has ~25 GB of prose. Corpus size, not GPU time, was the binding constraint before those landed.
