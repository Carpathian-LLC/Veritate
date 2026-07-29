# handoff

Rolling state summary. Newest facts win. Keep it short.

## boxes

| box | hardware | repo | dashboard |
| --- | --- | --- | --- |
| mirach | M3 Ultra, 256 GB unified | `/Users/mirach-00-usc1/Development/Veritate` | :8001 |
| fortis | Windows, RTX 5070 **12 GB VRAM** | `C:\GitHub\Veritate` (NOT `C:\Users\malka\Veritate`, that is a stale decoy) | :8001, reachable via `ssh fortis` |

fortis is reached with `ssh fortis`; it is a Windows box, so use `powershell -NoProfile -EncodedCommand <base64 utf-16le>` to avoid quoting hell. Its dashboard must be launched as an INTERACTIVE scheduled task (`schtasks /it`) or the browser window will not appear on the physical desktop and the process dies when the SSH session closes.

## live as of 2026-07-28

- **fortis: `core_50m`** — the one model. **50.64M params** as built by `native_trainer.py` (mirach's hybrid path would build 74.1M from the same preset — different trainers, different models). corpus `the_pile` alone, seq 2048, batch 2, muon, bf16, wsd. **11,250,585 steps = 42.9 GB = 200 tokens/param, ~7.5 days** at the measured 70,919 tok/s. No val curve (the_pile ships no val bin). Context is 2 KB.
  - **`--n_chunks` and `state_carry` DO NOTHING in `native_trainer.py`** — declared in argparse, never read. A first launch was budgeted 16x short because of it. Verify every run with `tok_per_s / step_rate` against the intended `batch x seq x n_chunks` before trusting the config.
  - The bench is unreliable in BOTH directions: it over-predicted 53% on mirach (AdamW probe vs Muon run) and under-predicted 3.2x here (22,367 predicted vs 70,919 measured). Live-log arithmetic is the only trustworthy source.
- **fortis: `core_200m`** — STOPPED at step 86,975 of 164,388. Resumable from ckpt 86,500. Headed for 4.4 tokens/param, i.e. the same dead end as wren. Kept on the user's instruction.
- **mirach: `wren_sft`** — capability SFT finishing at step 62,425. Eval chain `after_sft.sh` fires on exit: 280-item form IFEval + retention.

Deleted this session: `chat_200m`, `chatty_200m`, `wren` (both boxes), `chin200m`. One model per box is the standing goal — no experiment forks, no A/B arms, no version suffixes.

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

- **`trainer_sizes.json` is NOT in the canonical trainers repo.** Canonical has 4 files. mirach's copy at `trainers/common/trainer_sizes.json` is untracked and local-only — one sync from being wiped, and it is why fortis had no sizes until it was copied across by hand. Commit it upstream.
- **The app updater's "behind" counter can never clear.** `pull_update` applies a TARBALL; `check_update` measures `behind` via the GitHub compare API against `.git/HEAD`, which a tarball extract never advances. Verified: a pull copied 584 files, `ok: true`, and `behind` stayed 6. Also `head_short` is built from the ETag rather than a commit sha, so the panel shows `"613e9c` with a stray quote. Fix belongs in `veritate_mri/training/sync/app_sync.py` on mirach.
- 19 stale `veritate_*` per-size trainer dirs were deleted from fortis; `trainers/` there is now `common/` + `corpus/` only, matching canonical.

## corpus

fortis holds ~100 GB of bins including **the_pile 50 GB** and **slimpajama627b 20.5 GB**; mirach has ~25 GB of prose. Corpus size, not GPU time, was the binding constraint before those landed.
