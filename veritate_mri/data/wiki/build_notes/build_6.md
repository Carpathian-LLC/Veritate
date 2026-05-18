---
title: "Build 6: updater rework, corpus library, model recovery wiki"
date: 2026-05-09
tags: [build, updater, corpus, datasets]
summary: In-app updater no longer blocks on diverging branches or dirty trees; tracked source is force-reset to upstream while gitignored user data (data/, models/, plugins/) is preserved. Auto soft-reload on update is now the default. New Corpus library in Settings ships an apt-style installer for training data, with a local catalog spanning Tiny Shakespeare through RedPajama-V2 and one-click install for HuggingFace-hosted corpora. Adds datasets and pyarrow as required Python deps. New wiki entry documents how to recover models that vanish after engine updates.

---

## versions

- build: 6
- engine: v2.2.0
- mri: v0.2.0
- format: v0.3.0
- plugins: v0.1.1

## what changed

### updater

- `Update` button now does `git fetch` + `git reset --hard origin/<branch>` for app, models, and plugins syncs. Diverging branches and dirty working trees no longer block updates.
- Tracked source is overwritten to match upstream. User data in `data/`, `models/`, and `plugins/` is gitignored and untouched.
- "Auto soft-reload after update" toggle defaults to **on**: pulling a new commit immediately fires `lifecycle.restart` (the yellow "reload python" path), so new code is loaded without manual intervention.
- If your repo is wedged from a previously failed update, delete and re-clone the Veritate repository. Your `data/`, `models/`, and `plugins/` folders carry over.

### corpus library

- New **Corpus library** card in Settings → Update section. Apt-style installer that streams training data directly into `plugins/corpus/<stem>_train.bin` and `<stem>_val.bin`.
- Local catalog ships in [veritate_mri/sync/corpus_catalog.json](../../sync/corpus_catalog.json), with 9 entries spanning every model size band:

| Stem | Source | Size cap | Best for |
|---|---|---|---|
| `shakespeare` | direct URL (Karpathy char-rnn) | 1 MB | <5M params |
| `enwik8` | direct zip (mattmahoney.net) | 95 MB | 5M-50M |
| `wikitext103` | HF `Salesforce/wikitext` | 500 MB | 30M-200M |
| `tinystories` | HF `roneneldan/TinyStories` | 1 GB | 1M-50M |
| `pg19` | HF `deepmind/pg19` | 9.7 GB | 50M+ |
| `openwebtext10g` | HF `Skylion007/openwebtext` | 10 GB | 100M-500M |
| `the_pile` | HF `monology/pile-uncopyrighted` | 50 GB | 500M-3B |
| `slimpajama627b` | HF `cerebras/SlimPajama-627B` | 100 GB | 3B-10B |
| `redpajama_v2` | HF `togethercomputer/RedPajama-Data-V2` | 200 GB | 10B+ |

- Three install formats supported: `raw_bytes` (direct URL streamed as uint8), `raw_bytes_zip` (URL → unzip largest member → write), `hf_dataset` (lazy-import `datasets`, stream rows via `load_dataset(..., streaming=True)`, encode text column as UTF-8 with `\n\n` row separators).
- Catalog layering: local file ← optional remote `corpus_catalog_url` (set via the `URL` button) ← per-machine `corpus_user_sources` (added via "+ custom"). Later layers override by stem.
- Disk-space precheck refuses install when free space < 1.2 × expected size.
- Browser confirms before any install over 10 GB; backend additionally requires `confirm_large=true` in the install body.
- `val_split_ratio` carves the trailing fraction of the train file off as `val.bin` when no separate validation source is given.
- Training-tab corpus selector now shows a yellow warning when the chosen model size doesn't match the corpus's `recommended_min_params` / `recommended_max_params` range.

### model recovery wiki

- New entry [recovering_missing_models.md](../documentation/recovering_missing_models.md) documents how to restore models that disappear from the dashboard after engine updates (typically caused by a stale `.bin` format).

## what the user has to do

### required: install new deps

The corpus library uses HuggingFace `datasets` and `pyarrow` for everything except `shakespeare` and `enwik8`. Install them once:

```sh
pip install -r requirements.txt
```

Without this, install will fail with `the 'datasets' package is not installed` for any HF-sourced corpus. The dashboard surfaces a warning at the top of the Corpus library card when these deps are missing.

### required: reload python after updating

Click the yellow **reload python** button once after the update lands so the new updater logic and corpus_sync module are loaded.

### optional: install corpora

1. Open **Settings → Corpus library**.
2. Click **install** next to any corpus. Anything over 10 GB prompts a confirm dialog.
3. Watch the spinner + progress bar live. When it lands, the green `installed` tag appears on the row and the corpus shows in the Training tab dropdown.

Recommended starter: install `tinystories` (~1 GB) first to confirm the HF pipeline works on your machine before committing to multi-GB downloads.

### optional: add custom sources

Click **+ custom** in the Corpus library card to add a corpus by direct URL. The modal accepts stem, label, train URL, optional val URL, and optional sha256 digests for verification. Custom sources persist in `data/mri_settings.json` under `corpus_user_sources` and survive the next sync.

## known limitations

- `deepmind/pg19` and `Skylion007/openwebtext` were historically distributed as HF dataset scripts. HuggingFace deprecated dataset scripts in `datasets` ≥ 3.0; if these specific corpora fail to install with `Dataset scripts are no longer supported`, the catalog entry's `format` and source fields can be swapped to `raw_bytes_zip` or a parquet-backed mirror.
- Default caps on the very-large corpora (50 GB / 100 GB / 200 GB) are deliberately conservative. Raise `max_bytes_train` in [corpus_catalog.json](../../sync/corpus_catalog.json) if you actually need the full 825 GB / 627B tokens / 30T tokens.
- HF datasets occasionally gate behind a license click. If install fails with an authentication error, run `huggingface-cli login` once and retry.
