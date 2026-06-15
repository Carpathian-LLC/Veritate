# Corpus library

## What it is

Apt-style installer for training corpora, surfaced as the "Corpus library" card on the Settings tab. Resolves a catalog of corpora and installs each into `trainers/corpus/<stem>_train.bin` / `<stem>_val.bin`, where the Training tab corpus picker finds them.

## How it works

Module: [corpus_sync.py](../../../veritate_mri/training/sync/corpus_sync.py). Routes: [corpus_routes.py](../../../veritate_mri/routes/corpus_routes.py) (`/corpus/library/*`).

The catalog merges three layers by stem, later wins (`catalog()`, [corpus_sync.py:224](../../../veritate_mri/training/sync/corpus_sync.py#L224)):

1. local catalog file [corpus_catalog.json](../../../veritate_mri/training/sync/corpus_catalog.json), shipped in the repo;
2. optional remote catalog fetched from `corpus_catalog_url` in settings;
3. `corpus_user_sources` from settings (per-machine custom entries).

Four install formats (`install()`, [corpus_sync.py:736](../../../veritate_mri/training/sync/corpus_sync.py#L736)):

- `raw_bytes` — HTTP download of a uint8 byte stream, optional sha256 verify, atomic `.part` → rename.
- `raw_bytes_zip` — same, then the largest zip member replaces the download (enwik8).
- `hf_dataset` — stream rows from a HuggingFace dataset, UTF-8 encode the text column ([corpus_sync.py:530](../../../veritate_mri/training/sync/corpus_sync.py#L530)). Requires the `datasets` package; `hf_probe()` and `/corpus/library/install_deps` handle the missing-dep flow.
- `native` — the bins ship inside the repo at `veritate_mri/data/corpus/<stem>_{train,val}.bin`; install copies them into `trainers/corpus/` (`_install_native()`, [corpus_sync.py:675](../../../veritate_mri/training/sync/corpus_sync.py#L675)). No network, no sha. `catalog()` reads `size_train`/`size_val` from the bundled files and sets `native_available`. Shipped native corpora: `mcp_docs` (modelcontextprotocol.io documentation, autocomplete framing per [framing.md](../../corpus/framing.md)).

`val_split_ratio` carves the tail of a downloaded train file into val when no val source exists. Installs over 10 GB need `confirm_large=true`; a disk-space precheck refuses installs that would fill the disk.

`uninstall()` deletes the two `.bin` files from `trainers/corpus/` only — for native corpora the repo copy stays, so reinstall always works.

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py): `CORPUS_ROOT`, `NATIVE_CORPUS_ROOT`, suffix constants.
- [runtime/settings.py](../../../veritate_mri/runtime/settings.py): catalog URL + user sources.
- Frontend: `_corpusRenderCatalog` / `_corpusInstallTrigger` in [index.js](../../../veritate_mri/web/index.js) (Settings card `#corpusLibraryRow` and modal `#corpusLibraryModal`).
- Tests: [tests/mri/test_corpus_native.py](../../../tests/mri/test_corpus_native.py), [tests/selftest/checks/check_corpus_catalog.py](../../../tests/selftest/checks/check_corpus_catalog.py).

## Pitfalls

- `trainers/` is gitignored, so installed corpora never show in git status; the shipped native bins live under `veritate_mri/data/corpus/` precisely so they are versioned.
- Native catalog entries omit `size_train`/`size_val`; the values come from the bundled files at catalog time. Declaring them in JSON would drift after a corpus rebuild.
- Remote-catalog fetch failures are non-fatal; `catalog_status` carries the error and the UI falls back to the local catalog.
- `_PROGRESS` is in-process state; the dashboard polls `catalog()` during installs to render progress bars.
