---
build: 9
date: 2026-06-29
title: Marketplace dataset downloads, corpus-library market corpora, live extension uninstall
---

## What changed

- **Extension dataset downloads work.** The market extension's data section fetches each
  dataset's single hosted archive (`.tar.gz`/`.zip`) and extracts the CSVs into place. The
  `stocks`, `forex`, `futures`, and `indices` datasets are hosted (Carpathian COS) and downloadable.
  `crypto` and `crypto_extra` stay **"coming soon"** until their large archives finish uploading.
- **Corpus library gained the market corpora.** `crypto` and `stocks` byte corpora are published as
  real `raw_bytes` entries hosted on COS (with checksums); the old hardcoded "coming soon"
  placeholders are gone. Install them from Settings -> Corpus library.
- **Uninstall is fixed and live.** Uninstalling an extension (including a built-in) now actually
  deactivates it via `extensions/disabled.json`, and a `before_request` gate makes install/uninstall
  take effect **without a dashboard restart**.

## What you must do

- Restart the dashboard **once** to load the new download and uninstall code, then hard-refresh the
  browser.
- After that, datasets download from the market extension's data section and corpora install from
  Settings -> Corpus library. Send the two remaining crypto dataset URLs to flip them from
  "coming soon" to downloadable.

## Versions

| component | version |
|---|---|
| build | 9 |
| engine | v1.3.0 |
| mri | v1.3.0 |
| format | v1.5.0 |
| trainers | v1.1.1 |
