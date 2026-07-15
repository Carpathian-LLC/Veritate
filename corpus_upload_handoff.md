# corpus upload handoff

State as of 2026-07-14: seven corpora built + zipped, one trading dataset
zipped, all platform wiring shipped and tested. Everything below is
operator-only (uploads, one JSON edit per release, restart, git).
Full technical detail: worklog.md 2026-07-14 section and
developer_documentation/corpus/library_ladder.md.

## what is staged

`~/Library/Mobile Documents/com~apple~CloudDocs/Mirach-Corpuses/`

| zip | size | catalog entry to release |
|---|---|---|
| chat/chat_500mb.zip | 178 MB | corpus_catalog.json `chat_500mb` |
| chat/chat_5gb.zip | 1.74 GB | corpus_catalog.json `chat_5gb` |
| agent/agent_150mb.zip | 17 MB | corpus_catalog.json `agent_150mb` |
| agent/agent_1500mb.zip | 169 MB | corpus_catalog.json `agent_1500mb` |
| mcp/mcp_15mb.zip | 2 MB | corpus_catalog.json `mcp_15mb` |
| mcp/mcp_150mb.zip | 18 MB | corpus_catalog.json `mcp_150mb` |
| mcp/mcp_1500mb.zip | 184 MB | corpus_catalog.json `mcp_1500mb` |
| trading_datasets/crypto.zip | 6.8 GB | trading data_catalog.json `crypto` |

`manifest.md` in that folder carries the sha256s and per-zip placeholder URLs.
All bins are also already installed in `trainers/corpus/` on this box, so
local training does not wait on any of this.

## next steps

1. **Upload the 7 corpus zips to Carpathian COS** (contents: `<stem>_train.bin`
   + `<stem>_val.bin` at zip top level; folder prefixes also fine — the
   extractor matches by basename).
2. **Release each corpus** in
   `veritate_mri/training/sync/corpus_catalog.json`: replace the
   `https://api.carpathian.ai/cos/PLACEHOLDER/<stem>.zip` train_url with the
   real COS link and delete the `"coming_soon": true` line. One JSON edit per
   corpus; no JS or Python changes. sha256/size fields are already correct
   (they verify the extracted bins, not the zip, so COS re-zipping is safe).
3. **Upload trading_datasets/crypto.zip to COS**, then set the `url` field of
   the `crypto` entry in `extensions/canonical/trading/data_catalog.json`
   (currently `null` = coming soon). No other change; the extension downloader
   already unzips and deletes the archive.
4. **Restart the MRI server** when convenient — the running process predates
   the corpus_sync `zip_bundle` code. Do not restart while a dashboard-launched
   run is active.
5. **Verify one release end-to-end** (suggest `mcp_15mb`, smallest): uninstall
   it in the dashboard corpus library, reinstall from COS, confirm the sha256
   check passes and no `<stem>.zip` is left behind in `trainers/corpus/`.
6. **Git**: new files `veritate_mri/tools/build_agent_corpus.py`,
   `veritate_mri/tools/build_mcp_corpus.py`,
   `tests/mri/test_corpus_library_zip.py`,
   `developer_documentation/corpus/library_ladder.md`, this file; modified
   `corpus_sync.py`, `corpus_catalog.json`, `index.js`,
   `architecture/backend/corpus_library.md`,
   `architecture/frontend/settings_tab.md`, `worklog.md`.

## dropped

- **crypto_extra** (41 GB, 451 extra Binance pairs): removed from the trading
  data catalog entirely — no local data existed on this box and the dataset is
  not worth hosting. If it ever comes back, add a fresh `data_catalog.json`
  entry and follow step 3. Its empty source dir
  (`extensions/installed/trading/data/extension_data/crypto_extra/`) can be
  deleted whenever.

## rebuild notes

Builders are deterministic — same seed, same bytes, same sha256. Chat tiers
additionally require the restored Gutenberg cache at
`trainers/corpus/_pg_cache/` (30 texts; fetch list in the staging manifest).
Exact rebuild commands are in `manifest.md`. If a corpus is ever rebuilt with
different content, rerun the sha256s into corpus_catalog.json before release.
