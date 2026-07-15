# Corpus expansion handoff: grow the code corpus library

**You are Claude, in a checkout of the Veritate repo.** This is a self-contained brief for
expanding the code corpora (bigger tiers, more textbook data, new sources, new languages).
The pipeline, 12 shipped corpora, and all wiring landed 2026-07-15; this doc is how to extend
them without breaking determinism or the published library.

Read first (gating, every session): `CLAUDE.md`, `claude_preflight.md`,
`developer_documentation/agents/coding_roe.md`. Component reference:
`developer_documentation/corpus/code_corpus.md`. Sam runs all git; never stage or commit.

---

## What exists

One builder owns the pipeline: `veritate_mri/tools/build_code_corpus.py`
(tests: `tests/mri/test_build_code_corpus.py`). Two phases:

- `stage` pulls pinned sources into `~/.cache/veritate/code_corpus_staging/` (network).
- `build` assembles byte-level bins from that cache (offline, deterministic from a seed).

| family | source | notes |
|---|---|---|
| py, js | HuggingFaceTB/stack-edu metadata + Software Heritage S3 blobs (anonymous) | edu-classifier scored; revision pinned in `STACK_EDU_REVISION` |
| html, css | curated permissive-license repos, tarballs by pinned tag/commit (`REPOS`) | wpt capped via `REPO_BYTE_CAPS` |
| qa | donfu/oa-stackexchange, stackoverflow rows, coding tags, votes >= 5 | ChatML framed |
| textbook | Claude-authored jsonl shards in `trainers/corpus/_code_textbook_cache/` | 2,210 docs, all syntax-gated; py/js executed at authoring time |

Shipped (all live in the dashboard corpus library, Coding section): `py_code_100mb`,
`py_code_1gb`, `js_code_100mb`, `js_code_1gb`, `html_code_50mb`, `css_code_3mb`,
`code_qa_100mb`, `mixed_code_{raw,files,edu,qa}_200mb`, `code_textbook_v1`.
Zips + urls: `Mirach-Corpuses/manifest.md` (iCloud). Catalog:
`veritate_mri/training/sync/corpus_catalog.json`. Settings-page section mapping:
`CORPUS_STEM_CATEGORY` in `veritate_mri/web/index.js`.

## Hard rules (do not break)

1. **Never rebuild or overwrite a published stem.** Catalog sha256s verify the extracted
   bins; a rebuild that changes bytes breaks every future install. New content = new stem
   (`code_textbook_v2`, `py_code_5gb`), never a silent replacement.
2. **Determinism is the contract.** Pin every new source (HF revision, tag or commit sha);
   `stage` records revisions and tarball sha256s to `sources.json` in the staging dir.
   Same staging files + same seed = identical bytes. `Date`/randomness never enter builds
   except through the seeded PRNG.
3. **Corpus bins only in `trainers/corpus/` or the staging cache.** No manifests or code in
   `trainers/corpus/` (preflight rule 37); the authored-source caches (`_pg_cache`,
   `_code_textbook_cache`) are the sanctioned exceptions.
4. **Training on these corpora**: launch via dashboard `POST /trainers/run` with
   `model_type: "code"` (preflight 24a; omitting it mis-gates hooks and evals).
5. Multi-stem training mixes: `--corpus a:0.5,b:0.3,c:0.2` (weights normalize);
   `resolve_and_weight` returns weight-descending, so val loss tracks the heaviest stem.

## Playbook A: bigger py/js tiers (e.g. 5-10 GB for the GPU farm)

1. Restage deeper: `python veritate_mri/tools/build_code_corpus.py stage --family py --target-mb 12000`
   (same for js). Staging keeps score >= 3 (`STAGE_MIN_SCORE`); the stream is NOT sorted by
   score. Measured 2026-07-15: score >= 4 is ~13% of staged bytes, so an ultra-clean tier
   needs roughly 8x the staged volume of its target size.
2. Throughput reality: per-blob SWH fetches, ~15 MB/min with 64 workers and per-thread
   HTTP sessions (already implemented). 12 GB ~= 13 h. Do not run py and js staging
   concurrently with another bandwidth-heavy job; contention measured a 20x slowdown.
3. Restaging re-downloads from row 0 of the pinned revision: the old staging file is
   truncated. That is fine (same order, more of it), but bins built from the OLD staging
   file cannot be reproduced afterward unless it is backed up first. Back up
   `py.jsonl`/`js.jsonl` before restaging if any unpublished bin depends on them.
4. Build new stems from the deeper cache (`--min-score 3` for volume tiers, `4` for
   ultra-clean), then ship (Playbook D).

## Playbook B: extend the Claude-authored textbook corpus

1. Dispatch an authoring fleet (Opus agents, background). Each agent writes ONE new jsonl
   shard to `trainers/corpus/_code_textbook_cache/<lang>_NN_<theme>.jsonl` with lines
   `{"text": ..., "lang": "py|js|html|css", "topic": ...}`. Do not touch existing shards.
2. Non-negotiable authoring contract (this is what made v1 clean): each text is one
   self-contained file; concept prose in a docstring or block comment; complete runnable
   example; exercises with worked solutions; NO markdown fences or placeholders; python
   must pass `ast.parse`, js must pass `node --check`; agents must EXECUTE py/js documents
   and see every assert pass before appending; 600-3500 bytes per doc; varied domains.
   Themes not yet covered by v1: typing/generics depth, asyncio, sqlite, http clients,
   testing frameworks, regex golf, canvas/webgl, css houdini, accessibility deep-dives.
3. Rebuild as the NEXT version (rule 1): `build --family textbook --target-mb 50
   --out-train code_textbook_v2_train.bin ...` (target above pool takes everything).
4. Ship as `code_textbook_v2` (Playbook D). v1 stays published untouched.

## Playbook C: new sources or languages

1. Probe access first (anonymous HF: gated sets like the-stack fail; stack-edu, CSN,
   codeparrot-clean are open). Pin the revision before any staging.
2. New language from stack-edu: add the config to `STACK_EDU_CONFIGS`, add size caps and
   an `accept_<lang>` filter (syntax gate where a stdlib/tool exists, heuristics otherwise),
   wire into `ACCEPT`, extend the unit tests (every new filter gets an accept + reject test).
3. New curated repos for html/css volume: append to `REPOS` with a verified tag
   (`curl -I https://codeload.github.com/<repo>/tar.gz/refs/tags/<tag>` first), restage
   `web`. Remember the css lesson: the curated pool yields ~3 MB of genuinely distinct
   handwritten css; near-dup framework builds add nothing. Expand sources, not padding.
4. Keep one module owning the concern: new capability extends `build_code_corpus.py`,
   never a sibling script.

## Playbook D: shipping checklist (every new stem)

1. Build bins to the scratchpad; name the stem by ACTUAL yield (a pool underrun prints a
   warning; rename the tier rather than shipping a mislabeled size).
2. `zip -j -X` the two bins into the right `Mirach-Corpuses/<family>/` folder.
3. Catalog entry in `corpus_catalog.json`: stem, label, description, `format: zip_bundle`,
   sha256 + size of BOTH extracted bins, `trained_modes`, recommended param band,
   placeholder url `https://api.carpathian.ai/cos/PLACEHOLDER/<stem>.zip`,
   `"coming_soon": true`.
4. Map the stem in `CORPUS_STEM_CATEGORY` (`veritate_mri/web/index.js`) so it lands in the
   Coding section; `node --check` the file after editing.
5. Add the manifest row + rebuild command to `Mirach-Corpuses/manifest.md`.
6. Sam uploads the zip and provides the COS url. Before wiring it in, verify with a HEAD
   request: `content-length` must equal the local zip bytes and `content-disposition`
   must name the right file (this caught label mixups on 2026-07-15).
7. Set `train_url`, drop `coming_soon`, replace the manifest placeholder, then verify
   end-to-end: download the url, unzip, sha256 both bins against the catalog.
8. Update `developer_documentation/corpus/code_corpus.md` in the same change (preflight 25c).

## Open research thread

IDEA 4 in `ideas.md`: the mixed_code corpus-style ablation (raw vs files vs edu vs qa,
200M recipe, kill lines pre-registered). If a style separates, the next expansion should
scale THAT style's pipeline; re-validate at 1-3B before any farm-scale commitment (a 200M
winner picks the mix family, not the final ratios).
