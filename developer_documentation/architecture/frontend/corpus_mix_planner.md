# Corpus mix planner panel

## What it is

A panel inside the corpus library modal that turns several picked corpora into one weighted multicorpus spec (`stem:0.55,stem:0.45`) and hands that spec to the Training tab. It shows the full plan, its warnings, and the download/disk/time consequences before anything is committed. Backend: [../backend/mix_planner.md](../backend/mix_planner.md).

## How it works

Markup in [index.html](../../../veritate_mri/web/index.html): a `<details id="corpusMixPanel">` inside `#corpusLibraryModal`, above `#corpusList`. Styles are the `.mix-*` block in [index.css](../../../veritate_mri/web/index.css). All logic is the `// ---- Corpus mix planner ----` section of [index.js:12791](../../../veritate_mri/web/index.js#L12791), state in `corpusMixState`.

- **Selection.** Every catalog row renders a `.mix-check` checkbox (`_corpusRowHtml`, [index.js:12497](../../../veritate_mri/web/index.js#L12497)); `coming_soon` rows render it disabled with the reason in the title. `_corpusMixToggle` maintains `corpusMixState.picked` and drops any existing plan, so a stale plan is never shown against changed picks. `_corpusMixPickedHtml` restates each pick's label, stem, size, topic, family, and on-disk state, plus the concrete next action when it is not installed.
- **Controls.** Intent profile, target size (number + MB/GB), epoch cap, model params, and download speed, each with a one-line plain-language explanation. Profiles are server data, never a JS list: `_corpusMixLoadProfiles` reads `GET /corpus/mix/profiles`, and when that route is absent falls back to the saved `corpus_mix_default_profile` from `GET /settings` plus an `other...` free-text entry, saying so in the help line. `_corpusMixOnOpen` seeds the epoch cap from `settingsState.current.corpus_mix_max_epochs`; the model dropdown is built from every `manifest.sizes[*].params` in `GET /trainers`.
- **Plan preview.** `_corpusMixPlanRequest` POSTs `{stems, target_bytes, profile, max_epochs, model_params}` to `/corpus/mix/plan`. `_corpusMixPlanHtml` renders one row per source (label, topic, weight, bytes drawn, bytes available, epochs), tags any row whose `epochs >= cap`, prints every `warnings` entry verbatim, echoes `inputs` so the profile the planner actually used is visible, and shows `bytes_planned` against the requested target. A `404` is reported as "this server has no mix planner yet" rather than a JSON parse error.
- **Consequences.** `_corpusMixConsequenceHtml` states how many picks are already local, which need downloading and how many bytes, the disk that implies, and the time at the current MB/s. The rate control is measured, not assumed: `_corpusMixObserveRate` derives MB/s from a live install's `progress.bytes / (now - started_at)` and persists it in `localStorage`.
- **Download progress.** `_corpusMixDownloadMissing` runs the missing stems through `_corpusInstallTrigger` one at a time (that function returns a promise for this). `_corpusMixProgressHtml` shows, per source, bytes written vs total, percent, and elapsed, plus "source N of M". Data comes from the catalog `progress` block, repainted by the existing 2s install poll through `_corpusRenderCatalog`.
- **Handoff.** `_corpusMixAccept` stores the spec under `vt:corpus:mix:spec`, shows the copyable spec and the exact next step, and calls `_trApplyPendingMix` ([index.js:8741](../../../veritate_mri/web/index.js#L8741)), which writes the spec into the trainer form's corpus field, ticks the matching picker boxes, and consumes the key. `_trApplyDefaults` calls it after `_trRestoreFormState`, so a mix accepted while the Training tab was never opened still lands.

## Dependencies

- `POST /corpus/mix/plan` — weights, per-source arithmetic, warnings ([corpus_routes.py](../../../veritate_mri/routes/corpus_routes.py)).
- `GET /corpus/mix/profiles` — intent profile list. Optional; the panel degrades with an explanation when missing.
- `GET /corpus/library/catalog` — labels, sizes, family/topic, installed state, live `progress`.
- `POST /corpus/library/install` — via `_corpusInstallTrigger`.
- `GET /settings` — `corpus_mix_default_profile`, `corpus_mix_max_epochs`.
- `GET /trainers` — `manifest.sizes[*].params` for the model dropdown.

## Pitfalls

- Ticking a box in the Training tab's corpus picker rebuilds that field from the checked stems joined with `+`, which discards the weights. Re-accept the mix to restore them.
- The profile list is fetched once per page load; a transient failure leaves only the saved default plus `other...` until reload.
- The rate control overwrites itself from any live download while the field is not focused. That is deliberate (measured beats typed) but it will replace a hand-entered number.
- `_corpusMixMissing` excludes `coming_soon` picks because they cannot be downloaded; they are called out separately in the consequences block and still block a launch.
- Element ids and the no-hardcoded-profiles rule are guarded by [tests/mri/test_corpus_catalog_shape.py](../../../tests/mri/test_corpus_catalog_shape.py).
