# Training tab

## What it is

Live + historical view of training runs. Shows loss curves, learning rate, throughput, grad norm, plateau health, latest stats, plus a real-time SSE feed of per-step brain frames.

## How it works

Markup at [index.html:901–1043](../../../veritate_mri/web/index.html#L901).

- **Auto tune** (`#trainAutoTuneBtn`, next to the memory estimate) opens `#autoTuneModal`: a measured benchmark that runs the selected trainer's `--bench` mode on throwaway weights, streams its narration from the log-ring SSE, and writes the measured batch/lr/cadence into the manifest and form. The modal has no trainer picker; it uses the Training tab's selected trainer. Visible with Advanced consent + detected specs + a `"bench": true` trainer. The result reports the memory strategy (`#autoTuneTier`): a size that only fits by paging the optimizer to NVMe ([paged_optimizer](../../platform/paged_optimizer.md)) is labelled and its tok/s flagged disk-bound; a size whose weights+grads exceed the machine budget renders the infeasible panel (`#autoTuneInfeasible`) with the floor breakdown and remedies instead of a recommendation. `use_act_ckpt` is added to the applied args when the plan checkpoints. `_autoTuneRecommend` also sets `precision=fp32` on non-bf16 machines (`caps.can_use_cuda || caps.is_apple_silicon` false) and `qat_enabled=false` on CPU. Applying (`_autoTuneApply`) POSTs to `/trainers/tune_defaults`, which saves to the machine-local tuning store, not the manifest (see [../backend/trainer_tuning.md](../backend/trainer_tuning.md)). See [../../platform/bench.md](../../platform/bench.md).
- **Memory estimator** (`_trEstimateMemory`) reads `_sysSpecsCache.capabilities`: activation bytes-per-element are 2 for bf16 only on bf16-capable hardware (`can_use_cuda` or `is_apple_silicon`), else 4; with no CUDA flash kernel (CPU/MPS) it adds the quadratic `[batch, heads, seq, seq]` attention-score term held for backward; a QAT activation multiplier (`QAT_ACT_MULT`) applies when `qat_enabled` is on.
- **Corpus meta panel** (`_trUpdateCorpusMeta`): a `+` corpus mix renders as per-member blocks (`_trCorpusMixHtml`), each stem showing its size-proportional weight, bytes, and fingerprint, instead of 404ing on the joined stem.
- **Learn more links**: each meaningful setting listed in `WIKI_SETTING_PAGES` renders a `learn more` link beside its label, opening `/wiki/settings/<slug>/page` in a new tab (see [wiki_tab.md](wiki_tab.md)).
- **Form de-dup**: `use_act_ckpt`, `use_8bit_adam`, `l1_lambda`, `qat_enabled`, and `freeze_base` are NOT schema rows; the "Additional options" cards (core_plugins.py registry) own them (`_corePluginsArgs()` merges card args over form args at submit, and `_trEstimateMemory` reads the cards). Don't re-add them to `TRAINER_SCHEMA`.
- **Additional options cards** (`#trainCoreTrainersGrid`, registry at [core_plugins.py](../../../veritate_core/core_plugins.py)): compact rectangles, one pick per group (multi-member groups are mutually exclusive), descriptions collapsed behind the per-card &#9662; toggle.
- **Corpus picker**: `type: "corpus"` renders a collapsed `<details>` checkbox list; the hidden `data-arg` input carries the joined `"a+b"` stem spec (this also fixed the old multi-select submitting only the first stem via `_trCollectArgs`). When no corpus is present the picker renders the `.train-no-corpus-error` empty state, which carries a `[data-open-corpus-lib]` link wired in `_trWireArgListeners` to `_corpusOpenLibraryModal()` (the settings corpus-library downloader) so a user can install training data without leaving the form.
- **Conditional knobs**: `ARG_VISIBLE_WHEN` hides knobs that don't apply (wsd_* unless `lr_schedule=wsd`; `min_lr`/`warmup_steps` hidden on `constant`; `quant_mode` only with QAT on; `mtp_aux_weight` only when `n_predict>1`). `_trUpdateKnobVisibility()` runs on render and on every field change.
- **Simple mode + recipes**: by default the form renders only `SIMPLE_FIELDS` (name, corpus, size, precision, description, model_type, recipe, resume, total_steps); the "show all settings" toggle (`#trShowAllKnobs`, persisted in `localStorage.tr_show_all`) reveals the full grid. The `recipe` field is UI-only (`uiOnly: true`, skipped by `_trCollectArgs`); picking one applies a named knob combination from `TRAIN_RECIPES` (`_trApplyRecipe`), each combination backed by a measured A/B in `successes.md`. `optimizer` (adamw|muon) and `trunk` (dense|patched|recurrent|hybrid|looped|memory) are real schema rows riding the reserved-flag path in `vanilla_trainer.py`. When a research verdict lands, update BOTH `TRAIN_RECIPES` and the help text of `optimizer`/`trunk`.
- **Model size is the trainer**: plugin trainers carry a single-entry `sizes` table, so the size row is filtered out of the form (`_trArgsForPlugin`); launch omits `--size` and the trainer's manifest default applies server-side. `_trResolvedSize()` supplies the trainer's size to the composed-name preview, the memory estimator, and the corpus-size warning. The dropdown only renders for multi-size manifests (the native trainer).
- **Model type gates evaluations** (`model_type` schema field: `language` default, `code`, `statistical`, `other`): language probes (register fluency, reading comprehension, math, grammar, reasoning, concepts, writing health) and the deep-eval panel are meaningless for a non-text model. The selector flows to `save()` so those probes are never computed for a non-language model, and `applyEvalGate(refs, config)` (called from `loadClassroomFor` once `config.training_args.model_type` is known) hides the language panels (the `refs` language IDs plus any `[data-eval="language"]` panel) when the selected run is not a language model. Because trainers `parse_known_args()` and drop the unknown `--model_type` flag, the value rides to `save()` via the `VERITATE_MODEL_TYPE` env set by `trainer_runner`, not the CLI. See [save.md](../backend/save.md).

- Run picker at `#runPicker` populated by `loadRunsList()` ([index.js:3418](../../../veritate_mri/web/index.js#L3418)) which fetches `/runs`.
- `loadTrainCsv()` ([index.js:3485](../../../veritate_mri/web/index.js#L3485)) fetches the selected run's `train.csv` and parses train/val rows.
- Charts: `cLossT` (loss), `cLrT` (lr schedule), `cTpsT` (throughput), `cGnT` (grad norm), plus confidence-evolution and reading-grade panels.
- `#trainPlateau` div displays one of six health states from the plateau detector (IMPROVING, PLATEAU, REGRESSING, SLOWING, BOUNCING, WARMING).
- `#trainLatest` shows the last row's metrics.
- "ask ai" buttons (`#askAiRecentTrain`, `#askAiLossCurve`, `#askAiTrainHealth`) post the selected run name plus cached chart data to `POST /ai/ask` ([ai_assist.py](../../../veritate_mri/runtime/ai_assist.py)) and render the answer in the AI modal. With no run selected they short-circuit to `window.ai_fail()` (modal error "no training run selected"), never hitting the backend.

Polling starts when the tab activates ([index.js:2122–2123](../../../veritate_mri/web/index.js#L2122)):

- `startTrainPolling()` ([index.js:6237](../../../veritate_mri/web/index.js#L6237)) — three intervals (CSV 5s, runs 30s, classroom 30s).
- `trainStreamStart()` ([index.js:11447](../../../veritate_mri/web/index.js#L11447)) — opens `/train_stream` SSE.

Polling stops on tab switch.

### Flow actions, persistence, and stop

- The action picker (`#trainFlowModal`) sets `trainState.flow` via `flowPick()` (`index.js`). The
  selected flow is persisted to `localStorage["vt:training:flow"]` and restored on load so a reload
  lands on the same action. Valid flows: `scratch`, `continue`, `rag`, `synth`, `author`, `export`.
- Per-flow job control goes through one job registry (`TRAIN_FLOWS` → `TRAIN_JOB`/`SYNTH_JOB`/
  `RAG_JOB`/`AUTHOR_JOB`), each exposing `stop()`. Every stop button routes through `confirmDialog()`
  (the shared `#confirmModal`) before calling its endpoint: training `POST /trainers/stop`, synth and
  author `POST /teacher/synth/stop` (cooperative), rag `POST /rag/stop`.
- Per-action layout: the metrics/charts block is wrapped in `#trainMetricsSection` and shown by
  `_trToggleMetrics(flow)` only for the training flows (`scratch`/`continue`/`rag`); it is hidden
  when no action is picked yet and for `synth`/`export`. The synth panel shows live teacher output instead (`#synthLiveWrap`
  / `#synthLiveOutput`), polled from `GET /teacher/synth/samples` inside `_synthPollOnce`.
- Synth seed list (`#synthSeedList`, rendered by `_synthRenderSeeds` in `index.js`): catalog entries
  carrying a `group` field render nested under a group header row (one per language); entries without
  it render flat. Children sort by `tier` (`SEED_TIER_ORDER`: easy, basic, advanced) and show the tier
  as their label. The group header checkbox (`.synth-group-cb`) toggles all children and goes
  indeterminate on partial selection; only child checkboxes (`.synth-seed-cb`) feed `_synthSelectedIds()`.
  Catalog source: [seed_catalog.json](../../../veritate_mri/data/seeds/seed_catalog.json), served by
  `GET /teacher/seeds`.
- Synth reattach: the active synth job id is stored at `localStorage["vt:training:synth_job"]`;
  `_synthReattach()` resumes polling on load and re-selects the job in the `#synthJobSelect`
  destination picker. `GET /teacher/synth/status` falls back to reading the job dir from disk when
  the id is no longer in the server's in-memory `_JOBS`, so status survives a server restart.
  Training reattaches via the backend PID file; rag reattaches via its singleton status poll.
- Synth resume: starting a job sets it as the selected destination, so the Start button continues
  the same job instead of spawning a new one. `_synthSyncStartLabel()` relabels it `Resume` when a
  stopped destination job already has samples. The runner skips ids already in `samples.jsonl`
  (`_load_done_ids` in [synth.py](../../../veritate_mri/teacher/synth.py)), so only pending/failed
  prompts re-run. With no seed boxes ticked, `_synthStart` resumes from the job's stored
  `meta.seeds`. There is no manual free-memory control: `SynthJob.run()` unloads the teacher model
  (local providers only) when the job ends, whether stopped or completed.
- Synth status line (`#synthStatusLine`, set in `_synthPollOnce`): shows `completed/failed/skipped`
  plus the top failure reasons from `error_summary` (e.g. `timed out x25`). A job stopped by the
  circuit breaker reports `ABORTED` in hot color. Per-failure detail is persisted to `errors.jsonl`
  in the job dir; the breaker config lives in [synth.py](../../../veritate_mri/teacher/synth.py).
- Training files panel (`#trainFilesPanel`, below the main train panel): shown/hidden with the synth panel
  by `_trShowSynthPanel` (synth flow only; `display:none` otherwise). Lists every `veritate_mri/data/synth_jobs/<job_id>/` dir,
  rendered by `_trFilesRender` from the same `synthState.jobs` that `_synthLoadJobs` fetches (`GET /teacher/synth/jobs`),
  so it populates when the flow opens and on the panel's `refresh` button. Deleting (`_trFilesDelete`, behind a
  `confirm`) calls `POST /teacher/synth/delete` then `_synthLoadJobs()` — a light refresh that drops the job from
  both this list and the `#synthJobSelect` destination dropdown. A running job's delete button is disabled; the
  route refuses a live job (409) and rejects any id that does not resolve to a direct child of `veritate_mri/data/synth_jobs/` (404).
  Built corpora and trained models are not touched.

### Author-a-corpus panel

- `#authorPanel`, shown by `_trShowAuthorPanel` for the `author` flow only. It reuses the synth job
  machinery (`GET /teacher/synth/status|samples|jobs`, `POST /teacher/synth/stop`) and adds three
  routes of its own: `GET|POST /teacher/authoring/spec`, `POST /teacher/authoring/start`,
  `POST /teacher/authoring/build`. Pipeline reference:
  [corpus/authoring.md](../../corpus/authoring.md).
- Genre list (`#authorGenreList`, `_authorRenderGenres`) is built from the spec, not from markup:
  one checkbox per genre with its share of the byte target, the brief as the hover title. Inputs
  are target size in MB, calls at once (blank falls back to the Settings concurrency), and the
  variety floor, which is posted as `ngram_distinct_floor` and overrides the spec value for that run.
- `_authorRenderStats` shows records kept, MB written, wording variety, em dashes rewritten, and
  failed batches, plus a rejected-by-reason line. When `authoring.ngram_below_floor` is true it
  raises `#authorRepWarn` in hot color naming the measured percentage, the floor, and the fix. This
  is the guard that a template-expanded corpus cannot pass unnoticed.
- Build (`_authorBuild`) appears only when the job is stopped and has records. On success
  `#authorHandoff` prints the numbered upload steps returned by the route (which zip to upload,
  that the returned link replaces the PLACEHOLDER `train_url`, and that `coming_soon` then comes
  off) alongside the zip and bin sha256s.
- The active authoring job id is stored at `localStorage["vt:training:author_job"]` and re-polled
  when the flow reopens.

## Dependencies

- `/runs` and `/run/<name>/csv` from [runs_routes.py:256](../../../veritate_mri/routes/runs_routes.py#L256).
- `/train_stream` SSE from [train_routes.py:50](../../../veritate_mri/routes/train_routes.py#L50).
- Backend training CSV contract at [save.py:38](../../../veritate_mri/training/save.py#L38) — `step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed`. Any change to that header breaks this tab.
- [canvas_rendering.md](canvas_rendering.md) for chart helpers.

## Pitfalls

- A run with no `train.csv` doesn't appear in `/runs`. Trainer skeletons need to write at least the header at startup for visibility.
- SSE reconnects are silent. If the backend dies and restarts, the stream stops but the polled CSV keeps the tab usable.
- The plateau detector uses smoothed differences; very short runs (under ~50 steps) sit in WARMING the whole time.
