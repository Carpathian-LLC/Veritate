# Settings tab

## What it is

Dashboard preferences, trainer/plugin configuration, device preference, heartbeat consent, teacher model (provider + model), mesh role.

## How it works

Markup at [index.html:1266-1642](../../../veritate_mri/web/index.html#L1266). Sectioned panels for: display, runtime, engine, training, analytics, teachers, mesh, advanced.

- **Training** section holds the compute-device override (`#devicePreferenceSelect`, posts `device_preference`).
- **Warm models** section (`#warmModelsList` / `#warmModelsSummary`), directly after the Engine section, picks which exported models stay resident as C-engine subprocesses. Data-driven from `GET /backends` (`c.warm`); posts `warm_models`. See [warm_models_panel.md](warm_models_panel.md).
- **Analytics** section is three boxes: Detect system, Heartbeat, Advanced. Heartbeat is always on (no off switch) and shows the auto-generated, editable device name. Advanced groups the telemetry toggles (`analytics_advanced_enabled`, `share_current_training`, `heartbeat_send_errors`, `diagnostics_logs_enabled`) plus the review/preview buttons. `analytics_advanced_enabled`, `share_current_training`, and `heartbeat_send_errors` default on; `diagnostics_logs_enabled` defaults off.

- Settings load via `GET /settings`. The whole `mri_settings.json` object is returned and hydrates every form field.
- Each form change POSTs the patched key to `/settings`.
- `#sysDetectBtn` triggers `POST /sys/detect` to re-detect hardware (CPU, GPU, RAM) and store the result.
- The Detect system panel has no Auto tune button. Auto tune is a trainer benchmark and lives only on the Training tab (`#trainAutoTuneBtn`, see [training_tab.md](training_tab.md)); it writes the `measured` key of `data/system_specs.json`. `_renderSysSpecs` shows that measured line in green when present. See [../../platform/bench.md](../../platform/bench.md).
- The **Corpus library** card (`#corpusLibraryRow`, expanded view in `#corpusLibraryModal`) lists installable training corpora from `GET /corpus/library/catalog` and installs into `trainers/corpus/`. Entries tagged `built-in` ship inside the repo (`format: "native"`) and install as a local copy with no download. Rows render in a two-level grouping resolved client-side: **family** (who publishes the corpus) is the outer header, **topic** (what the corpus teaches) is the inner header. `CORPUS_FAMILIES` sets the family order (Carpathian corpora, then Public corpora) and `CORPUS_TOPICS` the topic order within each family (Chat, Agent / tool use, MCP protocol, Code, Knowledge, Special SFT, plus the remaining ids in that constant); each header prints the label and the constant's one-line blurb. `_corpusFamilyOf(c)` returns the catalog entry's `family` field, falling back to a per-stem table and then to `carpathian`. `_corpusTopicOf(c)` returns the entry's `topic` field, falling back to a per-stem table, then to the first of the entry's `trained_modes` that maps to a topic, then to `knowledge`. Empty groups are skipped, and any family or topic id outside the two constants renders under an `Other` header. The Code topic holds the `*_code_*`, `code_qa_*`, and `mixed_code_*` stems from the code corpus builder (see [../../corpus/code_corpus.md](../../corpus/code_corpus.md)). Unpublished entries carry `coming_soon: true` in the catalog entry itself (corpus_catalog.json) and render disabled with a `coming soon` tag; there is no client-side stem list. The modal also holds the **mix planner** (`#corpusMixPanel`), which combines several rows into one weighted training spec: see [corpus_mix_planner.md](corpus_mix_planner.md). See [../backend/corpus_library.md](../backend/corpus_library.md).
- In-flight installs show on the settings card itself, not only inside the modal. Every catalog entry carries a server-side `progress` object (`kind`, `bytes`, `total`), so `_corpusRenderActiveDownloads` paints one bar per active download into `#corpusActiveDownloads` on each catalog render, independent of which client started it or whether the page has been reloaded. `_corpusProgressBarHtml` builds that bar (determinate when `total` is known, striped otherwise) and is shared with the modal row renderer. `_corpusPollLoop` refreshes the catalog while the settings tab is active, at `CORPUS_POLL_ACTIVE_MS` (2s) when a download is running and `CORPUS_POLL_IDLE_MS` (15s) otherwise; `activateTab("settings")` refreshes once on entry so the panel is current immediately.
- An extension's downloadable add-on datasets render inside the **marketplace** (`#mktMarketplaceModal`, `mkt*` in `index.js`), per extension. While the marketplace lists each extension, it fetches `GET /extensions/<id>/data` and renders a per-extension **Data** subsection (containers keyed by `[data-mkt-data="<id>"]`, populated by `_mktDataLoad`/`_mktDataRender`); extensions with no datasets show no Data section. Each dataset row shows size (`size_gb` when present, else `approx_gb`) + local status with a **download** button (`POST /extensions/<id>/data/download {source}`; disabled `coming soon` when `downloadable:false`) and a **delete** button shown only when `present` (`POST /extensions/<id>/data/delete {source}`; reclaims disk, symlinked datasets only lose the link, a not-yet-hosted dataset confirms delete is permanent). Backend `note`/`error`/`reclaimed_gb`/`unlinked` results surface as one-line plain notes in `#mktActionStatus`, never raw stack/CLI output. Datasets live under `extensions/installed/<id>/data/extension_data/<source>`, declared per extension in its `data_catalog.json` and served by the generic per-extension data routes. The marketplace and dataset-catalog contracts are documented in the in-app wiki `extensions` category (`veritate_mri/data/wiki/extensions/`).
- A build-notices banner reads the build number from `versions.json` (via `/versions`) and shows acknowledgement prompts for new builds.
- The **System** panel holds the lifecycle controls: **Power save** (`#minimalModeBtn`, restore link mirrored in the top `#minimalModeBanner`), **soft reload** (`#softReloadBtn`), **reload python** (`#restartServerBtn`), and **kill** (`#killServerBtn`), wired to the `_lifecycle*` handlers in `index.js`. The three relaunch actions (power-save toggle, soft reload, reload python) plus the app-update pull (`#updatePullBtn`, `_appUpdatePullWithGuards`) raise `#lifecycleOverlay`, a full-screen grey pinwheel (`_lifecycleOverlayShow`/`_lifecycleOverlayHide`) whose blue arc runs the `pinstutter` keyframe (uneven-speed, held-plateau spin). It stays up until the page reloads or the action fails, so the relaunch is visible from any tab. The update pull keeps it up only when auto soft-reload is on, otherwise hides it once the pull returns. Kill does not raise it (the server does not come back).

Settings store at [settings.py](../../../veritate_mri/runtime/settings.py); see [../backend/settings.md](../backend/settings.md).

## Dependencies

- `/settings` GET and POST routes from [settings_routes.py](../../../veritate_mri/routes/settings_routes.py).
- `/sys/detect` from [sys_routes.py](../../../veritate_mri/routes/sys_routes.py).
- `/versions` from [sys_routes.py:115](../../../veritate_mri/routes/sys_routes.py#L115).

## Pitfalls

- Some settings only take effect after a dashboard restart (e.g., `pytorch_load_mode`, `mesh_role`). The UI doesn't yet flag which ones, so restart when in doubt.
- `device_name` is auto-generated on first setup (`_random_device_name()` in [settings.py](../../../veritate_mri/runtime/settings.py), e.g. `brave-otter-07`) and capped at 15 characters (validated server-side). It is the editable device id shown in the heartbeat box.
- `analytics_advanced_enabled`, `share_current_training`, and `diagnostics_logs_enabled` gate what fields the heartbeat ships; see [../backend/heartbeat.md](../backend/heartbeat.md) for the tier definitions.
- Teacher model is a dropdown (`#teacherModelList`): a "connected models" optgroup lists every model the provider reports via `POST /teacher/models` (`list_models()`, deduped), plus a "custom..." entry that reveals the free-text input (`#teacherModel`) for names not in the list. The hidden input always holds the value that saves; picking from the list writes into it. The list refreshes on provider change, base-url blur, api-key blur, and form hydrate; a failed fetch (e.g. local server down) leaves only "custom...", with the saved name still editable. Providers with `model_selectable: false` (Carpathian, where the API key picks the model) hide both controls. Selecting a model autosaves the teacher config immediately (`_saveTeacher()`): picking a concrete entry from the list, or committing a value in the custom input (`change`). Other teacher fields (provider, key, base-url, concurrency) still save via the Save button.
- Key policy: the bundled Carpathian `cai_` key in [settings.py](../../../veritate_mri/runtime/settings.py) is a PUBLIC shared key, intentionally committed. It is the `PUBLIC_AI_KEY` constant, injected live by `settings.get()` and never persisted to `mri_settings.json`, so rotating it in source takes effect on every install at next load. Only an operator's own `ai_api_key_user` override is stored. The AI-assist panel explainer copy is served the same way: `PUBLIC_AI_BLURB` maps to `ai_assist_blurb`, rendered into `#aiAssistBlurb` by the frontend (not hardcoded in HTML), so the wording is edited once in source.
- Teacher provider configs are remembered per provider (`teacher_configs` in settings): every Save snapshots that provider's key/model/base-url, and switching providers restores the remembered values (key shown as a mask; raw keys never reach the frontend). Picking a never-saved provider starts blank; keys are not carried across providers.
- The provider dropdown appends "(connected)" per `_teacherIsConnected()`: API providers with a stored key, and local providers (Ollama, LM Studio, llama.cpp) whose server answers `POST /teacher/models` with at least one model. Local providers are probed on form hydrate (`_teacherProbeLocalProviders()`) and labels update in place when a probe lands.
- Teacher `max_concurrency` (advanced box) is the parallel-request count synth fires. For local providers the backend clamps it to `LOCAL_MAX_CONCURRENCY` in `_resolve_concurrency` (teacher_routes.py) so a high global value never floods a single local GPU into an out-of-memory crash, regardless of `OLLAMA_NUM_PARALLEL`; no server-side parallelism tuning is needed. Cloud APIs take the value as-is. The field hydrates to the effective (capped) value via `_teacherEffectiveConc` so it never shows a number the backend silently overrides.

## Typing recorder

Records how a person actually types, so the draft trigger can be tuned against real
typing instead of an invented average. The generation-ahead controls it informs live in
the Generation tab ([generation_tab.md](generation_tab.md)), not here: they are
generation controls, and keeping a copy in Settings left one panel describing a
behaviour the other had already replaced.

**Typing recorder.** The evidence the draft trigger is tuned against. Type into
`calBox`; every keystroke is recorded, and pressing Enter marks the keystroke you were
finished on and starts the next question. That mark is the only label needed: it makes
one keystroke a known "done" and every other one a known "not done", which turns
tuning from taste into a scorable classification.

Each record is `{t, gap, ch, len, ctx, word, done, still_after}`:

- `ctx` / `word` describe where the person was **sitting during the gap**, so they come
  from the text BEFORE that keystroke. Taken after, a long reach for the next word
  lands on that word's first letter and is labelled `word` (mid-word hesitation) when
  it was really a `boundary` pause: the pauses that matter most would get the one
  label that hides them. `ctx` is `word` / `boundary` / `clause` / `sentence`.
- `box` / `prevLen` are the state the draft rule saw, kept for replay.
- `still_after` on a labelled keystroke is how long the person sat still before
  declaring the question finished: the latency a draft has to beat.

`_recScore` replays the session through `_draftDelayMs` and `_draftEligible` — the same
two functions the composer runs — and reports two numbers: how many labelled questions
would have started a draft, and how many drafts would have been generated mid-question
for nothing. Any rule change is scored the same way against the same recording.

**Nothing here summarizes.** No median, no percentile, no recommendation, no
self-applied threshold. Typing speed is not one number: the gap varies systematically
with where in the text it falls, and collapsing that to an average destroys the only
structure that separates "pausing mid-sentence" from "finished". The session is handed
back raw via `save session` (POST `/typing/samples`, stored under
`data/typing_samples/`, see [../backend/typing_samples.md](../backend/typing_samples.md))
or `copy raw json`.

`#calStrip` draws one bar per gap against a dashed rule at the current threshold:
`--dim` ordinary, `--cool` would start a draft, `--highlight` the keystroke you pressed
Enter on. State is carried by height as well as color (validated: worst CVD dE 21.6,
normal 25.8 against the panel surface). Each bar's `title` gives its gap, character,
context, trailing word, and the delay it had to beat.

`speculative_pause_ms` is the threshold the rule currently uses; `0` tracks the
typist's live median keystroke gap. The recorder and the composer feed ONE rolling
sample (`_recordTypingGap`). With under three samples `_typingMedianMs` falls back to
`PREFETCH_PAUSE_MIN_MS / PREFETCH_PAUSE_FACTOR`, the floor: falling back to the ceiling
made the first question of every session wait the longest, which is backwards.

`undo last finish` un-marks the most recent Enter and puts its text back in the box,
leaving the keystrokes untouched: an Enter pressed by accident otherwise mislabels the
one keystroke the whole session is scored against. `save session` clears the recorder
and starts fresh, because leaving it in place made every later save re-contain the
keystrokes already stored in an earlier file.
