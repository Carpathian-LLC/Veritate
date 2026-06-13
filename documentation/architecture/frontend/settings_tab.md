# Settings tab

## What it is

Dashboard preferences, trainer/plugin configuration, device preference, heartbeat consent, teacher model (provider + model), mesh role.

## How it works

Markup at [index.html:1266–1642](../../../veritate_mri/web/index.html#L1266). Sectioned panels for: display, runtime, engine, training, analytics, teachers, mesh, advanced.

- **Training** section holds the compute-device override (`#devicePreferenceSelect`, posts `device_preference`).
- **Analytics** section is three boxes: Detect system, Heartbeat, Advanced. Heartbeat is always on (no off switch) and shows the auto-generated, editable device name. Advanced groups the three opt-in telemetry toggles — `analytics_advanced_enabled`, `heartbeat_send_errors`, `diagnostics_logs_enabled` — plus the review/preview buttons.

- Settings load via `GET /settings`. The whole `mri_settings.json` object is returned and used to hydrate every form field.
- Each form change POSTs the patched key to `/settings`.
- `#sysDetectBtn` triggers `POST /sys/detect` to re-detect hardware (CPU, GPU, RAM) and store the result.
- `#sysAutoTuneBtn` opens the Auto tune modal (`#autoTuneModal`, shared with the Training tab; it uses the Training tab's selected trainer and prompts to pick one if none is selected): a measured benchmark that finds the real RAM ceiling and throughput sweet spot via a trainer's `--bench` mode, then writes the values into the trainer manifest and the `measured` key of `data/system_specs.json`. `_renderSysSpecs` shows the measured line in green when present. See [../../platform/bench.md](../../platform/bench.md).
- The **Corpus library** card (`#corpusLibraryRow`, expanded view in `#corpusLibraryModal`) lists installable training corpora from `GET /corpus/library/catalog` and installs into `trainers/corpus/`. Entries tagged `built-in` ship inside the repo (`format: "native"`) and install as a local copy with no download. Rows render grouped by purpose under labeled section headers (Chatting, Autocomplete, Facts, Statistics, then Other), resolved client-side in `_corpusCategoryOf` by stem (`CORPUS_STEM_CATEGORY`), then `trained_modes`, then a Facts fallback. Unpublished entries render disabled with a `coming soon` tag: the existing `chat_500mb`/`chat_5gb`/`agent_150mb`/`agent_1500mb` (`CORPUS_COMING_SOON`) and the Market LLM placeholders `crypto`/`stocks` (`CORPUS_MARKET_PLACEHOLDERS`, Statistics, injected at render only and absent from `corpusLibState.catalog`). See [../backend/corpus_library.md](../backend/corpus_library.md).
- A build-notices banner reads the build number from `versions.json` (via `/versions`) and shows acknowledgement prompts for new builds.

Settings store at [settings.py](../../../veritate_mri/runtime/settings.py); see [../backend/settings.md](../backend/settings.md).

## Dependencies

- `/settings` GET and POST routes from [settings_routes.py](../../../veritate_mri/routes/settings_routes.py).
- `/sys/detect` from [sys_routes.py](../../../veritate_mri/routes/sys_routes.py).
- `/versions` from [sys_routes.py:115](../../../veritate_mri/routes/sys_routes.py#L115).

## Pitfalls

- Some settings only take effect after a dashboard restart (e.g., `pytorch_load_mode`, `mesh_role`). The UI doesn't yet flag which ones — when in doubt, restart.
- `device_name` is auto-generated on first setup (`_random_device_name()` in [settings.py](../../../veritate_mri/runtime/settings.py), e.g. `brave-otter-07`) and capped at 15 characters (validated server-side). It is the editable device id shown in the heartbeat box.
- `analytics_advanced_enabled` and `diagnostics_logs_enabled` gate what fields the heartbeat ships; see [../backend/heartbeat.md](../backend/heartbeat.md) for the tier definitions.
- Teacher model is a dropdown (`#teacherModelList`): a "connected models" optgroup lists every model the provider reports via `POST /teacher/models` (`list_models()`, deduped), plus a "custom..." entry that reveals the free-text input (`#teacherModel`) for names not in the list. The hidden input always holds the value that saves; picking from the list writes into it. The list refreshes on provider change, base-url blur, api-key blur, and form hydrate; a failed fetch (e.g. local server down) leaves only "custom...", with the saved name still editable. Providers with `model_selectable: false` (Carpathian — the API key picks the model) hide both controls. Selecting a model autosaves the teacher config immediately (`_saveTeacher()`): picking a concrete entry from the list, or committing a value in the custom input (`change`). Other teacher fields (provider, key, base-url, concurrency) still save via the Save button.
- Key policy: the bundled Carpathian `cai_` key in [settings.py](../../../veritate_mri/runtime/settings.py) is a PUBLIC shared key, intentionally committed. It is the `PUBLIC_AI_KEY` constant, injected live by `settings.get()` and never persisted to `mri_settings.json`, so rotating it in source takes effect on every install at next load. Only a user's own `ai_api_key_user` override is stored. The AI-assist panel explainer copy is served the same way: `PUBLIC_AI_BLURB` → `ai_assist_blurb`, rendered into `#aiAssistBlurb` by the frontend (not hardcoded in HTML), so the wording is edited once in source.
- Teacher provider configs are remembered per provider (`teacher_configs` in settings): every Save snapshots that provider's key/model/base-url, and switching providers restores the remembered values (key shown as a mask; raw keys never reach the frontend). Picking a never-saved provider starts blank — keys are not carried across providers.
- The provider dropdown appends "(connected)" per `_teacherIsConnected()`: API providers with a stored key, and local providers (Ollama, LM Studio, llama.cpp) whose server answers `POST /teacher/models` with at least one model. Local providers are probed on form hydrate (`_teacherProbeLocalProviders()`) and labels update in place when a probe lands.
- Teacher `max_concurrency` (advanced box) is the parallel-request count synth fires. For local providers the backend clamps it to `LOCAL_MAX_CONCURRENCY` in `_resolve_concurrency` (teacher_routes.py) so a high global value never floods a single local GPU into an out-of-memory crash, regardless of `OLLAMA_NUM_PARALLEL`; no server-side parallelism tuning is needed. Cloud APIs take the value as-is. The field hydrates to the effective (capped) value via `_teacherEffectiveConc` so it never shows a number the backend silently overrides.
