# veritate documentation

The single platform reference. Every component, contract, and constraint lives in this file; the dashboard wiki tab serves it. Rules of engagement live in `claude_preflight.md`; research results live in `successes.md`, `failures.md`, and `ideas.md`.

## overview

Veritate trains, inspects, and serves byte-level language models on consumer hardware. Three parts, independent, communicating only through files on disk:

- **Trainer** (PyTorch): writes `models/<name>/checkpoints/` and hook dumps.
- **Inference engine** (C + asm, CPU): reads `.bin` model files, speaks stdin/stdout.
- **MRI dashboard** (Flask + JS): browser UI at `http://localhost:8001/` for training, inspection, generation, and settings.

Design rules:

1. Byte-level only. Vocab is 256; there is no subword tokenizer anywhere.
2. Every new mechanism carries a per-byte wall-clock budget on target hardware.
3. Every kernel is bitwise-identical to its scalar reference before shipping.

Launchers: `start.command` (macOS), `start.bat` (Windows), or `python veritate.py` with flags `--model`, `--port`, `--threads`, `--skip-build`, `--no-browser`. These, the engine build scripts, and subprocess entry points the platform spawns are the only sanctioned hand-run scripts.

## repo layout

| path | contents |
|---|---|
| `veritate_core/` | training-side library: model, optimizers, QAT, plugin surface |
| `veritate_mri/` | Flask app, routes, readers, runtime, training pipeline, inference, web UI |
| `veritate_engine/` | C engine: `src/`, `kernels/{scalar,arm64,x86_64,inline}/`, `bin/<os>/<arch>/veritate` |
| `veritate_mesh/` | optional inter-device federation (hub/node) |
| `extensions/` | third-party extension bundles (`canonical/`, `installed/`) |
| `data/corpus/` | training corpus bins, canonical location |
| `models/<name>/` | run output (gitignored) |
| `trainers/corpus/` | read-only mirror of corpus bins kept for installs that resolve it; in the updater's skip set |
| `tests/<area>/` | pytest suites mirroring platform areas |

Model naming: `<slug>_<size>` or `<corpus>_<size>_<precision>_<version>`. Every code file belongs to exactly one of the five source trees; a file that fits none is a defect.

## versioning

`versions.json` at repo root: `channel`, `build` (global monotonic counter, moves every changeset), and semver strings `engine`, `mri`, `format`. Each answers "can this artifact still be read by that code": major = existing artifacts or callers stop working (needs build note + migration), minor = additive, patch = behavior-only fix. Never bump any of them without explicit user permission.

Build notes are sections under `## build notes` at the end of this file: three to ten lines, user-facing actions only, one per build number. A breaking build also gets a `BUILD_NOTICES` entry in `veritate_mri/runtime/settings.py` keyed by the **integer** build number (a JSON round-trip that turns keys into strings silently breaks the modal); `pending_notices()` raises the dashboard modal until the user acknowledges. The modal message is written by hand; it is not derived from the build-note text.

## backend

### startup and lifecycle

`app.py` startup order: CLI args → model resolve → thread budget → pre-build hook → minimal-mode gate → heartbeat → route registration → app-sync hook → eager brain load → warm pool. The global exception handler always returns JSON, never HTML. `VERITATE_MINIMAL=1` must be set before process start; it propagates across restarts via `LAUNCH_CMD`.

Server `restart`/`soft_reload` do not kill a running training subprocess: `trainer_runner` reattaches via PID file (`_recover_from_disk`). Detached trainers (PPID 1) survive restart; runner-managed children are killed by a full stop.

### routes layer

One route module per concern under `veritate_mri/routes/`, each exposing `register(app)`. Routes never touch the filesystem directly — they call readers. `/generate` and `/agent/stream` loopback-gate any caller-chosen filesystem path.

### response compression

`runtime/compression.py` owns every gzip decision; no route module encodes anything itself. `register(app)` runs before the route modules so its `after_request` hook is the last one flask calls and sees the final body. A response is compressed only when it is a 200, its mimetype is in `COMPRESSIBLE_MIMETYPES`, it carries no `Content-Encoding` already, and the client sent `Accept-Encoding: gzip`; it then gains `Content-Encoding: gzip` and `Vary: Accept-Encoding`. Buffered bodies under `MIN_COMPRESS_BYTES` (1024, one TCP segment) and every binary or already-encoded payload pass through untouched, so a client that does not advertise gzip receives byte-identical bytes.

SSE streams are compressed with a `Z_SYNC_FLUSH` per source chunk, so each event leaves the compressor as its own decodable frame and time-to-first-token does not move; the stream's `Content-Length` is dropped. `GZIP_LEVEL = 6` is measured on cardinal-01 (i7-9700T pinned at 800 MHz) against a 945 KB, 32-event `/generate` trace stream: level 1 = 9.85x at 0.65 ms/event, level 6 = 11.89x at 1.49 ms/event, level 9 = 12.47x at 6.62 ms/event, against ~29 ms of engine time per event. Live on that box a 30-token `/generate` stream is 945 KB raw and 83 KB on the wire (11.4x), its 30 token events arriving spread across the stream, at a median time-to-first-token of 0.235 s against 0.249 s uncompressed. Pinned by `tests/mri/test_compression.py`.

### readers layer

`veritate_mri/readers/` is the data-access layer: routes call readers, readers own all `os.listdir`/`open`. Missing files return `None` or empty, never raise.

Key readers: `paths.py` (every canonical path; `corpus_search_dirs()` returns `(data/corpus, trainers/corpus)`, canonical first, and `_corpus_file_path()` resolves existing-file-wins — anything listing or globbing corpora walks both), `trainers.py` (builds the trainer manifest from `trainer_sizes.json`; default size comes from that file's `default_size` key), `bin.py` (`.bin` header parsing; `RETIRED_VERSIONS = {10}` hard-rejects with a clear error; `act_boost` returns `None` for v13), `wiki.py` (serves this file to the dashboard).

### settings store

`mri_settings.json` via `veritate_mri/runtime/settings.py::DEFAULTS`. `PUBLIC_AI_ENDPOINT`/`PUBLIC_AI_KEY` are injected live and never persisted. Load strips unknown stale keys once.

### auth

Two independent gates. Dashboard password: off unless `VERITATE_DASHBOARD_PASSWORD` is set; public surface is `/`, `/login`, `/logout`, `/static`; without `VERITATE_SECRET_KEY` sessions reset every restart. API bearer key: gates `/generate`, `/agent/stream`, `/v1/*` only; key and counters live in `mri_settings.json`.

### updater (app_sync)

Pulls a channel-mapped branch tarball. `DEFAULT_SKIP_DIRS` = `models/`, `plugins/`, `data/`, `experiments/`, `.git`, `.venv`, `venv`, `__pycache__` — matched on the **first path segment only**, so `data/corpus/` is skipped but `veritate_mri/data/` is not (pinned by `tests/mri/test_updater_skip_depth.py`). `_safe_extract` blocks absolute and `..` entries. The updater never deletes files dropped upstream. It deletes `venv/.req_hash` after a pull to force a pip re-run. `local_edits()` diffs the working tree against git state for the update gate, skipping venv dirs (detected by `pyvenv.cfg`) and dot-dirs.

`sync_common.py` (`.sync_state.json` state files) is the shared engine for `app_sync`, `corpus_sync`, and `models_sync`.

### heartbeat

Carpathian webhook daemon; presence is always on. Training detection: `plugin_runner.state()` primary, `train.csv` mtime fallback. Tier gates: `share_current_training`, `analytics_advanced_enabled`, `heartbeat_send_errors`. Machine identity from a hardware fingerprint (`_ensure_identity`).

### inference

- **Brain** (PyTorch): single global instance behind a lock. Three-way variant dispatch on `pos_emb` / `mtp.transforms.*` presence; consumers call contract methods (`forward` → `(logits, loss)`, `project_byte0`), never `isinstance` on a variant. CPU thread count autotuned per shape and cached. Lookahead drafting is CPU-only (`LOOKAHEAD_DEVICES`); byte-exactness is unproven on MPS.
- **C engine backend** (`c_engine.py`): subprocess per model. Warm pool (`cfg["C_WARM"]`): independent resident subprocesses, `warm_select` re-points without spawning, no automatic C-side eviction (the idle watcher only unloads PyTorch). Model discovery requires a `step_<N>.pt` placeholder even for bin-only deployments.
- **Read-ahead**: sends the open prefix (not the closed wire prompt) while the user types so the engine state cache can hit; on by default for dashboard and API. Only the C engine has a state cache; PyTorch ignores it.
- **Speculative prefetch** (generate-ahead): off by default; a real request always wins the engine (`take()` cancels the job); only `/generate` claims a buffer. A served/spent ratio below ~50% is a losing energy trade.
- **RAG injection** (`/generate` with a `rag` corpus path): BM25 top-1 by default (`RAG_K_DEFAULT`; multi-candidate injection is measured-worse at 200M). Passages land inside the final user turn of the ChatML prompt via `build_rag_prefix`, budgeted by `injection_budget(seq, prompt_bytes)`: the live model's window minus the prompt and a `REPLY_RESERVE_B` reserve, so injection never evicts the turn frame. A prompt already at the window injects nothing. Retriever hits carry whole chunks.
- **Streaming decode** (`/generate?fast=stream`, PyTorch backend, recurrent-mixer models with `slots % 64 == 0` i.e. seq a multiple of 256): unbounded-context generation over `forward_streaming`. Full windows commit into the carried per-block recurrent states (one forward each); only the current partial window is recomputed per byte, padded with a non-boundary byte to keep the slot count CHUNK-aligned. The prompt is never truncated — context beyond `seq` lives in the state. State commits only from full windows (a part-full window's conv tail carries padding-derived columns). Argmax-parity with `stream()` inside one window is pinned by tests/mri/test_stream_fast_streaming.py. Use with state-carry-adapted checkpoints (wren1_3+); a carry-off model's state arrives ~0 and the walk degrades to window-local context. **Persisted conversations**: add `state_id=<[A-Za-z0-9._-]{1,64}>` (loopback only) and the carried states + pending buffer save to `data/stream_states/<id>.pt` after every call — the next call sends ONLY the new bytes and continues byte-exactly (split-call parity pinned by tests). `state_reset=1` starts the id over. A state is bound to the exact checkpoint that wrote it; a mismatch errors instead of silently degrading.
- **Experience log**: every completed `/generate` exchange (both backends) appends one JSONL record — base64 prompt/output bytes, model, params — to `data/experience/YYYYMMDD.jsonl` (`inference/experience.py`). The replay substrate for sleep consolidation (ideas.md IDEA 20 T3): the model trains on its own thought and actions. RAG-injected prompts are recorded as injected (what the model actually saw). Partial replies on client disconnect still record. `VERITATE_EXPERIENCE_LOG=0` disables; 512 MB/day cap; never breaks serving (all failures degrade to a warn). `tools/build_experience_corpus.py` turns the log into `experience_{train,val}.bin` for sleep consolidation (dedupe, min-reply filter; rehearsal = the corpus mixer weighting the model's own base corpus alongside it).
- **Sleep controller** (`training/sleep.py`, routes `/sleep` GET, `/sleep/wake` POST, `/sleep/now` POST): when serving has been idle `sleep_idle_min` minutes and no trainer is running, builds the experience corpus and launches a consolidation run on `sleep_model` through the one trainer. Dose scales with use: `steps = new_exchanges × sleep_steps_per_exchange` clamped to `[sleep_min_steps, sleep_max_steps]`; under `sleep_min_exchanges` new exchanges the night is skipped. The recipe is the model's own `config.json` `training_args` with only the sleep levers overridden (constant `sleep_lr`, no warmup, `sleep_corpus` mix, assistant loss mask, `ckpt_every = sleep_ckpt_every`). Dense sleep checkpoints make an early wake cheap; when a sleep run ends, its intermediates are deleted and older sleep finals thin to `sleep_keep_finals` — checkpoints not created by sleep are never touched. `/sleep/wake` stops the run (serving resumes from the newest checkpoint); `/sleep/now` skips only the idle timer. State (last sleep, in-flight run, surviving finals) lives in `data/sleep/state.json`; every state change (fell asleep / woken / awake) also appends to `data/sleep/history.jsonl`, served newest-first in the `/sleep` payload's `history`. Watcher thread ticks every 60 s (skipped in `--minimal`). Off until `sleep_enabled` is set and `sleep_model` names a model. UI: the gen-bar chip shows awake countdown / sleeping progress + eta with a wake button, and a sleep review box directly below the Generation chat shows live status, sleep-now / wake buttons, a usage-by-hour ledger (exchanges per local hour over 7 days, `activity_by_hour` in the payload), and the event history (box stays visible while the feature is disabled so history remains reviewable). Launch guards, each from a measured failure (cardinal 2026-08-20): save()-stamped bookkeeping keys (`corpus_bytes`, `corpus_sha256`, `output_dir`) are stripped from the recipe before launch; both experience bins must reach one draw window (`seq × n_chunks + 2` bytes); the resume step is the latest checkpoint on disk, never config.json `step` (a model with no `.pt` cannot sleep); a sleep that gains no steps records a `failed` event and a 60-min cooldown instead of the watcher retrying every tick. Measured constraint: a recipe-batch sleep step (200M, bs48, seq1024×4) costs ≥920 s on an i7-9700T (bf16 CPU emulation; ≥65-80x an M3 Ultra) with 13 GB RAM peak — serving survives (p50 unchanged, worst 5.7x during multi-core phases) and on-box `.pt`→bin export is 9.8 s. Tests: tests/training/test_sleep_controller.py.
- **Repetition guard**: `rep_window` / `rep_penalty` / `no_repeat_ngram` on `/generate` and `/prefetch`. When the caller sends neither, chat-framed and RAG prompts default the hard ban ON (`no_repeat_ngram` 16, `rep_window` 256) and plain completion prompts default OFF; explicit params always win. The soft penalty always defaults off. Grading of SFT checkpoints stays bare-greedy; the guard is a serving default, never a measurement setting.
- **OpenAI-compatible serving** (`/v1/chat/completions`, `/v1/chat/mri`, `hybrid_routes.py`): shared ChatML framing and model routing (local model on either engine, teacher provider, or the public cloud model). `mri:true` is an opt-in telemetry flag and never changes sampling. The chat UI was extracted 2026-08-20 to a separate project (see `CHAT_HANDOFF.md`); the Generation tab is the conversational surface.
- **Hallucination detector**: span-level grading, verdict priority refused > grounded (≥0.85) > likely_hallucinated > partially_grounded > low_confidence > ungrounded_ok. Deferred mode grades the exact streamed answer with no re-generation. Provenance is BM25 similarity, not causal attribution.
- **Addons**: see the addons section.

### deps installer

Torch wheel repair uses `--index-url` (never `--extra-index-url`) — this is the load-bearing choice that prevents CPU-torch on a GPU box. Escalation ladder: `--user` → site-packages → UAC/`sudo -n`. `install_torch()` returns `restart_required: True`.

### sys metrics and sysprobe

`sys_metrics.py`: per-OS hardware detection feeding the HUD; failures fall back to `available: false`, never raise. Anything printed reports the specific compute device and memory kind (CPU/RAM, CUDA/VRAM, MPS/unified) — never a generic label. `sysprobe`: cross-platform hardware benchmark, independent of any model; the GPU probe is fp32 deliberately (matches training-loop dominance).

### mesh

Optional federation, `mesh_role` off/hub/node/both; routes register only when non-off. Machine-to-machine surface consumed by `veritate_mesh/client.py`; no UI expected.

### train_stream

`/train_stream` is an in-process SSE channel with a frontend `EventSource` subscriber, but no trainer publishes to it — trainer subprocesses cannot call into the server process; the dashboard polls `train.csv` instead. Treat as unwired until a publisher exists.

## frontend

Single-page app: `veritate_mri/web/index.html` + `index.js`. Tabs: Generation, Models, Training, Distillation, Wiki, Logs, Settings. The chat tab was extracted 2026-08-20 to a separate project (see `CHAT_HANDOFF.md`); the Generation tab is the conversational surface. Hash-based navigation; the `valid` array in `index.js` is the sole allowlist of routable tabs — an unknown tab silently falls back to `generation`. Adding a tab: add to `valid`, add an `activateTab` branch, and update the tutorial selectors (renaming a tab breaks the tutorial spotlight silently). `MINIMAL_HIDE_TABS` hides tabs in minimal mode.

Data transport: polling (train CSV 5s, runs 30s) plus SSE for streams. SSE has no HTTP status on failure — the log view re-polls a snapshot to detect stalls.

Conventions:

- Canvas charts: `fitCanvas` returns early on a detached canvas; `ResizeObserver` coalesces redraws; drawers are idempotent.
- Dropdowns anchor to the same horizontal edge as their trigger.
- Every `localStorage` call is wrapped in try/catch; localStorage is shared per origin, so two dashboards on one machine collide.
- Standalone modules (prune, tutorial) use the IIFE pattern.
- HUD meters hide when their source field is null — null ≠ 0; never fabricate a reading. macOS has no first-party CPU temperature API.
- Models tab: `modelShape` is the one module-global driving dimension-scaled UI; a `0` field means "not known yet".
- Generation tab: chat stop markers are matched **without** the closing bracket because byte models reproduce markers approximately. Read-ahead and speculative prefetch are separate controls with different risk profiles. Chat is the default mode and is never capability-gated; agent and autocomplete grey out until their tier is trained, refreshed by a 15s capability poll.
- Training tab: only multi-size manifests show the size dropdown. Auto-tune writes to the machine-local tuning store, never the manifest.
- Corpus mix planner: profiles are server data, never hardcoded. Ticking the Training-tab corpus picker rebuilds the field from checked stems joined `+`, discarding weights — re-accept the mix to restore them.
- Seed packs (`veritate_mri/data/authoring/seeds/<vertical>.json`, read by `readers/seeds.py`): the subject list interview mode draws its opening questions from, split into named topic groups so a corpus can be narrowed rather than always being one file. Ships five verticals: `conversation` (3,500 seeds / 50 topic groups) and `code`, `technical`, `business`, `medical` (1,520 seeds / 40 groups each). 9,580 seeds over 210 topics, no duplicate anywhere within or across the set. Vertical and genre are ORTHOGONAL: the pack supplies the subject, the genre supplies the behaviour and its gate, so `code` seeds with the `conversation` genre give conversations about code and with `instruct` give task-following about code. No new genre is needed per vertical. A vertical is selectable ONLY when its pack is on disk and parses — `PLANNED_VERTICALS` lists the roadmap and renders disabled, because a vertical with no seeds would silently fall back to a genre's thin `situations` list and build a corpus about the wrong thing. **Seeds are the ceiling on interview mode**: measured 2026-08-20, one seed yields ~150 usable openers before the distinct-5-gram ratio reaches the 0.90 floor, so `conversation` alone is roughly 525,000 conversations (~1.1 GB), a 1,520-seed pack is ~230,000 (~500 MB), and the full set is about 1.4M conversations (~3.2 GB). The cost line in the panel warns when a request exceeds what the selected topics can carry. Topic selection is remembered per vertical, not globally — a group id means nothing outside its own pack.
- Concurrency is a fixed set of powers of two, `providers.CONCURRENCY_CHOICES` = 2…256. `LOCAL_MAX_CONCURRENCY` was raised 4 → 256 on 2026-08-20: the old value protected a single small GPU but capped a 100k-conversation run at ~36 days on a 32-core / 275 GB box. Throughput is ~0.48 conversations/minute per worker.
- Distillation tab, interview mode (`#interviewPanel`, the default): two-pass generation and the only mode that clears the acceptance gate. Pass 1 writes USER turns; pass 2 sends each to the teacher as a real chat request and keeps the genuine reply; follow-ups chain to any depth. `teacher/interview.py` holds the length blend (brief 0.20 / normal 0.55 / thorough 0.25, chosen per TURN) and the cleanup; `teacher/interview_job.py` runs it and deliberately writes the **SynthJob on-disk contract** (`samples.jsonl` / `state.json` / `errors.jsonl` / `plan.json`) so `/teacher/synth/status`, `/stop`, `/samples` and `/teacher/authoring/build` all work on it unchanged — only `/teacher/interview/start` is new. Every conversation still passes `RecordGate`. Measured 2026-08-20: 256-379 B median assistant turn against 120 B for dialogue-scripting, 100% unique turns, gate PASS.
- Interview run visibility. Every teacher call in this mode goes through `interview.ask()`, which reports the start and the end of the call to an optional watcher; `interview_job.CallFeed` is that watcher and holds the last 40 calls, the open calls keyed by worker thread, and a 200-sample latency window. `GET /teacher/synth/calls?job_id=` serves it: `calls` (newest first, each with `kind` of `seed` / `answer` / `follow-up`, `id`, `sent`, `got`, byte counts and `ms`), `inflight` (same shape plus `elapsed_ms`), and `stats` (`calls`, `in_flight`, `per_min`, `p50_ms`, `p95_ms`, `reply_bytes`). **In memory on the running job only** — a run makes one call per turn plus one per follow-up, so 10,000 conversations at depth 3 is 50,000 calls and nothing that size belongs on disk; the route answers empty for a job that is not loaded. Displayed text is clipped to 700 bytes a side. The panel polls at 2 s and reruns the clock on open calls at 200 ms. Each call carries a SEND/RECEIVE split: `Client.complete(on_first_token=)` fires on the first content delta of the stream, so `wait_ms` is time to the first word and `ms - wait_ms` is the reply arriving. Providers that cannot stream never report it and read as sending for the call's whole life. A failed call keeps its row, its error and its latency (60 s of it reads as a timeout, 0.2 s as a refusal) and is excluded from the latency percentiles. The feed shows every OPEN call plus a short tail of finished ones (`IV_CALLS_DONE_SHOWN`); it is a monitor of the present, not a scrollback, and the panel scrolls within its own height. The per-call bar is log-scaled against the run's median call (`p50_ms`): half the track at typical, full track a decade above it, split between waiting and receiving by their real share. `state.json` carries the same counters as `call_stats` so calls made, calls failed and conversations salvaged survive a restart.
- Interview durability. A teacher failure part way through a conversation ends it at its last complete exchange instead of discarding it: a conversation costs `2*depth-1` calls and every one is paid for the moment it returns. Stop is one of those failures (the client raises `TeacherCancelled` from its cancel check), and the run loop drops only what is still QUEUED on stop, then keeps reading results, so a conversation in flight is written like any other; `/teacher/synth/kill` is the one path that abandons them. Pass 1 appends every opener to `openers.jsonl` as it arrives and a resume uses that pool before asking for anything new, minus the openers already answered (read back as the first user turn of each record in `samples.jsonl`); a torn last line is skipped, not fatal. The opener pass gives up after `FAILURE_ABORT_STREAK` consecutive failed batches instead of running its full round budget against a dead teacher, each round paying the client's own five retries. `samples.jsonl` is fsynced every `STATE_FLUSH_EVERY` records and at close, which bounds what a power cut can take.
- `GET /teacher/synth/errors?job_id=&limit=` tails `errors.jsonl` (default 12, max 100) so a failing run says what the teacher said. `POST /teacher/synth/kill` abandons calls already in flight, against `/stop`, which drops the queue but lets in-flight conversations finish and be written.
- Teacher HTTP failures are classified identically on both request paths in `teacher/client.py`. `_http_fatal` decides retry-or-raise from the status (401/403 -> `TeacherAuthError`, anything outside `RETRY_STATUS` -> `TeacherError`, both immediate and both carrying the first 300 bytes of the server's own error body); `_exhausted` classifies a run that used up its retries (429 -> `TeacherRateLimitError`, 5xx -> `TeacherUnavailableError`). The streaming path catches `HTTPError` ahead of `URLError`/`OSError`, which it subclasses, and honours `Retry-After` the same way. Every Distillation interview call takes that path: `interview.ask` always passes `cancel_check` and every provider but Anthropic declares `supports_stream`. Pinned by `tests/mri/test_teacher_client_errors.py`.
- Banned phrases are user-owned data. `RecordGate` counts which phrase matched (`stats()['banned_hits']`, top 12) rather than only that one did, and the Distillation tab shows the counts live and edits the list at `POST /teacher/authoring/banned` (`{"phrases": [...]}`, lower-cased, deduplicated, max 500 of 80 chars). An empty list is legal and bans nothing. Matching is whole-word and case-insensitive against assistant text, and a gate compiles its list at run start, so an edit applies to the next run.
- Distillation tab: owns `#authorPanel` and `#synthPanel`, which moved out of the Training tab 2026-08-20. Their visibility belongs to the tab's mode switch (`_distSetMode`, persisted to `vt:distillation:mode`) — `_trShowSynthPanel` / `_trShowAuthorPanel` no longer touch them, and `body.training-active` no longer hides them. Job rehydration runs from `_distOnTabActivated` on every activation, not from the training flow picker. Both Start buttons are bound through `_distGuardedStart`, never directly. **`/teacher/synth/status` reports `completed` as RECORDS (lines in `samples.jsonl`) while `state.json` counts CALLS — progress must use `calls_ok` / `calls_failed` / `calls_remaining`, never `completed`, or it reads past 100%.**
- Distillation tab, corpora list (`#distJobsList`): every directory under `veritate_mri/data/synth_jobs/` with its record count, bytes on disk, categories, and last write. `use` loads the row into the destination picker of the active mode, `rename` writes `label` into that job's `meta.json` via `/teacher/synth/rename` (capped at `JOB_LABEL_MAX`, blank clears back to the id), and `delete` calls `/teacher/synth/delete`. Both routes resolve the job id through `_job_dir`, which 404s anything outside the jobs root. `/teacher/synth/jobs` is the single source for all three destination pickers and this list; option text goes through `_distJobName`, so a renamed job shows its label everywhere. Controls on this tab carry no inline styling: one `.tab-body[data-tab="distillation"]` rule in `index.css` styles every select and input, and each mode's body is a numbered three-step sequence (`.dist-step`) ending in the build row.
- Distillation tab, running state: while a mode's job runs, `_distSetRunning(mode, running, what)` puts `is-running` on that mode's `.dist-work`, which hides every `.dist-config` block (both setup steps, the mode's explainer, the import fold) and shows `#<mode>RunBar` - one line naming the destination and the plan, plus the Stop button, which lives in that bar and nowhere else. Progress, warnings, reject counts, and live output stay visible; `#<mode>ProgressStep` is revealed on the first status poll and never hidden again. The running mode's card carries `has-run` so a run stays visible after switching modes. Every status poll calls `_distSetRunning` with the fresh `running` flag - a poll that skips it strands the form collapsed after the job stops. Training contention does not collapse anything: distilling during a run is supported and only `#distContentionNote` reports it.
- Distillation tab, job listing: `/teacher/synth/jobs` is the only reader of the job list. `_synthLoadJobs` fetches once, fills all three destination pickers through `_distFillJobPickers` (driven by the `DIST_JOB_PICKERS` table) and renders `#distJobsList`; no mode builds its own options. The list binds one delegated click listener keyed on `data-act`, never one per row. While a job runs its row is patched in place by `_distSyncJobRow` (record count only); the row is rebuilt exactly once, on the running-to-stopped edge. `_distSetRunning` memoises the last state pushed per mode in `distState.running` / `distState.runWhat`, so an unchanged poll writes no DOM.
- `_count_lines` counts non-blank lines incrementally, keyed by `(size, mtime_ns)` and capped at `LINE_COUNT_CACHE_MAX` entries. A file that only grew resumes from the byte offset the previous count covered, and only when `_ends_on_record_boundary` confirms that offset starts a fresh line - a writer caught mid-record would otherwise be counted twice. Measured 2026-08-20 on a 60.1 MB, 60 000-record `samples.jsonl`: 25.22 ms for a full re-read, 0.002 ms on a cache hit, 0.136 ms when 20 records were appended between polls. The delete route drops the job's entry. `/teacher/synth/jobs` reads this plus `meta.json` and two stats per job; it never builds the full status document.
- Training tab flows are `scratch`, `continue`, `rag`, `export`. `synth` and `author` became Distillation modes and were removed from `TRAIN_VALID_FLOWS`, `NO_PICKER`, `TEACHER_REQUIRED_FLOWS` and the flow labels; a stale `vt:training:flow` naming either now falls through to no flow. `#trainFilesPanel` was deleted with them - the corpora list on the Distillation tab owns job deletion.
- RAG train panel: corpus stem fixed to `rag_ui`; requires a configured teacher; one job at a time (409 on a second start).

## training

### the trainer

There is exactly one trainer: `veritate_mri/training/veritate_trainer.py`, ordinary tracked platform code. Sizes and tuned defaults are data in `veritate_mri/data/trainer_sizes.json` (34 sizes, 5m–1t): `shared_defaults` → the size's `defaults` → user settings, resolved identically by `readers/trainers.py` and the trainer so the offered set never drifts from the supported set. `default_size` in the same file prefills the Training form. Adding or retuning a size is a JSON edit, nothing else. No tunable is ever a literal in the trainer.

The dashboard builds the trainer's manifest record synthetically (`_native_record()` in `readers/trainers.py`); third-party trainer plugins may exist as `trainers/<id>/{manifest.json, trainer.py}` bundles discovered by `_walk()`, must import only `veritate_core.plugin`, and require explicit user permission to create. The prune panel's generated plugins are the one sanctioned way new `trainers/<id>/` dirs appear.

Plugin import surface (`veritate_core.plugin`): `save`, `paths`, `model`, `qat`, `hardware`, `multicorpus`, `oom_recovery`, `bench`, `mem_planner`, `mem_executor`, `get_teacher_client`. Nothing else. `hardware.resolve_precision` downgrades bf16 to fp32 on unsupported devices; forced-device fallback warns rather than raises.

### launching runs

Launch via `POST /trainers/run` with `id="native/trainer"` and `args` (never a hand-rolled launcher — that bypasses the runner, the heartbeat, and the env plumbing below). `trainer_runner.start()` enforces one training instance globally and sets `VERITATE_PLUGIN_ID`, `VERITATE_DEVICE`, and thread caps on spawn. `NEGATABLE_BOOL_FLAGS` emit `--no-<flag>`; other manifest-default-true booleans cannot be disabled through the API — raise batch instead of chasing act_ckpt.

Load-bearing launch facts:

- **`model_type` is mandatory and silently defaults to `language`.** Values: `language` | `code` | `statistical` | `other`. It is not a manifest field; it rides only the `VERITATE_MODEL_TYPE` env var set by `trainer_runner`. The dashboard also puts it on argv, where it is listed in the trainer's `SCHEMA_IGNORED_FLAGS` and does nothing. An absent value on a statistical model produces meaningless language-eval panels. To fix a mislaunched run, set `training_args.model_type` in the model's `config.json`; save reads it fresh each checkpoint and never overwrites it (`model_type` is deliberately excluded from `RUN_ARG_KEYS`).
- **`loss_mask`**: `require_loss_mask_decision` refuses to start on a ChatML-dense mix without an explicit `--loss_mask`. Forgetting it fails silently otherwise — loss falls, the run looks healthy, the model can't answer. Role-masked (assistant-only) loss is opt-in and costs no throughput.
- **Resume takes shape from the checkpoint weights, not from `--size`.** `shape_for_run` reads layers/hidden/ffn out of the `.pt` (adjusting the block count by `trunk_block_overhead()` for the patched trunks) and announces any field the preset disagrees with. `heads` is not recoverable — `qkv` is packed — so it still comes from the preset, which makes `--size` worth passing anyway. `--layers/--hidden/--ffn/--heads` on argv cannot set shape; a value that disagrees refuses the launch rather than being ignored. `apply_resume_overrides` reads `cfg["training_args"]`, which flat old configs lack. `total_steps` is absolute, not additional. An omitted `lr_schedule` on resume silently inherits the base's decayed `wsd` tail. Old `config.json` files must keep resuming; trainer changes stay backwards-compatible with them.
- **A resume that strands parameters refuses to start.** `load_resume_state` loads `strict=False` (QAT legitimately owns tensors its source lacks), so it reports every missing and unexpected tensor and, on a plain resume, raises rather than training a partly-random model. See failures.md 2026-08-12 for the launch this was written for.
- **`resolve_val_path()` follows the heaviest-weighted corpus**, not the first listed.
- **Val draws are deterministic per seed and shared across runs.** `make_data_loader` builds its own `np.random.RandomState(seed)`, so with the same seed the Nth val evaluation reads byte-identical windows in every run. Comparing two runs AT THE SAME STEP is therefore exact — same windows, only the weights differ. Comparing a single run against itself across steps is not: part of the step-to-step wobble is which windows that eval happened to draw, and the same bumps recur at the same steps in unrelated runs (wren1_0 and wren1_1 both spiked at step 1,000). Regress the trend or compare across runs; do not read a single val point as a change in the model.
- Never launch with `... | tee file`; it masks a crashed trainer's exit code.
- Verify the budget arithmetic before trusting any throughput number: `tok_per_s / step_rate == batch × seq × n_chunks`.

Pre-launch checklist (all mandatory): read the ledgers; no version-suffixed names; explicit flags for everything; low dose + heavy replay for capability SFT; baseline before the experiment arm; test the no-quality-loss speed levers on this box at this shape — batch sized to the machine (act_ckpt off when not memory-bound), `n_chunks` amortization, `torch.compile` only on a dense trunk (it crashes the hybrid trunk on MPS; see failures.md before chasing any lever); confirm the run sits near the measured throughput ceiling before spending the compute.

### save contract and storage

Every checkpoint save goes through `veritate_mri/training/save.py::save()`; every per-step CSV row through `append_train_row()` in the same module (header: `step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed`). No trainer writes `.pt`, runs dumps, or appends CSV directly. `save()` writes checkpoint + `config.json` + the full hook-dump suite into `models/<name>/hooks/step_<N>/` in one call, atomically (`.tmp` + `os.replace`). Callers: the trainer and the grounded-SFT script.

`models/<name>/` layout: `config.json`, `train.csv`, `checkpoints/step_<N>.pt`, `hooks/step_<N>/`, `veritate.bin`. Checkpoint retention prunes `.pt` files only — `hooks/step_<N>/` is never pruned; only resumability is dropped, not the research artifact. `n_params_total` in `config.json` is the honest size denominator, never the preset's nominal name.

Language-dump gating: `language` and `code` both get the full language dump set (fluency/reading/grammar/reasoning/concepts/writing_health/math/generation) plus the deep-eval suite; `NON_LANGUAGE_TYPES = {statistical, other}` in save.py skips them (the eval-deep route mirrors the same gate). Architecture probes (`probe`, `lens`, `classroom`, `surprise`, `quant_kl`) and the checkpoint itself always run.

Dump failures print `DUMP FAILED:` to the run log without aborting the run — grep for it after the first checkpoint of any new-variant run. New model variants must pass the dump suite at the real run shape (real seq, hidden, layers, real 12–45-byte prompts) before launch; a tiny-shape smoke passes while the real shape crashes.

### hooks and dumps

The dump artifact set is 13 artifacts under `hooks/step_<N>/` (probe, lens, classroom, grades, math, grammar, reasoning, concepts, surprise, quant_kl, writing_health, reading_comprehension, generation), producers in `checkpoint_probe.py`, deterministic `PROBE_SEEDS` sampling so step-to-step comparisons stay valid, each dump wrapped so one failure doesn't abort the rest.

Cost, measured on wren_base (200m hybrid, MPS, three consecutive checkpoints within 2s of each other): **~137s for the full suite**, of which `generation` 33s, `reading_comprehension` 31s, `writing_health` 24s, `reasoning` 20s, `math` 20s — that five is `save.HEAVY_DUMPS`. The remaining eight total ~9s, because they read weights and activations instead of generating text. `save.ALL_DUMPS` is the full label set and `save()` asserts it matches the dumps it actually runs.

`--hooks full|light|off` picks how much of that a run pays per checkpoint (`full` is the default, so behaviour never changes because the flag appeared); `--hooks_full_every N` promotes every Nth checkpoint back to the full suite, counted in checkpoints rather than steps. `light` is what makes dense checkpointing affordable — at `ckpt_every 250` over 8,000 steps the full suite costs over an hour of wall clock, `light` + every 4th costs ~1%. Use it when the reason for frequent checkpoints is crash recovery rather than probe resolution.

Field symmetry: every per-token frame the dashboard renders is emitted by both training-time and inference-time capture; adding a field touches both in the same commit and updates the TFRM field contract. Three capture paths: PyTorch forward hooks (~20–30 ms/token), engine `trace_record_t` (~3 ms/token), training-time dump via `model.hook_spec()`. `hook_spec()` requires shape scalars and `blocks[L].ff.up/.down` weight exposure; slot-stream trunks must implement `probe_columns(tokens)` (snapshot at the last live slot, not the last position) or global-block panels read zero. If a frame's size changes, bump the trace version in `veritate.h`.

Adding a hook: update the field contract first, emit on every path in the same commit, gate dashboard render on field presence.

### evals

Deep-eval suites (MMLU/HellaSwag/IFEval) run on demand per checkpoint. The `default` and `form` IFEval sets measure different things — 47% of `default` rules also require a correct answer, so obedience gains can score flat on it. The shipped IFEval is hand-written and not comparable to published benchmark numbers. Eval-set builders are seeded and reproducible; a rebuild changes the grading key, so old-checkpoint scores stop being comparable.

### memory planning and optimizers

- `mem_planner.plan_training_memory(...)`: pure arithmetic, four buckets (params, grads, Adam moments + master copy, coarse activations), escalation ladder none → checkpoint_activations → checkpoint+bf16_optimizer → checkpoint+page_optimizer_to_nvme → infeasible_reduce_batch_or_seq. Budget = `hardware.unified_memory_bytes() × USABLE_FRACTION (0.85)`. No seq² attention term — under-predicts past seq≈2k.
- `mem_executor.apply_plan(model, plan)`: idempotent activation-checkpoint wrap. `optimizer_offload=True` in a plan is data the trainer must act on — building plain AdamW for an offload tier defeats the plan and OOMs. `make_optimizer` returns PagedAdamW for offload tiers.
- `PagedAdamW`: NVMe-mmap moment storage, bitwise-standard decoupled AdamW math; `state_dict()` carries step counts + `state_dir` only, so checkpoints stay tiny and resume rebinds files in place. I/O-bound once state ≫ RAM; frees optimizer buckets only. Durability comes from checkpoint cadence, not live state files.
- `bench.run(...)`: measures real training steps, stops on a predictive budget check (unified-memory overshoot is SIGKILLed, not catchable). `max_batch` answers the memory question, `best_batch` the launch question — throughput is not monotonic in batch. Probe with the optimizer the run will use.
- Muon (`optimizer=muon`): 2-D non-embedding params → Muon, everything else → AdamW, one optimizer surface (`MuonAdamW`). A Muon-saved checkpoint cannot resume under AdamW. Muon is skipped when NVMe paging is active.
- SLM (`slm_ref` + `slm_keep`): selective loss on top-excess tokens, training loss only (validation stays unmasked). Reference must be canonical dense; adds ~20–30% step time. See failures.md before using at byte level.

### model variants

`trunk` values: `dense` (default) | `patched` | `hybrid` (patched + recurrent global mixer) | `looped` | `recurrent` | `memory` | `hybrid_moe` (hybrid + MoE global FFN). Constructor defaults keep old checkpoints loading `strict=True`.

- Patched: boundary rule — a byte is a boundary unless ASCII alphanumeric or UTF-8 continuation (0x80–0xBF); position 0 forced. Local enc → gather boundaries into `seq/PATCH_STRIDE` slots → global blocks → scatter → local dec → tied head.
- **`layers` counts GLOBAL blocks only on every patched-family trunk** (`patched`, `hybrid`, `hybrid_moe`, `hybrid_monarch`, `hybrid_pkm`, `hybrid_pkm_fire`, `looped`). `VeritatePatched` builds one flat `blocks` list of `N_LOCAL_ENC (2) + layers + N_LOCAL_DEC (2)`, so `--size 200m` (`layers=16`) writes **20** `blocks.N.*` entries. Anything that infers a layer count by counting checkpoint entries must subtract the overhead or it builds a model four blocks too deep — and `load_resume_state` uses `strict=False`, so that loads without an error (see failures.md 2026-08-12). `config.json` `shape.layers` records the constructor argument, not the entry count; the two legitimately differ.
- Recurrent: chunkwise-parallel training over 64-byte chunks (verified vs per-token recurrence to 4e-11). `state_rule`: `gla` (default) | `delta` | `pinned` (both non-default rules are falsified — failures.md). `state_carry=chunks` carries state across windows; `bptt_window` is the memory knob.
- Memory: fast weights are per-sequence state, never in the checkpoint; `forward()` resets per call; `forward_carry()` persists across windows. Not exportable.
- MoE (`hybrid_moe`): 8 routed quarter-width experts + 1 shared, top-2, aux-loss-free bias balancing plus sequence-wise aux loss; MPS-safe routing (iterative argmax, no sort/bool indexing, fp32 bookkeeping).

Exportability (`export.py`): `dense` → v9/v11/v12 int8; `hybrid` → v13 fp16/fp32/int8; everything else refuses early with `ValueError` naming the variant and serves through the PyTorch brain instead. Shape resolution falls back `shape` → `training_args` → top-level flat config. A non-canonical checkpoint pushed through the int8 writer would produce a silent-garbage bin — the refusal is the guard.

### model growth (IDEA 21)

`training/grow.py` also carries checkpoint-to-checkpoint function-preserving growth (beyond the trainer's mid-run FFN widen): `python -m training.grow <src.pt> --layers N --hidden H --ffn F --heads K --out <dst.pt>` (run from `veritate_mri/`) emits a larger checkpoint that computes the **same function** as the source — measured max |Δlogits| ≤ 5e-7 fp32 on width/heads/ffn growth and exactly 0 on depth growth (tests/training/test_grow_function_preserving.py, 9 tests). Mechanics: Net2Net duplicate-and-split with RMS-exact channel scaling (`sqrt(H'/(H·n))` on duplicated stream channels so RMSNorm is input-independent-exact), tied lm_head/tok_emb grown once with the tie kept, new heads silenced by zero-initialized `attn.proj` columns (input side copies an existing head), GLA mixer head_dim growth with q/v scaled to keep `o_norm` and the `D**-0.5` rescale exact, depth by appending a copy of the last global block with `attn.proj`+`ff.down` zeroed (residual identity). Duplicates are deliberately symmetric (exactness contract; no tie-breaking noise). Heads cannot grow at fixed hidden (head_dim would re-partition) — the tool refuses; head growth = hidden grows at fixed head_dim. Refuses looped trunks, `state_rule` delta/pinned, MoE/PKM/Monarch FFNs, and depth growth on non-hybrid patched trunks. Optimizer state is dropped — a grown checkpoint resumes with a fresh optimizer, so continue runs set `warmup_steps > 0`. Canonical 270M→~472M path: `200m` (16L/1024/4096/16h) → `400m` (24L/1280/5120/20h), head_dim 64 preserved.

Dashboard surface: `POST /models/grow` `{source, step?, target_size (trainer_sizes key or explicit shape), name}` runs the growth in a background worker; `GET /models/grow/status` reports phase/error/result; `GET /models/grow/options?source=X` enumerates valid targets server-side (≥ on all axes, > on one, `validate_growth`-approved) with exact per-trunk param counts. On success the new model is a normal continue-train candidate: `models/<name>/checkpoints/step_0.pt` + config cloned with shape/`grown_from`/description, `training_args.size` stamped to the target key (the trainer takes heads from the size preset on resume — explicit shapes whose heads match no key strict-fail on load; the UI only offers keys), `warmup_steps` preset (fresh optimizer). UI: "grow model" button in the Training tab's continue flow opens the grow modal (checkpoint + server-enumerated target + params before→after + the not-an-upgrade caveat). Tests: tests/mri/test_grow_route.py, tests/training/test_grow_function_preserving.py.

Context growth (`--seq` / `target_seq`, same tool and route): the learned `pos_emb` and `slot_pos_emb` tables are EXTENDED, never interpolated — old rows copy bit-exact, new rows copy the last learned row (keeps embedding norms and the late-position signal in-distribution; each new position trains independently). Logits on in-domain inputs (boundary count ≤ the old slot count) are bit-identical after seq growth (asserted `delta == 0.0`); inputs whose boundaries previously overflowed the slot cap were silently truncated by the source model and gain real capacity instead. Slot stride is derived from the checkpoint (pos rows ÷ slot rows); new seq must be a ≥ multiple of it. Config `shape.seq` + `training_args.seq` both updated so a continue run trains at the longer window; `export.py` reads seq from config, writes it (and slots) into the bin header, and cross-validates both tables — grown-seq models export to v13 with no engine change (engine memory scales with the longer context, inherently). UI: seq selector (current/2x/4x) + "keep size — grow context only" target option.

### MPS rules

Fixed or bucketed tensor shapes only (dynamic padding recompiles kernels; measured 23x slowdown). In model forward code avoid bool-tensor advanced indexing, int64 `sort`, and in-place writes on freshly indexed tensors — replace with `F.embedding` lookups, int32 sort, out-of-place masks. Prefer op patterns the canonical dense path already exercises. Keep tensors on device; every `.to(device)` round-trip stalls. 8-bit AdamW silently falls back to fp32 on MPS — do not chase it. Stability-smoke any test-time-learning module at real width, not toy width.

## corpus

### locations

Canonical: `data/corpus/` (`paths.CORPUS_ROOT`), holding `<stem>_train.bin` / `<stem>_val.bin` pairs and nothing else — no code, no manifests. Builders and downloads write here via `paths.corpus_train_path()` / `corpus_val_path()`. `corpus_search_dirs()` still resolves `trainers/corpus/` (`LEGACY_CORPUS_ROOT`) second for installs that have one; this install deleted it 2026-08-18.

### framing

Record separator across all modes: literal `<|endoftext|>` bytes. Chat mode is ChatML: `<|im_start|>{role}\n...<|im_end|>`; inference hard-stops on `<|im_end|>`. Agent mode is Hermes function calling: `<tool_call>{"name","arguments"}</tool_call>` / `<tool_response>...</tool_response>`, single-line schema-strict JSON. At byte level these markers are literal learned bytes, not reserved token ids. `models/<name>/config.json::capabilities` declares trained tiers (autocomplete < chat < agent, additive; `mark()` never regresses a trained tier; a missing key reads as autocomplete-only). Never invent a framing; these are the canonical ones.

### builders and library

All builders are deterministic from a fixed seed, write to `data/corpus/`, and split val before anything can straddle train/val.

- Code corpus: two-phase `stage` (network → jsonl cache) / `build` (offline, deterministic). Sources pinned (stack-edu, curated tarballs, oa-stackexchange, syntax-gated textbook docs). Per-document stable-hash val bucket (1-in-50), dedup before split.
- Authoring pipeline (teacher-driven): `corpus_spec.json` holds all recipe data; gates run first-failure-wins (JSON parse → schema → turn count → marker → NUL → length → dash policy → banned phrase → exact dup sha1 → opening-cap repeat → simhash near-dup). `simhash64` uses blake2b, not builtin `hash()` (process-salted). `build_sft_corpus.build()` always carves ≥1 conversation into val. A catalog entry writes straight into the shipped `corpus_catalog.json` — rebuilding a published stem breaks its recorded sha256.
- RAG corpus: held-out facts (`TEST_FRAC`) never enter the bins; the test set measures in-context copy on unseen facts. Teacher-required. Reruns with the same stem overwrite bins and the held-out set, and teacher sampling is unseeded, so reruns are not byte-identical.
- Curriculum corpus: generated child-concept corpus for 10M-class models. A procedurally generated corpus has an entropy floor set by generator complexity, not byte count — never use one to measure model capacity or growth.
- Bigram index: `<stem>_train_bigrams.npz` sidecars for the writing-health PMI probe; on-demand builds are capped and cover only the head of a large corpus — capped and `--all` (uncapped) PMI values are not comparable. Tokens are lowercase `[a-z][a-z']*` only.
- Corpus library: catalog merge order local → remote → user sources, later wins. Five install formats (raw_bytes, raw_bytes_zip, zip_bundle, hf_dataset, native); `coming_soon` gates unpublished tiers; `uninstall()` removes only `data/corpus/` copies. Size ladders were retired 2026-08-20: one entry per corpus, and `recommended_min_params` / `recommended_max_params` are null on every entry. The chat/agent/mcp ladders that motivated them were withdrawn as dead data in the same change (see acceptance gate below); behavior data scales with task diversity, not parameter count, and Chinchilla governs knowledge volume separately. Tiers over GitHub's 100MB limit ship as `zip_bundle` on Carpathian COS.

### acceptance gate

`tools/corpus_audit.py` scores a ChatML bin by **unique turns and unique content bytes, never by file size**, and both build routes (`/teacher/authoring/build`, `/teacher/synth/build_corpus`) return its report as `audit`. It exists because of the 2026-08-20 library audit: `chat_5gb` held 5.14 GB of bytes over **708 unique user turns** (376 KB of real text, one turn repeated 1,298,507 times) and passed every size-based check the platform had.

Four checks, calibrated against the corpora that survived that audit with headroom below the weakest passing one:

| check | floor / ceiling | set by |
| --- | --- | --- |
| unique user turns | ≥ 0.95 | cogito, 95.9% |
| unique content bytes | ≥ 0.85 | mixed_chat, 99.2% |
| median assistant turn | ≥ 200 B | veritate_sft, 242 B |
| artifacts per 1k assistant turns | ≤ 5.0 | mixed_chat, 4.5 |

Artifact patterns carry a **scope**: register tics (`ai_disclaimer`, `no_personal`, `canned_refusal`, `filler_opener`) are counted only inside ASSISTANT turns, because charging a corpus for a user turn that opens "Sure, I'm trying to decide between..." is a false positive that failed an otherwise clean corpus on 2026-08-20. Structural damage (mojibake, markup, truncation) stays scoped to the whole file.

The unique-content ratio is taken against **turn text, not file bytes** — ChatML markers are overhead, and scoring against file size wrongly penalises a corpus of many short unique turns. The artifact rate is unstable on small corpora (2 hits in 43 turns reads as 46/1k); treat it as meaningful only past a few thousand turns. `sft_idk` fails the gate deliberately — repeated refusal phrasings are its purpose. A corpus with a reason to repeat is a judgement call, not a bug; the gate reports, it does not block the build.

### sizing

Chinchilla in bytes: `bytes = batch × seq × n_chunks × total_steps`; divide by 4.55 (prose) or 4.12 (code) for token equivalents before sizing any run. "Correct output shape containing invented words" is the diagnostic signature of undertraining, not an instruction-tuning bug.

## engine

### formats

`.bin` versions v3–v13; compatibility is forward-only, and the full table lives with the loader (`veritate_engine/src/veritate.h`). Load-bearing points: v9 int8 non-MoE dense; v11 adds ternary (5-trit-per-byte base-3 packing, per-row `int32 gamma_q24`, 256-byte LUT decode) and MoE layout (top-1 routing only; per-expert up/down when `n_experts>1`, uniform per-tensor scale); v12 = MTP; v13 = hybrid trunk. v10 is rejected at load. Dense paths v3–v12 require head_dim 64; v13 is head_dim-generic. No Mamba-2 hot path; no speculative decode for v13; MoE with `n_experts>1` + non-INT8 quant refuses at load.

v13 specifics: strictly additive format with its own loader/forward (`hybrid.c`). fp32/fp16 bins dequantize weights exactly at load and compute fp32. int8 bins compute int8: weights stay int8 with per-output-row fp32 scales (export `_write_big`), activations quantize per-matvec (dynamic per-tensor maxabs → 127, `hybrid_quant_act`), int32 accumulate, dequant at output — every big matmul (qkv, proj, ffn up/down, recurrent gate) is int8-in/fp32-out while the residual stream, GLA state, conv ring, KV cache, and norms stay fp32. Scalar references (`hybrid_matvec_i8_scalar`, `hybrid_matmul_i8_scalar`) plus AVX2 (x86) and sdot (arm64) kernels behind CPUID dispatch; `VERITATE_HYBRID_SCALAR=1` forces scalar. Parity pinned by tests/engine/test_decode_parity.py (int8 SIMD vs scalar; fp16 vs int8 greedy 12/12 byte-identical, successes.md grd1). Measured on cardinal-01 (i7-9700T 800 MHz): wren-class 200M decodes 9.48 ms/byte p50 fp16 vs 6.63 int8 (1.43x); note the v13 scheme differs from v9/qat.py (per-row + dynamic scales vs per-tensor + fixed scale-32). ~28MB fixed decode state at h=768/seq=1024 regardless of conversation length; boundary table baked from Python `chr(b).isalnum()` at export, not re-derived in C. `VERITATE_PREFILL_BATCH` is per-arch, set by `c_engine.prefill_batch()` from a table keyed on (platform, machine): linux x86_64 batches at 32, every other tier stays sequential at 1. Apple Silicon must stay sequential (batching costs 14.4s against 1.15s on a cold 734-byte prompt, M3 Ultra). A tier earns an entry only with a measurement taken on that arch. Worker count auto-calibrates per box at load; parity holds at any thread count.

### kernels and dispatch

Three layers: `main.c` (CLI) / `dispatch.c` + `model.c` (forward, kernel composition) / per-backend kernels (scalar, x86_64 AVX2/VNNI/AVX-512, arm64 NEON SDOT — all wired behind runtime function-pointer dispatch selected once at load). Locked dispatch signatures: `matmul_int8`, `score_dot_v`, `softmax_rows`, `layernorm_i16_to_i8`; `attn_dot_inline`/`attn_hsum_inline` stay compile-time-bound (called millions of times per prefill). Correctness bar: bitwise oracle match to scalar for matmul, ≤1 LSB for attention/softmax/layernorm, decode bit-equivalence via `VERITATE_VERIFY_DECODE`. SIMD accumulates in the same 16-partial-sum order as scalar with `-ffp-contract=off`.

OS-specific primitives live behind the single shim in `veritate_engine/src/`; per-arch kernels never include OS headers. One binary per OS + major ISA; no fat binaries. Causal ablation (v8) zeros `ffn_neurons[L][pos][N]` pre-`ffn_down`; default `(-1,-1)` is a no-op.

Metal backend: bridge + `matmul_int8.metal` shader, detection and verify CLI exist, but `dispatch.c` still routes matmul to CPU — Metal is not in the forward path.

### state cache

`VERITATE_STATE_CACHE` (env-gated; unset = byte-identical behavior): persists six `hybrid_t` fields plus logits at snapshot time, keyed by two rolling hashes seeded from the model id, prefix-consistent so an extended prompt reuses its base's cache. When a trace is requested the scan ceiling caps at `n-1` so the last position is re-stepped for a real trace frame. Knobs: `VERITATE_STATE_CACHE_MB` (default 4096), `VERITATE_STATE_CACHE_LOG`. Eviction is oldest-mtime-first; a single oversized snapshot can still exceed the cap.

`paths.engine_binary_path()` is the single source of truth for the binary path. Env overrides: `VERITATE_ACT_BOOST`, `VERITATE_MAX_LAYERS`, `VERITATE_STATE_CACHE`.

## addons

Decode-time logit-bias plugins, both runtimes.

Python: `veritate_mri/inference/addons/<id>/{manifest.json, addon.py}` (both required or the addon is silently skipped). `Addon`: `__init__(**params)`, `reset()`, `observe(byte_int)`, `bias_logits(logits)` on a 1-D length-256 tensor. Manifest: `name`, `description`, `kind` (`"decoder"` only), `params` (dashboard renders one control per declared param; undeclared params forbidden). Chains compose in selection order; biases add. Endpoints: `GET /addons`, `GET /generate?...&addons=<id1>,<id2>`. Forbidden: importing `veritate_mri.*`/`veritate_engine.*`, writing outside the addon folder, retaining state across `reset()`.

C: vtable `reset`, `observe`, `bias_logits`, `destroy`; `logits` is `float*` length 256; blocked bytes get `-1.0e30f`. Chain seeded from `VERITATE_ADDONS=<csv>` env; `chat_traced` swaps the chain per request via a wire header token. Parity tolerance vs Python: ≤1e-5 per byte in float32 logit space.

## extensions

Third-party dashboard extensions (distinct from trainer plugins — never relabel one as the other). Layout: `extensions/{canonical,installed}/<id>/` bundles with `manifest.json` and an entry module exposing `register(app)`; discovery by `extensions/registry.py`. The marketplace installs from `extensions/catalog.json` (currently empty). Extensions call the REST API below; they get no private imports.

## api

### external (key-gated)

`/generate`, `/agent/stream`, `/v1/chat/completions`, `/v1/models` — bearer key from `mri_settings.json::api_key` (`PROTECTED_EXACT`/`PROTECTED_PREFIXES` in `routes/api_auth_routes.py`). SSE streaming; `mri_compact_frames`, `api_read_ahead_enabled`, `api_generate_ahead_enabled` settings modulate frames and typing-time behavior.

### internal (dashboard surface)

One route module per concern in `veritate_mri/routes/`. High-traffic groups: `/trainers` + `/trainers/run|stop|tune_defaults|sysprobe`, `/runs/*` (incl. timelines and eval-deep), `/models/*`, `/corpus/*` (incl. `/corpus/mix/plan`), `/settings` + `/settings/notices`, `/sys_metrics`, `/backends`, `/wiki*`, `/addons`, `/atlas/*` (concept/neuron/lifetime/circuit/concepts_inverted — pure derivations over hook dumps), `/teacher/*`, `/mesh/*` (machine-to-machine), `/models/git/*` (models repo sync), `/app/*` (updater). `/teacher/target_status` (GET) answers where distillation calls will land and whether that machine is already training: `{provider, model, kind, host, targets_this_machine, training_active, run, contention, contention_kind, reason}`. A hosted-API teacher never contends. A local teacher on this box contends while `trainer_runner.state()` reports running. A local teacher pointed at another box reports `training_active: null` — unknowable from here, and reporting `false` would be a guess dressed as a fact. The route never raises: a guard that 500s would block the start it is meant to advise.

The authoritative settings-key list is `DEFAULTS` in `veritate_mri/runtime/settings.py` — treat the code as the reference; notable keys beyond the obvious: `speculative_enabled`, `speculative_bytes`, `speculative_chunk_bytes`, `speculative_pause_ms`, `read_ahead_enabled`, `api_read_ahead_enabled`, `api_generate_ahead_enabled`, `corpus_compose_chunk_bytes`, `corpus_compose_val_ratio`, `corpus_compose_seed`, `trainer_sizes_path`, `corpus_mix_max_epochs`, `pytorch_load_mode`, `warm_models`, `device_preference`, `api_key`.

## settings reference

The Training-tab "learn more" links resolve to the sections below by slug.

### recipe

Preset bundles (`TRAIN_RECIPES` in `index.js`) that prefill the form; applying one overwrites the affected fields and nothing else. Every value remains individually editable after applying.

### optimizer

`adamw` (default) or `muon`. Muon routes 2-D non-embedding weights through Muon and the rest through AdamW behind one optimizer surface. A checkpoint saved under one optimizer does not resume under the other. Muon measured 1.60x fewer bytes to equal quality at 10M (see successes.md).

### trunk

Architecture selector: `dense` | `patched` | `hybrid` | `looped` | `recurrent` | `memory` | `hybrid_moe`. `dense` is the canonical exportable trunk; `hybrid` exports via the v13 engine format; the rest serve through PyTorch. `hybrid` composes boundary patching with a constant-state recurrent global mixer — the best measured quality/wall-clock trade (successes.md). `hybrid_moe` adds a mixture-of-experts global FFN (top-2 of 8 quarter-width experts + 1 shared).

### precision

`fp32` or `bf16`. bf16 halves activation memory and is the default where supported; unsupported devices resolve to fp32 with a warning, not an error.

### batch_size

Sequences per step. Size it to the box: memory ceiling from the planner, then pick the measured best-throughput batch (not the max that fits — throughput collapses once memory pressure bites). Moderate jumps only, or retune LR.

### seq

Context window in bytes. Directly costs throughput and activation memory; fixed shapes only on MPS.

### n_chunks

Chunks per optimizer step (gradient accumulation). Raises tokens/step at near-zero memory cost because activation memory is bounded by `bptt_window`, not chunk count. Fewer optimizer updates at a fixed token budget degrades convergence — don't maximize blindly.

### bptt_window

How many chunks gradients flow back through when state is carried. The real memory knob for recurrent/hybrid runs.

### base_lr

Peak learning rate, reached after warmup, then shaped by the schedule. Per-size tuned defaults come from `trainer_sizes.json`.

### min_lr

Floor the schedule decays to; the constant tail level for `wsd`.

### lr_schedule

`cosine` | `linear` | `constant` | `wsd` (warmup–stable–decay). On resume, an omitted schedule inherits the base run's decayed tail — state it explicitly.

### warmup_steps

Linear ramp from zero to `base_lr` before the schedule shape applies.

### weight_decay

Decoupled weight decay (AdamW-style) in both optimizer paths.

### grad_clip

Global grad-norm clip applied every step.

### label_smoothing

Accepted but currently inert: no loss path passes it to `cross_entropy`. Leave at 0.

## concepts

- **Byte-level model**: reads and writes raw bytes, vocab 256. Markers like `<|im_start|>` are learned byte sequences. ~4.55 bytes per prose token — convert before comparing budgets to token-based literature.
- **Runs and checkpoints**: a run appends to `train.csv` and saves `step_<N>` checkpoints; every checkpoint carries its full hook-dump suite for the dashboard. Resume continues the same model dir; `total_steps` is absolute.
- **Corpus**: `<stem>_train.bin`/`<stem>_val.bin` byte pairs; multicorpus mixes weight stems (`corpus_mix_max_epochs` caps repeats via water-fill; weights sum to exactly 1; deterministic).

## hardware tiers

Launcher-level tiers (distinct from engine kernel tiers): mac_arm, mac_intel, linux_x86, linux_arm, windows_x86 — each pins a Python range, torch wheel, and compute backend (`veritate.py::_detect_tier`, `VERITATE_TIER` env). mac_intel is pinned to `torch~=2.2` permanently (PyPI dropped Intel-macOS wheels after 2.2.2): no MPS, no compile, no efficient SDPA there. Supported arches for every change: macOS arm64, Linux x86/ARM, Windows x86 — degrading one to fix another is a regression.

## build notes

### build 1

Veritate 1.0.0. First public build: single-trainer platform, 34 sizes from one Training form, C engine formats v3–v13, in-app documentation. Nothing to migrate.
