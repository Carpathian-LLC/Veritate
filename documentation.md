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

- **Brain** (PyTorch): single global instance behind a lock. Three-way variant dispatch on `pos_emb` / `mtp.transforms.*` presence; consumers call contract methods (`forward` → `(logits, loss)`, `project_byte0`), never `isinstance` on a variant. CPU thread count autotuned per shape and cached. The requested step is resolved against disk at load (`readers/checkpoints.py::resolve_step`): a rolling prune deletes checkpoints under a live server, so a step cached at boot or held in a stale dropdown resolves to the newest on disk and the load reports the step it actually served. Lookahead drafting is CPU-only (`LOOKAHEAD_DEVICES`); byte-exactness is unproven on MPS.
- **C engine backend** (`c_engine.py`): subprocess per model. Warm pool (`cfg["C_WARM"]`): independent resident subprocesses, `warm_select` re-points without spawning, no automatic C-side eviction (the idle watcher only unloads PyTorch). Model discovery requires a `step_<N>.pt` placeholder even for bin-only deployments.
- **Decode cost is bimodal (hybrid trunk).** The model file carries a 256-entry `boundary[]` table (`hybrid.h`); when the byte fed in is a boundary byte (whitespace/punctuation) the step runs the GLA recurrent global-block stack, which a non-boundary step skips (`hybrid.c::hybrid_step`). Prefill amortizes one weight stream over every boundary byte in a chunk; single-byte decode cannot, so **every word-initial byte pays a full global-block weight stream**. Measured on cardinal-01 (i7-9700T 8c @ 800 MHz, wren1_3 fp16, greedy, 128 bytes): non-boundary p50 10.07 ms, boundary p50 50.11 ms, 24 of 127 bytes boundary = **54% of decode time**. The split is memory-bandwidth bound, so quantization attacks it directly: on identical weights (wren1_0) fp16 15.17 ms/byte vs int8 10.42 ms/byte = **1.46x**, greedy-byte-identical output. Serving int8 is the first lever on any bandwidth-limited box. Python-side cost across the whole pipe (frame read, parse, event dict, JSON, SSE, socket) is 0.02 ms/byte, 0.1% of generation: the transport is not the bottleneck and coalescing frames buys nothing.
- **Engine worker count** (`engine_threads` setting -> `VERITATE_HYBRID_THREADS`): 0 lets the engine calibrate per (box, model shape); a positive value pins it. The engine's ladder (1, 2, 4, ... cap) climbs while each rung beats the previous by `HYBRID_CALIB_KNEE` (13%) and stops at the first that does not, timing **non-boundary steps only** (`hybrid.c`). On a bandwidth-limited box that can stop a rung short: cardinal-01 measures 4 threads 16.58 ms/byte vs 8 threads 14.85 ms/byte, a 10.4% gain the knee rejects, and the pick is unstable across models on one box (`wren1_0` cached 8, `wren1_3` cached 4) because the test compares against the immediately previous rung and run-to-run noise straddles the threshold. Pin `engine_threads` when the calibrated pick leaves cores idle.
- **Read-ahead**: sends the open prefix (not the closed wire prompt) while the user types so the engine state cache can hit; on by default for dashboard and API. Only the C engine has a state cache; PyTorch ignores it.
- **Speculative prefetch** (generate-ahead): off by default; a real request always wins the engine (`take()` cancels the job); only `/generate` claims a buffer. A served/spent ratio below ~50% is a losing energy trade.
- **RAG injection** (`/generate` with a `rag` corpus path): BM25 top-1 by default (`RAG_K_DEFAULT`; multi-candidate injection is measured-worse at 200M). Passages land inside the final user turn of the ChatML prompt via `build_rag_prefix`, budgeted by `injection_budget(seq, prompt_bytes)`: the live model's window minus the prompt and a `REPLY_RESERVE_B` reserve, so injection never evicts the turn frame. A prompt already at the window injects nothing. Retriever hits carry whole chunks.
- **Streaming decode** (`/generate?fast=stream`, PyTorch backend, recurrent-mixer models with `slots % 64 == 0` i.e. seq a multiple of 256): unbounded-context generation over `forward_streaming`. Full windows commit into the carried per-block recurrent states (one forward each); only the current partial window is recomputed per byte, padded with a non-boundary byte to keep the slot count CHUNK-aligned. The prompt is never truncated — context beyond `seq` lives in the state. State commits only from full windows (a part-full window's conv tail carries padding-derived columns). Argmax-parity with `stream()` inside one window is pinned by tests/mri/test_stream_fast_streaming.py. Use with state-carry-adapted checkpoints (wren1_3+); a carry-off model's state arrives ~0 and the walk degrades to window-local context. **Persisted conversations**: add `state_id=<[A-Za-z0-9._-]{1,64}>` (loopback only) and the carried states + pending buffer save to `data/stream_states/<id>.pt` after every call — the next call sends ONLY the new bytes and continues byte-exactly (split-call parity pinned by tests). `state_reset=1` starts the id over. A state is bound to the exact checkpoint that wrote it; a mismatch errors instead of silently degrading.
- **Experience log**: every completed serving exchange appends one JSONL record — base64 prompt/output bytes, model, params — to `data/experience/YYYYMMDD.jsonl` (`inference/experience.py`). Both surfaces record: `/generate` on either backend, and every local completion served through `/v1/chat/completions` and `/v1/chat/mri` (buffered, streamed, and MRI paths share the single chokepoint `hybrid_routes.py::_local_events`, which stop-truncates and records; its records carry `meta.route = "v1"`). Remote completions (teacher provider, public cloud model) are not the box's own experience and are not recorded. The `model` field is the model DIR name (C engine: the bin path's parent; pytorch: `BRAIN_MODEL`) — sleep attributes exchanges by it. Records from before 2026-08-23 carry an artifact basename instead (`veritate.bin` / `step_N.pt`); sleep still attributes those when exactly one model owns that basename, otherwise they belong to nobody and are consolidated by nobody. The replay substrate for sleep consolidation (ideas.md IDEA 20 T3): the model trains on its own thought and actions. RAG-injected prompts are recorded as injected (what the model actually saw). Partial replies on client disconnect still record. `VERITATE_EXPERIENCE_LOG=0` disables; 512 MB/day cap; never breaks serving (all failures degrade to a warn). `tools/build_experience_corpus.py` turns the log into `experience_{train,val}.bin` for sleep consolidation (dedupe, min-reply filter; rehearsal = the corpus mixer weighting the model's own base corpus alongside it). The train/val split runs on BYTES: an exchange goes to val whenever val holds less than `val_frac` of what has been written, so val stays a sample spread across the window. `min_val_bytes` raises a floor under val, filled from the oldest exchanges first; the sleep controller passes the trainer's draw window so a small night cannot produce an empty val bin. Raw transcripts alone do not bind facts (failures.md 2026-08-21 m2), so `tools/extract_facts.py` is the extraction pre-pass of the tell-it-once loop: rule-based, precision-first mining of declarative residence/occupation/guarded-copula facts from user turns and assistant restatements into the `build_fact_sft` schema (`--days`/`--model` window; the record `model` field may be a bin filename — matching is substring on the extension-stripped basename; negations, questions, hedges, and hypotheticals reject; repeats dedupe; a subject+relation restated with a new object keeps the newest and marks `revised`; `--report` prints what was kept and what was blocked and why). `build_experience_corpus --facts` runs that extraction over the same window and emits `{stem}_fact_sft_{train,val}.bin` plus an auditable `{stem}_facts.json` alongside the raw bins, off by default; the sleep launch chooses the bin mix. Closed-loop acceptance: the m2 chat renderings of the 50 e4 gold facts recover at 100% precision / 100% recall (tests/mri/test_extract_facts.py, floors 90%/98%).
- **Sleep controller** (`training/sleep.py`, routes `/sleep` GET, `/sleep/wake` POST, `/sleep/now` POST): per-model sleep. Models opt in via the `sleep_models` list (any model with a checkpoint); an install's old single `sleep_model` string migrates to a one-model enrollment automatically (`runtime/settings.py::_ensure_settings`). Enrolled models take turns on the one trainer: when serving has been idle `sleep_idle_min` minutes (default 2 — short because preemption makes it safe; the gate only keeps a run from starting between two messages of one exchange, and turning `sleep_preempt` off makes it serving's only protection) and no trainer is running, the enrolled model with the most pending own-exchanges above `sleep_min_exchanges` sleeps; the next model waits until the run finalizes. Own-conversations-only: each model consolidates only exchanges its own serving produced — experience records resolve to a model dir by exact name, or for pre-2026-08-23 records by uniquely-owned artifact basename; unresolvable records are consolidated by nobody. The corpus build writes a per-model filtered view of the experience log under `data/sleep/filtered/` and runs `tools/build_experience_corpus.py` over it (`sleep_use_extraction`, default on, additionally mines declarative facts from the same window through `tools/extract_facts.py` and renders them via `build_fact_sft` into `experience_fact_sft_{train,val}.bin`; `sleep_study_paths`, a list of source trees or documents, renders those through `tools/build_study_corpus.py` into `study_{train,val}.bin`. Raw transcripts are NOT the default mix any more: `sleep_corpus` defaults to `experience_fact_sft:0.75,mixed_chat:0.25`, because consolidating raw experience-log transcripts is falsified — 0/50 closed-book with degraded val, failures.md 2026-08-21 m2 — while the same facts in drilled study form score 45/50. Batch sizing and the too-small gate measure the heaviest mix member the controller actually builds, not the raw experience bins). Dose scales with that model's use: `steps = new_own_exchanges × sleep_steps_per_exchange` clamped to `[sleep_min_steps, sleep_max_steps]`. That clamp only bounds the dose; the run FINDS it. `sleep_stop_on_val_rise` (default 0.02) is handed to the trainer as `--stop_on_val_rise`; an armed resume scores the starting weights on the yardstick before its first step (a `val` row at the resume step, lr 0), and the run checkpoints and stops when two consecutive later readings exceed that start by more than the fraction, naming the best checkpoint in the log. Two readings, not one: fitting a drill corpus lifts the yardstick before replay brings it back (exp_fastsleep_0902, 2026-09-02: +12.0% at step 20, +1.9% at step 40 with the first facts bound); the price is one more checkpoint interval of a run that really is damaged, which the publish gate still holds back. A consolidation run has no useful fixed length: measured on wren1_9 (2026-08-25, lr 2e-4 over whole-function study chunks) val improved to 0.9690 at step 10 and rose to 1.0648 by step 20, so a run given 120 steps stopped itself at 20 (~34 min). The comparison is against the start rather than the previous reading because a drift where each step sits inside tolerance is exactly how wren1_3 walked +1.8% across five runs with nothing tripping, and against the start rather than the run's best because a run can improve on its start throughout while wobbling above its own best (wren1_12, successes.md 2026-08-26). 0 disables the rule and restores fixed-length runs. Consolidating at a rate low enough to be harmless is what forced the old ~129-epoch dose (13.8 h on cardinal against this feature's own 10-20 min/night design target); `training/fuse.py` is the other half of that trade, interpolating a consolidated checkpoint back toward its pre-sleep weights (theta <- alpha*theta_ft + (1-alpha)*theta_prev, CLI `tools/fuse_checkpoints.py`) so damage is bounded after the fact instead of avoided by an unusable learning rate. The recipe is the model's own `config.json` `training_args` with only the sleep levers overridden (constant `sleep_lr`, no warmup, `sleep_corpus` mix, assistant loss mask, `ckpt_every = sleep_ckpt_every`, `sleep_optimizer` (default `adamw`: the run starts with a fresh optimizer state regardless, and Muon's Newton-Schulz phase is 20.3 s of a 107.5 s step on cardinal at batch 7 against 1.2 s for AdamW, 2026-09-02), `sleep_freeze_blocks` (default 0; N reaches the run as `--freeze_blocks N`, see the `freeze_blocks` setting: on cardinal 15 of 20 blocks frozen halves the step), and `log_every = 1` because a dose can be shorter than a pretrain recipe's logging interval, which would leave the run with no `train.csv` row at all, and `hooks = off` because the checkpoint dump suite generates text in eager PyTorch to trend a research run across many checkpoints while every sleep checkpoint but the last is deleted when the run ends — ~137 s a checkpoint on a Mac Studio, and long enough on cardinal to stall a whole step). `sleep_state_carry` pins the run's recurrent-state regime (`chunks` or `off`); empty, the run inherits `state_carry` from `training_args`, which `fork.py` and `grow_routes.py` copy forward without the producing run rewriting it, so a grown model can carry an ancestor's regime (wren2 trains with `--state_carry chunks` while its config reads `off`). Consolidating an unbounded-context model with the carry off trains it in a regime it is never served in. When a run ends having gained steps, `finalize()` first gates on quality: `regressed()` compares the run's LAST val row to the BEST that model has ever scored (`val_best` in sleep state, raised only by runs that actually published), and a rise beyond `sleep_val_tolerance` (default 0.02, fractional) holds the checkpoint back from serving — the .pt survives and can be promoted by hand through `POST /export/<name>`, but a model walking the wrong way must not become the source of its own next training set. The high-water mark, not the previous run, is the baseline: run to run self-training degrades a model a little each round and every round sits inside the tolerance while the total walks up (wren1_3 drifted +1.8% across five runs with no single run above 0.5%), so the tolerance has to bound how far the model may ever fall from its best rather than how far it may fall per run. Comparing across runs is sound only because `sleep_yardstick` (default `mixed_chat`) pins `val_bin` to one fixed corpus: left to the mix, validation follows its heaviest member and the experience bins are rebuilt every launch, so each run would score different data. With no history the comparison falls back to within the run, led by the starting-weights reading the armed trainer logs at the resume step with lr 0 (a previous run's final row at that step carries a real lr and is not the reference). The gate FAILS CLOSED: a run that logged no val row, or logged exactly one with no history to compare it against, is held rather than published, because absence of evidence is not evidence of safety — three degraded exports reached the served model through the old fail-open path on 2026-08-24 (`val_first` null, `val_last` null, `served` true). The gate is a guardrail against collapse, not a quality optimizer. Otherwise `finalize()` calls `publish(model)`: it re-exports the newest checkpoint over the model's `veritate.bin` so the box serves what it just consolidated, then fires the publish hook (`set_publish_hook`, registered by `app.py` as `backends_routes.reload_bin`) to respawn any C engine serving that model — the engine reads a bin into memory and closes the file, so the swap needs no lock but a live subprocess holds the pre-sleep weights until it restarts, and the hook respawns rather than merely closing because `/generate` errors on a missing subprocess. Only a model that already serves a bin is published (a PyTorch-only model must not grow an engine artifact because it slept) and the dtype is read back off the bin in place (`readers/bin.py::weight_dtype`), so an int8 box stays int8. Every bin write in `training/export.py` now goes through `_atomic_bin`: sibling temp, fsync, `os.replace`. A `veritate.bin` opened `"wb"` is truncated before the first tensor lands, so an export that runs out of memory or disk halfway used to leave the box with nothing to serve; a failed publish now leaves the previous weights serving, is logged, and does not fail the sleep — the `awake` history event carries `served`, `held`, `val_first` and `val_last`. Dense sleep checkpoints make an early wake cheap; when a sleep run ends, its intermediates are deleted and older sleep finals thin to `sleep_keep_finals` per model — checkpoints not created by sleep are never touched. `/sleep/wake` and `/sleep/now` take a `model` (JSON body or query); omitted, the only enrolled model is assumed, and with zero or several enrolled the request is a 400. `/sleep/now` skips only the idle timer. Per-model state (last sleep, in-flight run, surviving finals, cooldown) lives keyed by model in `data/sleep/state.json` (the flat pre-per-model file migrates on load); every state change (fell asleep / woken / awake / failed) appends to `data/sleep/history.jsonl` with the model name, served newest-first in the `/sleep` payload's `history`. `/sleep` returns `models`: one row per enrolled model (`state`, `pending_exchanges`, `last_sleep_ts`, `sleeps_in_s`, `cooldown_s`, `finals`) plus the global `state`/`run`/`history`/`activity_by_hour`. Watcher thread ticks every 60 s (skipped in `--minimal`). Off until `sleep_enabled` is set and `sleep_models` enrolls a model. UI: the gen-bar chip is the one-line summary (single sleeper countdown, or `N models enrolled · <model> sleeping x/y`) with a wake button; the sleep review box below the Generation chat shows one compact row per enrolled model with its own sleep-now / wake button, plus the global usage-by-hour ledger and event history (box stays visible while disabled so history remains reviewable). **Preemption (weak hardware).** A sleep child sized for the training box takes every core. Measured on cardinal-01 (i7-9700T 8c, at the 800 MHz firmware clamp that box carried until 2026-08-24): an unyielding sleep run costs a served request 2.5-3x throughput (17.8 -> 44.6-54.6 ms/byte) and ~200x first-byte latency (13 ms -> 2.9 s), because the trainer child holds 7 of 8 cores. The ratio is what generalizes, not the absolute numbers: the same box at its rated 2.0 GHz base serves 8.7 ms/byte and sleeps at 72-87 s a step. With `sleep_preempt` on (default), every generation marks `runtime/serving.py` and the controller suspends the trainer child for the duration of the request, resuming it after `sleep_resume_s` of quiet; a suspend preserves all process state, so no step work is lost and only wall time stretches. Serving under an active sleep run then measures 18.1 ms/byte at 12 ms first byte, indistinguishable from an idle box. The watcher polls at `WATCH_PAUSED_S` for the resume check and runs its full pass on the `WATCH_EVERY_S` cadence, because a suspend can land at any point inside a watch period. `sleep_reserve_cores` keeps cores off the child's BLAS budget and `sleep_nice` deprioritizes it (both via the `_cpu_budget` / `_nice` run modifiers, stripped before argv), covering the window before a suspend lands. The sleep batch is sized to two bounds, whichever is tighter. The DATA bound is `min(remembered_recipe_batch, train_bytes // draw_window)`, where the recipe cap is the LARGEST batch that model's recipe has ever declared, recorded into sleep state at launch before the run stamps its own over it — the trainer writes its launch args back into `config.json` at every checkpoint, so read naively the cap ratchets down to whatever the weakest box last used and never recovers when the box gets faster (cardinal 2026-08-24: unclamped from 800 MHz to 2 GHz and stayed pinned at the 4 its clamped sleeps had written over the original 48). The bound matters because a step draws `batch × seq × n_chunks` bytes and a pretrain recipe's batch is set against a corpus thousands of times larger than a night of conversation. The BOX bound is `prev_batch × sleep_step_seconds / prev_step_s`, from what the model's last sleep on this machine actually cost: `finalize()` records `step_s` and `step_batch` into that model's sleep state (not read back off `train.csv`, because a model dir travels between machines and its rows carry the throughput of whichever one wrote them), and a model that has never slept here takes the data bound alone. `sleep_batch_size` pins the batch and overrides both. `eval_iters` is sized the same way — `min(recipe_eval_iters, val_bytes // (batch × draw_window))`, never below one — because a recipe's count re-measures a night's val split dozens of times for one number (475 s, three times a training step, on cardinal). Requires psutil for suspend/resume and niceness; without it sleep runs unyielding and logs a warning. Launch guards, each from a measured failure (cardinal 2026-08-20): save()-stamped bookkeeping keys (`corpus_bytes`, `corpus_sha256`, `output_dir`) are stripped from the recipe before launch; both experience bins must reach one draw window (`seq × n_chunks + 2` bytes), and the build is given that window as its val floor, so the gate binds on total own-conversation bytes (about `2 × window`) rather than on the val stride; the resume step is the latest checkpoint on disk, never config.json `step` (a model with no `.pt` cannot sleep); a sleep that gains no steps records a `failed` event and a 60-min cooldown for that model only, but a run whose `train.csv` advanced and was simply woken between checkpoints records `lost` with `steps_lost` and takes NO cooldown: the cooldown exists to stop the watcher retry-storming a launch that cannot work, and a run that was training is not that. `ckpt_every` is `min(sleep_ckpt_every, sleep_ckpt_seconds / measured step_s)` once the model has slept on this box, because a step count is seconds on a training box and minutes on a weak one (cardinal: 25 steps is 69 minutes, and waking inside that window threw all of it away). The step count stays the ceiling, so a fast box is unaffected. Measured constraint (cardinal-01 i7-9700T 8c, 200M hybrid, seq1024×4, log-sized batch 4): a sleep step costs **72-87 s at 189-227 tok/s** at the chip's rated 2.0 GHz base, and **166 s at 99 tok/s** under the 800 MHz firmware clamp that box carried until 2026-08-24, both at 675-736% CPU and ~5.7 GB RSS. At the recipe's batch 48 under the clamp a step cost ≥27 min and no step ever completed. On-box `.pt`→bin export is 9.8 s. Tests: tests/training/test_sleep_controller.py, tests/mri/test_sleep_routes.py.
- **Repetition guard**: `rep_window` / `rep_penalty` / `no_repeat_ngram` on `/generate` and `/prefetch`. When the caller sends neither, chat-framed and RAG prompts default the hard ban ON (`no_repeat_ngram` 16, `rep_window` 256) and plain completion prompts default OFF; explicit params always win. The soft penalty always defaults off. Grading of SFT checkpoints stays bare-greedy; the guard is a serving default, never a measurement setting.
- **OpenAI-compatible serving** (`/v1/chat/completions`, `/v1/chat/mri`, `hybrid_routes.py`): shared ChatML framing and model routing (local model on either engine, teacher provider, or the public cloud model). The public model is selected only by its own name (`cloud`); a `model` that is neither `cloud`, a teacher id, nor a local model with a checkpoint is a 404 (`code: model_not_found`), never a fallback to the cloud, so a typo or an empty model dir cannot send a conversation off-box. `mri:true` is an opt-in telemetry flag and never changes sampling. Local completions are recorded to the experience log, so API traffic is sleep-consolidatable and moves the sleep controller's idle clock. The chat UI was extracted 2026-08-20 to a separate project (see `CHAT_HANDOFF.md`); the Generation tab is the conversational surface.
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
- Interview run visibility. Every teacher call in this mode goes through `interview.ask()`, which reports the start and the end of the call to an optional watcher; `interview_job.CallFeed` is that watcher and holds the last 40 calls, the open calls keyed by worker thread, and a 200-sample latency window. `GET /teacher/synth/calls?job_id=` serves it: `calls` (newest first, each with `kind` of `seed` / `answer` / `follow-up`, `id`, `sent`, `got`, byte counts and `ms`), `inflight` (same shape plus `elapsed_ms`), and `stats` (`calls`, `in_flight`, `per_min`, `p50_ms`, `p95_ms`, `reply_bytes`). **In memory on the running job only** — a run makes one call per turn plus one per follow-up, so 10,000 conversations at depth 3 is 50,000 calls and nothing that size belongs on disk; the route answers empty for a job that is not loaded. Displayed text is clipped to 700 bytes a side. The panel polls at 2 s and reruns the clock on open calls at 200 ms. Each call carries a SEND/RECEIVE split: `Client.complete(on_first_token=)` fires on the first content delta of the stream, so `wait_ms` is time to the first word and `ms - wait_ms` is the reply arriving. Providers that cannot stream never report it and read as sending for the call's whole life. A failed call keeps its row, its error and its latency (60 s of it reads as a timeout, 0.2 s as a refusal) and is excluded from the latency percentiles. The feed shows every OPEN call plus a short tail of finished ones (`IV_CALLS_DONE_SHOWN`); it is a monitor of the present, not a scrollback, and the panel scrolls within its own height. The per-call bar is log-scaled against the run's median call (`p50_ms`): half the track at typical, full track a decade above it, split between waiting and receiving by their real share. `state.json` carries the same counters as `call_stats` so calls made, calls failed and conversations salvaged survive a restart, plus `eta_hours`: hours of wall clock left at the measured median call, `remaining × (2*depth-1) × p50_ms / max_concurrency`. A job is sized in conversations but paid for in calls, so a target that reads as reasonable can be a week — 30,250 conversations at depth 3 against a 65 s median is 171 hours at concurrency 16. Shown in the interview stats row as `estimated time left`; it reads `measuring...` until the first calls return.
- Interview depth vs the genre turn bound. A conversation of `depth` replies is `2*depth` turns. Genre bounds are written for scripted dialogue (`conversation` allows 4-8), so `/teacher/interview/start` widens the selected genres' `max_turns` to at least `2*depth` in the run's own spec copy, the same way it overrides `ngram_distinct_floor`. Without it every complete conversation past depth 4 was rejected as `turn count out of range` after paying for all `2*depth-1` calls. `min_turns` is untouched: a salvaged conversation shorter than the floor is still not a record.
- Interview durability. A teacher failure part way through a conversation ends it at its last complete exchange instead of discarding it: a conversation costs `2*depth-1` calls and every one is paid for the moment it returns. A failure with NOTHING complete to keep raises instead, so the run records the teacher's own error (`TeacherUnavailableError: upstream unavailable: 500 ...`) rather than a bare `empty conversation` — swallowing it discarded the only text that said which endpoint was down. Stop is one of those failures (the client raises `TeacherCancelled` from its cancel check), and the run loop drops only what is still QUEUED on stop, then keeps reading results, so a conversation in flight is written like any other; `/teacher/synth/kill` is the one path that abandons them. A cancelled conversation is recorded as `stopped` and does NOT advance `consec_fail`: `TeacherCancelled` subclasses `TeacherError`, so counting it walked one Stop into `FAILURE_ABORT_STREAK` and reported it as a dead teacher. Pass 1 appends every opener to `openers.jsonl` as it arrives and a resume uses that pool before asking for anything new, minus the openers already answered (read back as the first user turn of each record in `samples.jsonl`); a torn last line is skipped, not fatal. Pass 1 runs `max_concurrency` batches at a time against the same client — serially it was the whole cost of a large run, and the dry and failure streaks are counted per batch in submission order within a wave. The opener pass gives up after `FAILURE_ABORT_STREAK` consecutive failed batches instead of running its full round budget against a dead teacher, each round paying the client's own five retries, and that give-up sets the job's stop flag: pass 2 never queues against an endpoint pass 1 already found dead. `samples.jsonl` is fsynced every `STATE_FLUSH_EVERY` records and at close, which bounds what a power cut can take.
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

Launch via `POST /trainers/run` with `id="native/trainer"` and `args` (never a hand-rolled launcher — that bypasses the runner, the heartbeat, and the env plumbing below). `trainer_runner.start()` enforces one training instance globally (a launch is refused while a stopped child is still exiting, and a child's late exit never stamps a newer run's state: `/trainers/stop` marks the state stopped the instant it signals, the process exits later) and sets `VERITATE_PLUGIN_ID`, `VERITATE_DEVICE`, and thread caps on spawn. `NEGATABLE_BOOL_FLAGS` emit `--no-<flag>`; other manifest-default-true booleans cannot be disabled through the API — raise batch instead of chasing act_ckpt.

Load-bearing launch facts:

- **`model_type` is mandatory and silently defaults to `language`.** Values: `language` | `code` | `statistical` | `other`. It is not a manifest field; it rides only the `VERITATE_MODEL_TYPE` env var set by `trainer_runner`. The dashboard also puts it on argv, where it is listed in the trainer's `SCHEMA_IGNORED_FLAGS` and does nothing. An absent value on a statistical model produces meaningless language-eval panels. To fix a mislaunched run, set `training_args.model_type` in the model's `config.json`; save reads it fresh each checkpoint and never overwrites it (`model_type` is deliberately excluded from `RUN_ARG_KEYS`).
- **`loss_mask`**: `require_loss_mask_decision` refuses to start on a ChatML-dense mix without an explicit `--loss_mask`. Forgetting it fails silently otherwise — loss falls, the run looks healthy, the model can't answer. Role-masked (assistant-only) loss is opt-in and costs no throughput.
- **Resume takes shape from the checkpoint weights, not from `--size`.** `shape_for_run` reads layers/hidden/ffn out of the `.pt` (adjusting the block count by `trunk_block_overhead()` for the patched trunks) and announces any field the preset disagrees with. `heads` is not recoverable — `qkv` is packed — so it still comes from the preset, which makes `--size` worth passing anyway. `--layers/--hidden/--ffn/--heads` on argv cannot set shape; a value that disagrees refuses the launch rather than being ignored. `apply_resume_overrides` reads `cfg["training_args"]`, which flat old configs lack. `total_steps` is absolute, not additional. An omitted `lr_schedule` on resume silently inherits the base's decayed `wsd` tail. Old `config.json` files must keep resuming; trainer changes stay backwards-compatible with them.
- **A resume that strands parameters refuses to start.** `load_resume_state` loads `strict=False` (QAT legitimately owns tensors its source lacks), so it reports every missing and unexpected tensor and, on a plain resume, raises rather than training a partly-random model. See failures.md 2026-08-12 for the launch this was written for.
- **`resolve_val_path()` follows the heaviest-weighted corpus**, not the first listed. The `val_bin` flag overrides it with one named corpus regardless of the mix, so a run whose mix moves can still be compared to the run before it (sleep pins it via `sleep_yardstick`). An override naming a corpus with no `_val.bin` on the box warns and trains blind rather than failing.
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

- `mem_planner.plan_training_memory(...)`: pure arithmetic, four buckets (params, grads, Adam moments + master copy, coarse activations), escalation ladder none → checkpoint_activations → checkpoint+bf16_optimizer → checkpoint+page_optimizer_to_nvme → infeasible_reduce_batch_or_seq. Budget = min(`hardware.unified_memory_bytes() × USABLE_FRACTION (0.85)`, `hardware.available_memory_bytes()`): a box already holding served engines or another job plans against what is actually free, because a suspended sleep child keeps its RSS (cardinal 2026-09-02: 23 GB total, 12.7 GB available with the dashboard, five warm engines and a neighbour's download resident). No seq² attention term — under-predicts past seq≈2k, and `_activation_bytes` models a dense trunk: on the hybrid trunk it over-predicts ~7x (4.4 MB/token predicted vs 0.63 MB/token measured, fp32, 2026-09-02), which is what keeps activation checkpointing on for CPU consolidation runs that would fit without it.
- `mem_executor.apply_plan(model, plan)`: idempotent activation-checkpoint wrap. `optimizer_offload=True` in a plan is data the trainer must act on — building plain AdamW for an offload tier defeats the plan and OOMs. `make_optimizer` returns PagedAdamW for offload tiers.
- `PagedAdamW`: NVMe-mmap moment storage, bitwise-standard decoupled AdamW math; `state_dict()` carries step counts + `state_dir` only, so checkpoints stay tiny and resume rebinds files in place. I/O-bound once state ≫ RAM; frees optimizer buckets only. Durability comes from checkpoint cadence, not live state files.
- `bench.run(...)`: measures real training steps, stops on a predictive budget check (unified-memory overshoot is SIGKILLed, not catchable). `max_batch` answers the memory question, `best_batch` the launch question — throughput is not monotonic in batch. Probe with the optimizer the run will use.
- Muon (`optimizer=muon`): 2-D non-embedding params → Muon, everything else → AdamW, one optimizer surface (`MuonAdamW`). A Muon-saved checkpoint resumes under AdamW with fresh optimizer state (restore skipped, logged); sleep does this by default (`sleep_optimizer`). Muon is skipped when NVMe paging is active. Newton-Schulz orthogonalization runs in the dtype `hardware.bf16_supported(device)` allows: bf16 where the device accelerates it, fp32 where it does not. `torch.optim.Muon` hardcodes bf16 and exposes no dtype, so a device without bf16 acceleration takes the vendored Muon instead — on a CPU the bf16 `addmm` has no fast path and orthogonalizing one 1024x4096 weight costs 203.9 s on one core against 0.775 s across seven in fp32 (i7-9700T, 2026-08-24). The dtype is an instance attribute, never a param-group default: in `defaults` it serializes into `state_dict()` and `load_state_dict` would restore the writing box's dtype onto this one.
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
- Fact chats (`tools/build_fact_chats.py`, stem `fact_chat`): the same atomic facts as natural multi-turn conversations, told by the user and echoed by the assistant among distractor turns (the falsified raw-transcript consolidation arm, failures.md 2026-08-21). `--recall` renders in-context recall conversations instead: the fact is told, small talk follows, the user asks for it back and the assistant answers from the conversation, half of them first person ("Where do I live?" / "You said you live in ..."). That corpus trains the skill of using what was just said, not the facts, and belongs in a chat replay mix (lab 2026-09-03-in-context-recall-sft). `--gap-bytes N` pads the small talk between the telling and the asking with long filler turns until the question sits at least N bytes after the fact; trained at a `seq` shorter than N with `state_carry chunks` and `bptt_window 2`, the carried recurrent state is the only path from fact to answer, so the loss trains the state to hold it (lab 2026-09-05-working-memory-program).
- Study corpus (`tools/build_study_corpus.py`, stem `study`): the general study-form generator of the tell-it-once loop. `build_fact_sft` renders atomic `{subj,obj}` facts as flashcards; this renders the SAME mechanism (varied surface forms, material in the assistant turn, both directions) for arbitrary documents, so code and long-form prose consolidate through one path and an atomic fact is simply a small chunk. Structure-aware chunking: Python by `ast` over top-level defs and classes, C by brace matching from a function-header regex, markdown by heading section, anything else by paragraph — which is also the fallback when a source file does not parse. Oversized chunks split on line boundaries so a code chunk never breaks mid-token. Four forms per chunk: recite (label → body), continue (head → tail), infill (pre/`<<<gap>>>`/post → gap, which for code IS fill-in-the-middle), identify (excerpt → label, the reverse direction that beats the reversal curse). The load-bearing invariant, asserted by `test_every_form_answers_with_the_material`: every form's assistant turn is the chunk's own bytes or its label, never model prose — consolidation runs under `loss_mask=assistant`, so a form violating this trains the model on its own output, which is what collapsed wren1_3 on 2026-08-24. `--holdout-frac` reserves a share of CHUNKS from training entirely and writes `{stem}_exam.json`; the train/val bin split is by exchange, so the same chunk lands in both and val loss measures fitting rather than recall — only a chunk-level holdout is a control. `--limit` caps chunks after shuffling for runs sized to a fixed budget.
- Study recall (`tools/study_recall.py`): the PRIMARY closed-book recall metric, measured by likelihood rather than by generation. "Does the model know this chunk?" is a likelihood question: teacher-forced NLL over the chunk's bytes, given only its label, is how strongly the weights expect that content. One forward pass instead of N sequential decode steps -- measured on cardinal 2026-08-25, generation-based scoring cost ~90 s a chunk and four checkpoints would have taken four hours, while the likelihood pass scores all 64 chunks in under five minutes. The assistant role mask does the span selection, so loss lands only on the chunk bytes and never on the prompt that names them: the prompt contains the label, and without that mask a model could score well by predicting its own question. Reported signal is the GAP between studied and held-out chunks; held-out NLL falls too as the model gets better at the domain, so only the gap isolates memorization. **Caveat, measured:** teacher-forced NLL averages over the whole chunk and is dominated by continuing code once started, so it can look excellent while label->content retrieval is absent. wren1_8@10 scored a 0.66 gap while free-running generation produced only whitespace. Always confirm a recall claim with `study_exam` before reporting it.
- Study exam (`tools/study_exam.py`): closed-book exam over study chunks, the generalization of `e4_retention_quiz` from facts to documents. No context, no retrieval. Scoring is graded rather than exact, because a byte-level model will not reproduce a 1 KB body verbatim: sequence similarity, common-prefix share (for code, the signature), and whether the label is recovered from an excerpt. Similarity is measured against the first `--max-new` bytes of the chunk, identically for both splits, so a decode budget is never reported as a memory failure. The reported signal is the GAP between studied and held-out chunks, not the studied score: a model merely good at code scores well on recite without having memorized anything.
- Capability probe (`tools/val_eval.py`): next-byte val loss for any checkpoints of any model on one NAMED corpus (`--val-bin mixed_chat`, a stem, filename, or path), with `--baseline <model>:<step>` scored first on the same bytes and every row reporting its delta against it. The draw is seeded as the trainer seeds it (`seed + 1`) through the trainer's own loader, so the Nth iteration reads byte-identical windows across models and a same-iteration comparison is exact; the role mask is not applied, matching `evaluate()`. Validated by reproducing a `train.csv` val row to six decimals. A consolidation run's own val rows cannot answer "did this cost general ability": left without `val_bin` the trainer validates on the heaviest mix member, which for a study run is the study material. Measure a consolidation run against its STARTING weights, not the run's best: the 2026-08-26 code-QA run improved both chat corpora 2.4-6.0% at every checkpoint while `stop_on_val_rise` halted it for rising 3.4% above its own best (successes.md). `training/fuse.py` (CLI `tools/fuse_checkpoints.py`) interpolates a checkpoint toward another (`theta <- alpha*tuned + (1-alpha)*base`); measured inert on a run that does not degrade, so it is the repair for a damaging run, never a default.
- Image corpus (`tools/build_image_corpus.py`): encodes a directory of images through a trained codec into `<stem>_{train,val}.bin`. Records are `caption bytes + code block + <|endoftext|>`; the code block is a FIXED length for a given codec and geometry, so the image is the last `image_code_bytes` bytes before each separator and everything before it is the caption. Fixed length is what removes the need for a marker inside a record that image bytes could collide with, and the builder refuses any record whose code block contains the separator anyway. Captions come from a `<image>.txt` sidecar and are optional; without them the corpus is unconditional. `build()` returns `image_code_bytes`, which the run needs. Reading image files needs Pillow; the packing path does not. `build_streaming()` (alias `build_from_cache`) is the trainer's path: frames in the decoded-pixel cache `fit_image_codec` wrote are read from it, the rest are decoded by a thread pool one batch ahead of the codec, encoding runs in batches of 64 on the device, and train/val is split by the content hash in each filename (so a picture keeps its side of the split as the set grows). CLI: `python -m tools.build_image_corpus <stem> <codec> --set <set>`.
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

One route module per concern in `veritate_mri/routes/`. High-traffic groups: `/trainers` + `/trainers/run|stop|tune_defaults|sysprobe`, `/runs/*` (incl. timelines and eval-deep), `/models/*`, `/corpus/*` (incl. `/corpus/mix/plan`), `/settings` + `/settings/notices`, `/sys_metrics`, `/backends`, `/wiki*`, `/addons`, `/atlas/*` (concept/neuron/lifetime/circuit/concepts_inverted — pure derivations over hook dumps), `/teacher/*`, `/mesh/*` (machine-to-machine), `/models/git/*` (models repo sync), `/app/*` (updater). `/images/decode_bench` (POST) is the image-decoder benchmark described under image decoding; `/images/sets` (GET) lists the picture sets and fitted codecs, `/images/ingest` (POST, body `{set, sources:[paths], min_edge?, caption_from_folder?, copy?}`) collects photos from folders on this machine into `data/images/<set>/` in a background thread (400 on a bad set name or a missing folder, 409 while one runs), and `/images/ingest/status` (GET) reports it; `/images/pick_folder` (POST) opens the operating system's folder chooser on the dashboard's machine and returns the chosen path (`{ok, path}`, or `cancelled`, or `unavailable` on a headless box so the form falls back to a typed path). `/train/discovery` also returns `image_sets` and `codecs`. `/images/models` (GET) lists image models; `/images/generate` (POST) generates a PNG from one in any mode (see image models, generation). `/images/caption/options` (GET), `/images/caption/preview` (POST), `/images/caption` (POST, background), `/images/caption/status` (GET), `/images/caption/stop` (POST) are the captioning stage. `/images/mri/<model>` (GET) and `/images/mri/<model>/<step>/<file>` (GET) serve the image probe to the Models tab; `/images/mri/<model>/prompt` (POST, background), `.../prompt/status` (GET) and `.../prompt/stop` (POST) draw a prompt at several checkpoints with and without the words; `/images/live/<model>` (GET) is the Training tab's live view of an image run (progress.json, checkpoints, latest probe, run-log tail; 404 when there is no such run directory). `/teacher/target_status` (GET) answers where distillation calls will land and whether that machine is already training: `{provider, model, kind, host, targets_this_machine, training_active, run, contention, contention_kind, reason}`. A hosted-API teacher never contends. A local teacher on this box contends while `trainer_runner.state()` reports running. A local teacher pointed at another box reports `training_active: null` — unknowable from here, and reporting `false` would be a guess dressed as a fact. The route never raises: a guard that 500s would block the start it is meant to advise.

The authoritative settings-key list is `DEFAULTS` in `veritate_mri/runtime/settings.py` — treat the code as the reference; notable keys beyond the obvious: `speculative_enabled`, `speculative_bytes`, `speculative_chunk_bytes`, `speculative_pause_ms`, `read_ahead_enabled`, `api_read_ahead_enabled`, `api_generate_ahead_enabled`, `corpus_compose_chunk_bytes`, `corpus_compose_val_ratio`, `corpus_compose_seed`, `trainer_sizes_path`, `corpus_mix_max_epochs`, `pytorch_load_mode`, `warm_models`, `device_preference`, `api_key`.

## image models

### pipeline

An image becomes a byte string and is then trained by the one trainer, unchanged. Four stages:
`ImageCodec` encodes an image to codes, `build_image_corpus` writes those codes into an ordinary
corpus bin, the trainer's `objective=masked_grid` lever fits a bidirectional byte model to them, and
`ImageCodec.decode` turns generated codes back into pixels. Vocab stays 256 throughout, so the corpus
format, the mix planner and the engine never learn that these bytes came from pictures.

Text conditioning costs no architecture. A record is `caption bytes + code block + <|endoftext|>`, so
a caption is simply the bytes that precede an image in the same stream, and a text corpus can be
mixed with an image corpus in one run through the existing multicorpus weighting.

### codec

`veritate_core/plugin/image_codec.py`. A patchify conv encoder at exactly the decoder's patch size,
residual VQ, and the shared `PatchDecoder`. Every codebook holds `CODEBOOK_ENTRIES` = 255 entries so
one code is one byte and `MASK_BYTE` (255) can never appear in data -- the masked objective needs a
value that is not a real code, and reserving the top one costs a single entry instead of a 257th
vocab slot the engine would have to learn. Planes are residual and emitted coarse to fine, so a
prefix of the byte string decodes to a valid lower-fidelity image; that is what makes anytime output
possible. `code_bytes(h, w)` = `planes * (h/patch) * (w/patch)` is the number the trainer needs.
Codecs live at `data/codecs/<name>.codec.pt` (`paths.codec_path`). A corpus bin is unreadable without
the codec that wrote it.

### decoding

`veritate_core/plugin/image_decode.py` owns image-decoder measurement. It builds decoders at random
weights and decodes one frame; it loads no checkpoint, reads no corpus, and saves nothing. Three
arms:

- `coord` — per-pixel MLP evaluated over output tiles, latent features gathered by bilinear sample.
- `patch` — byte-indexed patch dictionary (256 entries, one code is one byte) plus a residual
  predicted from each code's 3x3 neighbourhood, decoded in bands of grid rows.
- `conv_full` — conventional upsampling conv stack. The control: it materializes full-resolution
  feature maps, which is the structure the other two exist to avoid.

Two quantities decide a decoder, and they are reported separately. **Arithmetic** is the dominant
one: measured per pixel at 640x480, `patch` costs 430 FLOP, `coord` 8,160, and `conv_full` 29,664,
which at 1920x1080 is 0.89 / 16.9 / 61.5 GFLOP for one frame. **Peak activation bytes** is the
second: the tiled arms hold a fixed working set as the frame grows and bottom out at the output
buffer itself (6.2 MB at 1080p), while `conv_full` grows at a constant 128 B/px, reaching 265 MB.
That working set governs cache residency and whether the decoder runs on a small device; it is not
by itself a latency wall, since 265 MB against cardinal's measured ~12 GB/s is ~22 ms. Peak is
observed through a `TorchDispatchMode` that sees every aten output, so the figure does not depend on
what a decoder claims about itself, and it is measured in a separate pass from latency because the
mode distorts the clock.

`bench(height, width, ...)` returns per arm: `ms_p50`, `ms_all`, `gflop` (analytic, exact for the
matmuls), `achieved_gflops`, `peak_activation_bytes`, `params`. Geometry that the patch grid, the
latent grid or the conv stack cannot tile raises rather than silently cropping the frame. Every
shape knob is an argument; `POST /images/decode_bench` passes the body through and guards only the
frame edge (`MAX_EDGE`), so a mistyped resolution cannot allocate the box out from under a training
run. The route is synchronous, like `/trainers/sysprobe`: the caller sizes the cost through arms,
reps and resolution.

### trainer

`veritate_mri/training/image_trainer.py` is the canonical image trainer, registered as
`native/image_trainer` beside the text trainer and shown in the Training tab under **Train an image
model**. It owns the whole image pipeline in one launch: decode a sample of the set into the pixel
cache (`data/image_cache/`), fit the codec if none is named, stream the whole set into a corpus if
it is missing or stale, then train. The dashboard needs one field filled in: which set of pictures.
Every stage reports to `models/<name>/progress.json` (`veritate_core/plugin/image_progress.py`:
stage, done/total, rate, ETA, device, notes such as the last checkpoint) from the first second, and
the Training tab reads it (see *live view*, below). Stop is a request: SIGTERM (the Stop button)
sets a flag, the stage in progress raises `StopRequested`, the training loop saves a checkpoint of
the step in flight first, and the process exits 0 with `progress.json` saying `stopped`. A failure
leaves `failed` and the message, so a refused cache or a missing set is readable in the tab.

Shared with the text trainer by import, never by copy: the flag parser and its unknown-flag policy,
the size table (`trainer_sizes.json`; the trunk is dense, so every size applies), the lr schedule,
the optimizer builder, `config.json`, resume, evaluation, the checkpoint/save contract and the
`train.csv` rows the dashboard plots. Owned here: geometry, the codec, the corpus, and a loop with
one objective and no chunking. Fixed by what an image model is rather than offered as knobs:
`trunk=dense`, `objective=masked_grid`, `causal=False`, `hooks=off` (the checkpoint probes are text
probes). The manifest the form renders and the flags the process parses are one dict
(`readers/trainers.IMAGE_TRAINER_MANIFEST`), so they cannot drift; its `size_defaults` is
deliberately empty because the text table's per-size tuning (seq 512 at 10m, QAT on) would override
the geometry.

Stages, each skipped when its output already exists:

1. **Pictures.** `data/images/<set>/`, filled by `tools/ingest_images.py` or the form's **Choose
   folder…** button, which opens the OS folder chooser (`POST /images/pick_folder`), names the set
   after the folder and collects it (`POST /images/ingest`). Content-addressed and hardlinked: one copy per picture however many
   folders hold it, originals never moved, pictures under `--min-edge` (512) rejected rather than
   upscaled, `<image>.txt` sidecars or folder names as captions. Re-running ingests only what is new.
1b. **Captions** (optional, its own stage). `tools/caption_images.py` or the form's *Captions* block:
   a vision teacher (any configured provider; the model must be a vision model, e.g. `llava:13b`
   on Ollama, `gpt-4o`, `claude-sonnet-4-6`) describes every picture into `<image>.txt`. Style
   `sentence` / `tags` / `detailed` / `custom` prompt, `max_words`, `max_edge` (pictures are
   downscaled to this and sent as JPEG; tokens are what a caption costs), `concurrency`,
   `overwrite`. *Try on one picture* (`POST /images/caption/preview`) writes nothing; *caption all*
   (`POST /images/caption`, status/stop beside it) skips pictures that already have a caption and is
   resumable. The corpus sidecar records the caption count, so the next launch rebuilds the corpus
   and the words reach the model. Without this stage captions are folder names.
2. **Codec.** `codec` blank means `<set>_<h>x<w>_p<patch>x<planes>_codec` (`images_320x320_p20x4_codec`,
   plus `_x2` under an output scale, see `out_scale`): named after what a codec depends on, so
   every model trained on those pictures at that frame reuses it, whatever its size or name.
   Missing, it is fitted with `fit_image_codec` on a SAMPLE of the set:
   `codec_images` (8,192) pictures in content-hash order, which is a random draw; 0 fits on all.
   A codec does not improve past ~10k pictures, and the sample is all that is decoded into the
   pixel cache (8,192 x 320x320 is 2.5 GB; the whole of a 140k-picture library at 1920x1080 would
   have been 869 GB, which is what a typed frame size produced on 2026-09-05). The cache is refused
   before it starts when it would exceed 85% of the free disk. Decoding is the one CPU stage, by
   nature of JPEG: JPEGs decode at the coarsest DCT scale that still covers the frame
   (`Image.draft`), EXIF orientation is applied, `DECODE_WORKERS` (= cores) threads run it.
   `codec_epochs`, `codec_batch_size`, `codec_lr` as before; held-out L1 and PSNR per epoch. A
   named codec is reused and a `patch`/`planes` mismatch refused, because a corpus is unreadable
   under a different codec.
3. **Corpus.** `<codec>_img_{train,val}.bin` (`images_320x320_p20x4_img`, reused the same way) for the
   WHOLE set (`build_image_corpus.build_streaming`):
   frames in the cache are read from it, every other picture is decoded by a thread pool one batch
   ahead of the codec, so Pillow and the GPU overlap and no picture is decoded twice; the sidecar
   `<stem>.image.json` records set, codec, geometry, image and caption counts and `image_code_bytes`.
   A re-launch whose sidecar matches skips the encode; a grown or newly captioned set rebuilds.
4. **Run.** `seq` = `image_code_bytes + caption_bytes` rounded up to 64 unless set explicitly (a value
   below the image is refused, not clipped). A frame above 1024 px on either side is refused
   (`MAX_EDGE`); the form's single *picture size* control offers 160-640 px squares and sets
   `height` and `width` together, snapping a pair it does not offer back to 320. **Memory is
   planned before the first step** (`plan_micro_batch`): weights + grads + optimizer state (Muon
   one momentum, AdamW two) plus, per layer, activations and the two attention tensors a device
   without a flash kernel holds for backward (`heads x seq^2`, the dominant term for a picture
   model: fp32 under sdpa's fallback, the working dtype under the explicit path, see *Speed*
   below; `attention_bytes`), against 70% of unified memory or what is free, whichever is less,
   and 30% less than that when the model is compiled (a compiled graph measured 1.27x the eager
   peak). Calibrated against measured peaks on an M2: 20m at 16 pictures 8.2 GB measured / 8.8
   estimated, 80m at 4 pictures 6.1 / 6.0, 80m at 8 pictures 9.1 / 10.7. When the batch
   does not fit in one forward the step becomes `accum` forwards of `micro` pictures with the loss
   scaled by `1/accum` (`image_grid.masked_step(scale=)`): the optimizer sees the same batch, the
   result is the same, the step is slower; the run log and `progress.json` say so (`micro_batch`,
   `grad_accum`). An out-of-memory error mid-run frees the device cache, halves the pictures per
   forward and retries the step; at one picture it is a clear error naming the size as too big for
   this device (800m at seq 1152 is that case on a 24 GB Mac: ~21 GB at one picture). A relaunch
   of a name whose earlier attempt never saved a checkpoint is allowed (`POST /trainers/run` only
   409s when weights exist) and drops that attempt's `train.csv` rows first. Every draw's context
   is the record's own bytes (PAD, separator, caption: the window generation builds; the previous
   picture's code tail the bin holds before the separator is padded out), and `caption_dropout`
   (0.1) of draws carry no context at all, so the model also learns to draw unprompted. The
   picture probe runs every `probe_every` steps (100) as well as at every checkpoint, so the
   first pictures appear at step 100, not at the first save. Logs `s/step`, `img/s` beside
   `tok/s`; `config.json` carries `training: image`, the codec name, the set, the geometry and
   `out_scale`, which is everything decoding a sample later needs.

**Speed** (measured, M2 with 10 GPU cores and 24 GB, seq 1152, batch 16, 2026-09-05; the profile
script and its output are in the handoff). The 137 s/step the first 80m run showed was paging: the
old code ran 16 pictures per forward at ~29 GB on a 24 GB box. Within memory, four things decide a
step, and each is measured rather than assumed:

| lever | measured | what the trainer does |
|---|---|---|
| precision | GEMM fp16 3.2 TFLOPS, fp32 2.8, **bf16 1.5**; a 20m step 4.28 s in bf16 vs 3.54 fp16 vs 3.47 fp32 | `precision=auto` (default) picks the half precision `hardware.half_precision_probe` measures fastest on this GPU at launch (fp16 on M1/M2; bf16 stays where it is fast); fp16 runs under `torch.amp.GradScaler`; the log prints the rates and `config.json` records `precision_resolved` |
| attention | sdpa on MPS has no fused training kernel and upcasts to fp32: 131 ms fp32 / 146 ms fp16 per layer-batch; written out in fp16 **74 ms, half the memory** | `veritate_core.model.attention` takes the explicit form for non-causal half-precision on MPS (`EXPLICIT_ATTENTION_DEVICES`); text runs, fp32 and CUDA are untouched. Full step: 20m 3.56 -> 3.15 s, peak 11.7 -> 8.2 GB; 80m at 4 pictures 9.44 -> 8.54 s |
| Muon | its Newton-Schulz orthogonalization is hardcoded bf16 upstream: **1.46 s per 80m step** vs 0.98 in fp16/fp32 | `optim.ns_dtype` is the measured half precision on a GPU (the vendored Muon carries it); bf16 keeps `torch.optim.Muon` |
| compile | inductor on MPS (torch 2.14): 20m step **3.15 -> 2.37 s (1.54x over eager, 1.8x over the old default)**, ~9 s to compile | `compile=auto` (default) wraps the training forward in `torch.compile` on a GPU, never on the CPU; probes and evaluation run the eager model on the same weights; a compile failure falls back to eager with a log line and the run continues |

Where that leaves this box, best configuration: 20m 2.4 s/step (5,000 steps in 3.3 h), 80m at
8x2 pictures 8.5 s eager (12 h; compiled unmeasured at this size), 200m at 4x4 19 s (26 h). More
pictures per forward is not faster once memory is tight: 80m at 16x1 measured 12.7 s and 17 GB
against 8.5 s at 8x2, which is why the planner's budget is what is free, not what is installed.
The remaining cost is the token count itself (1,152 per picture: 4 planes x 256 cells + captions),
which is the next lever (ideas.md, IDEA 24). Every log line and `progress.json` carry `s/step`.

Measured on the fit path (M2, CPU forward, batch 16): decoding the batch in one pass instead of one
image per Python iteration is 1.84x, bitwise identical (`PatchDecoder.forward` accepts a batch;
`render`, the F0 bench path, is untouched). Pillow decode is the dominant cost of a fit over a photo
library and is paid once per (set, geometry). Measured 2026-09-05 on the user's set (1200-2400 px
JPEGs, M2, one thread): full decode 22 img/s at a 320 frame, `draft` + EXIF decode 70 img/s, 3.1x;
at 1920x1080 the DCT scale cannot drop and it is 15 vs 19 img/s, which is why the frame size, not
the decoder, decides the cost of this stage.

### generation

`veritate_core/plugin/image_sample.py`; `POST /images/generate`; the **Images** panel at the top of
the Generation tab. One mechanism serves every mode, because the model was trained to fill masked
cells given the bytes before them: **text** (all cells masked, caption in front), **variation**
(encode the source photo, regenerate a `strength` share of cells), **inpaint** (regenerate the cells
under a rectangle given as frame fractions), **expand** (the source shrunk to `expand` of the frame
in the centre, the model paints the margin), **unconditional** (no caption). Decoding is MaskGIT's
parallel refinement: one forward per pass predicts every masked position, the most confident commit,
and the count still masked follows the cosine schedule the trainer sampled from -- `passes` is the
knob F2 measures (4 should match causal AR; more than 8 fails the CPU budget). `MASK_BYTE` cannot be
emitted (its logit is -inf); cells are kept or regenerated across all planes at once so kept content
stays coherent; the window is laid out as training saw it (pad, separator, caption, image). Output
is the model's training frame. `temperature` 0 is greedy; a `seed` replays. The last model loaded
stays resident. `/images/models` lists what can generate (models whose `config.json` says
`training: image`).

Words steer only as far as the captions the model trained on: with folder-name captions the model
learns those names, not open vocabulary. Describe-anything generation needs a caption per picture
(the captioning stage, above, writes those sidecars from a vision teacher).

### probe and the Models tab

`veritate_core/plugin/image_probe.py` runs at every checkpoint of an image run and writes
`hooks/step_<N>/image/`, one file per question a person asks of a picture model:

- `samples.png` -- eight pictures from nothing, the SAME seeds at every step so the timeline shows
  one draw evolving, plus pictures from held-out captions when the set has any.
- `passes.png` -- the first sample forming pass by pass (`image_sample.fill(trace=)`): grey cells are
  still undecided; `pass_committed` / `pass_confidence` per pass. A model that has learned structure
  commits the layout early and the detail late.
- `fill.png` -- four held-out pictures: original / half the cells hidden / the model's completion.
- `layers.png` -- the logit lens for a picture: the residual after every block read through the
  model's own output head and decoded, so the tab shows what it would draw if it stopped at layer
  1, 2, ... L. `lens_agreement_per_layer` (with the final layer), `lens_accuracy_per_layer`, and
  `commit_layer`: the first layer that already agrees with the final answer on 90% of hidden cells,
  i.e. where the decision is made. `residual_norm_per_layer` beside it: how much each layer carries.
- `confidence.png` -- original / hidden / filled / a per-cell confidence map for the first held-out
  picture, plus `mean_confidence`, `calibration` (accuracy per confidence band) and
  `expected_calibration_error`: is the confidence earned, or noise.
- `cell_loss.png` -- loss per grid cell averaged over the held-out batch: where in the frame it
  struggles; `centre_loss`, `edge_loss`, `centre_edge_loss_ratio`.
- `nearest.png` -- the training picture closest to each sample by cell-wise hamming distance over a
  fixed 16,384-record sample of the train bin, with `novelty_per_sample` / `novelty_mean` (share of
  cells that differ; 0 is a copy). Memorisation shows as novelty falling toward 0.
- `attention.png` -- where the centre cell attends, one map per layer, recovered from the qkv
  projection with a forward hook; `attention_entropy_per_layer` and `attention_entropy_per_head`
  (0 focused, 1 uniform).
- `recon.png` -- the codec's own reconstruction of the held-out pictures: the ceiling the model
  cannot beat, so blur is attributed to the right stage.
- `formation.png` -- the order the first sample formed: a heat map of the decode pass in which each
  cell (plane 0) was decided, blue early, orange late, read off the same `fill(trace=)` as
  `passes.png`; `commit_pass_map` (per cell), `commit_pass_per_plane` (mean pass per plane: the
  structure plane should commit earlier than the detail planes, and earlier as training goes on),
  `formation_passes`.
- `planes.png` -- coarse to fine: the first sample rendered from 1, 2, ... all of its planes
  (`ImageCodec.decode(codes, planes=k)`; residual planes are plane-major so a prefix is a valid
  coarser picture). What the left tile gets right is layout; the difference across the row is
  what the detail planes carry.
- `metrics.json` -- all of the above plus fill accuracy overall and per plane (plane 0 is coarse
  structure, the last plane fine detail; a model learns them in that order), loss at 25/50/75/100%
  hidden, codes in use out of 255 and their entropy (a collapse shows as a handful), held-out count,
  **what forms first**: `sample_sharpness` and `heldout_sharpness` (mean absolute Laplacian of the
  samples and of the codec's reconstructions of the held-out pictures, so `detail_ratio` is
  detail against the ceiling) and `colour_match` (1 minus half the L1 distance between 4x4x4
  colour histograms of samples and held-out pictures; 1 is the same palette),
  `attention_distance_per_layer` (attention-weighted grid distance from a plane-0 cell to the
  cells it reads, in cells, over heads: how far each layer looks), and the raw material the tab
  derives progression from: `sample_codes_b64` (the same-seed samples' codes, so consecutive
  checkpoints can be diffed cell by cell), `loss_map` and `confidence_map` (per grid cell),
  `grid` (gh, gw).

Everything is recovered with forward hooks under no_grad; a probe never changes the model. The
trainer passes the train bin for the novelty pass. `GET /images/mri/<model>` returns every step's
metrics with the files present; `GET /images/mri/<model>/<step>/<file>` serves a picture.

**Rendering.** Probe pngs hold 192 px tiles (`THUMB`; `metrics.json` records `thumb` and `gap`
so the tab can crop any probe, old 96 px ones included) and are shown at a fixed 128 css px per
tile (`_imriPhoto`: the width is computed from the tile count, so the browser never stretches
them, and they stay sharp on a 2x display); click any picture for the full-size png. Every map --
formation order, churn, where it improved, confidence, loss per cell, attention by head -- is drawn
from the numbers in `metrics.json` as a fixed-cell SVG (`_imriCellHeat`: square cells, never
stretched to the panel, hover for the value, a gradient legend under each), the pngs serving only
as a fallback for probes that predate the numbers. The filmstrip canvases are sized for the
display's pixel ratio. The view is grouped into sections: what it draws, how a picture forms,
inside the model, over training.

**The Models tab shows an image model OR a text model, never both.** The decision comes from the
model's kind (`training` in `/timelines`, else `config.json`) and is made before anything is
loaded: `ensureLearningLoaded` decides first and returns for an image model, the classroom mirror
and canvas renders are gated the same way, and the timeline picker labels image models. For an
image model every byte-level panel is hidden and the picker panel's copy changes; the image view
then shows: a strip of every probed checkpoint (click to select) and KPIs (fill accuracy,
confidence, calibration error, novelty, decides-at-layer, codes in use, attention spread,
centre/edge loss, held-out count); what it draws with the closest training picture under each
sample and its novelty; how a picture forms pass by pass; **the order it forms** (the formation
map, mean commit pass per plane as bars, and the per-plane commit pass over training) beside
**coarse to fine** (the planes render); **what forms first** (plane-0 fill accuracy, colour match
and detail ratio on one 0-1 axis over training: the order the curves rise is the order the model
learns, palette and layout before fine detail) beside **how far it looks** (attention reach per
layer as bars and its mean over training: untrained attention reads far and near-uniform, trained
texture layers settle at 1-2 cells while a few layers stay global); the fill test beside the
confidence map and the calibration bars; through the layers with the agreement/accuracy-per-layer
chart; residual depth and where it struggles; attention per layer and the per-head heatmap; and
over training: fill accuracy per plane, loss by hidden fraction, confidence and calibration error,
novelty, decision depth, codes in use, and the codec ceiling. It polls while a run is training that
model.

**Progression.** A scrubber under the checkpoint strip (previous / next, a range slider, play at
1.2 s per checkpoint, *follow latest* which tracks the newest probe while a run trains). **One draw
through training**: a filmstrip of one same-seed sample cropped out of every checkpoint's
`samples.png` (client-side canvas crops on the known 96 px / 4 px grid), and a second strip of one
held-out picture's completion at every checkpoint with its original at the left; click a frame to
jump there. **Churn** (new): the same-seed samples' codes at consecutive checkpoints are diffed cell
by cell -- the settling map shows, per cell, the share of samples whose code there changed since
the previous probe (blue settled, orange still moving), the filmstrip outlines those cells in
orange, the *settled since last* KPI is the share of cells unchanged, and the churn curve (all
planes and per plane) falls toward 0 as the model converges; a jump is a phase change, and a region
that keeps flickering late is one the model has not learned. **Where it improved**: loss per cell at
the first probe minus now, green easier, red harder -- a model typically learns borders and
backgrounds before subjects, which shows as green edges around a red centre.

**Prompt it.** A prompt box at the bottom of the image view (built once per model, so typing
survives the polls): words, a seed, passes, an optional photo (then it is a variation of that
photo guided by the words). *Draw at this checkpoint* or *draw at every checkpoint* start a
background job (`POST /images/mri/<model>/prompt` with `{caption, steps?, seed, passes, image?,
mode}`; `GET .../prompt/status`; `POST .../prompt/stop`) that draws, at up to 12 evenly spaced
checkpoints with the last always included, the picture WITH the words and the same-seed picture
WITHOUT them. Each result shows both, and `steering` -- the share of cells the words changed -- is
plotted over training as **caption influence**: rising means the model is learning to listen to
captions; flat near zero means the captions it trained on did not teach it these words (folder
names, for instance). Results arrive one checkpoint at a time while the job runs.

### live view on the Training tab

`GET /images/live/<model>` returns `progress.json`, the checkpoints on disk with the last one's time,
the latest probe's metrics and files, whether the runner is on this model, and the tail of the run
log; it works before `config.json` exists, which is most of a first run's wall clock. The Training
tab shows it whenever the running trainer is `native/image_trainer`, the run picked in *live
training* is an image model, or the Images action is open with a last run known, and hides the
byte-model panels (register fluency, comprehension, concepts, lens drift) meanwhile. The view: a
GPU/CPU chip with the device and, from `progress.json` notes, chips for the precision the run
computes in, *explicit attention* and *compiled* when those paths are on; the four stages as bars
(done with duration, reused, running with done/total, rate and time left, pending), the trainer's
message, a *model saved* line (checkpoints on disk, last step and age, steps to the next save),
KPIs once training (step, loss, held-out loss, seconds per step, pictures/s, lr, time left, fill
accuracy, pictures per forward, parameters, pictures), the loss curve from `train.csv`, and from
the latest checkpoint: its samples, the pass-by-pass formation strip, the formation-order map
beside the coarse-to-fine render, and its fill test, with colour match and detail against the
codec ceiling in the caption and a link to the full Models-tab view; then the run log tail. A
failed run shows its reason in red above the stages. The form's memory estimator derives the image
seq (`_imgfDerivedSeq`), counts both attention tensors in the bytes the trainer will hold them in
(half on Apple silicon under a half precision, fp32 otherwise) and treats `auto`/`fp16` as half on a
GPU, so a size that cannot fit reads red ("WILL NOT FIT even one picture at a time") before launch,
and one that fits only in pieces reads amber; the size hint carries the same word.

**Continue a model from the Images form.** The Model card's *start from* select lists saved image
models (`/images/models`: those with a checkpoint) as "continue <name> · step N". Picking one hides
name, size and picture size, sends `--resume <name>`, and the trainer pins the model's pictures,
frame, codec, corpus and size from its `config.json` (`pin_structural_args`) whatever else the form
sent; `total_steps` is the new end. The run route skips its name check on resume.

Not yet: output beyond the training frame, and F1 itself -- no codec has finished fitting on real
photographs yet (the first real launch, 2026-09-05, was stopped in its decode stage at 1920x1080);
the pipeline is proven on synthetic frames end to end.


## settings reference

The Training-tab "learn more" links resolve to the sections below by slug.

### objective

`next_byte` (default) is ordinary causal byte prediction and is what every language run uses.
`masked_grid` trains a bidirectional model to fill masked positions of an encoded image. An image has
no causal order, so predicting it left to right spends one forward per byte for nothing; the masked
form lets generation run in a few parallel refinement passes instead, which is the difference between
a compute-bound and a bandwidth-bound decode. Masking follows a cosine ratio schedule, loss lands only
on masked code positions, and attention is built non-causal (`Veritate(causal=False)`, a flag that
holds no weights, so every existing checkpoint still loads). Requires `image_code_bytes`, `seq >=
image_code_bytes`, and `trunk=dense`: the patched trunks gather on text byte boundaries, which a code
stream has none of. Stamps `training_kind=image` on `config.json`.

### image_code_bytes

Bytes of encoded image at the end of each corpus record, reported by `build_image_corpus.build()`.
Read only under `objective=masked_grid`, which uses it to locate the image inside a record. Draws are
record-aligned: every window is `seq` bytes ending at a code block's last byte, so the image always
occupies `[seq - image_code_bytes, seq)` and whatever caption fits precedes it. The next-byte loader's
uniform draws would cut images in half. A record with less than `seq` bytes of history (the first
pictures in a bin, or a small val bin) is left-padded with `PAD_BYTE` (0) rather than dropped, so
every picture trains.

### image_set

Which set of pictures an image run trains on: a directory name under `data/images/`. Sets are made by
`tools/ingest_images.py` or the form's *add photos* panel; the trainer refuses a set that does not
exist or holds no pictures. Only pictures the codec and corpus stages can read count
(`png jpg jpeg webp bmp`; HEIC needs `pillow-heif`).

### codec

The image <-> bytes codec for the run. Blank fits a new one named `<model>_codec` on the set before
training; a name reuses that codec. A corpus is only readable under the codec that wrote it, so a
named codec whose `patch` or `planes` differ from the form is refused rather than silently producing
a corpus the model cannot decode.

### height_width

Training frame in pixels: what the model reasons about and pays for. Every picture is cover-scaled
and center-cropped to it, so nothing is stretched or padded. Both must be multiples of `patch`.
`image_code_bytes = planes x (height/patch) x (width/patch)`; at the defaults (320x320, patch 20,
4 planes) that is 1,024 bytes, inside IDEA 24's 2,048-code F1 budget. The form sets the pair from
two selects: a **long edge** (160 to 960 px) and a **shape** (square, landscape 4:3 / 3:2, wide
16:9, portrait 3:4 / 2:3), the short edge snapped to a multiple of the patch; the hint under them
shows the exact frame, the bytes and tokens per picture, and the step time relative to the 320
square, estimated from the token count (linear layers scale with it, attention with its square, in
the 60/40 shares an 80m step measured on an M2). A bigger frame is the expensive way to a bigger
picture: 640 square at patch 20 is 4,096 bytes and ~8x the step time, 960 is ~30x and does not fit
this class of machine. Bigger finished pictures come from `out_scale` instead; a coarser `patch`
(40) at a larger frame keeps the token count and lets the codec carry the pixels.

### out_scale

How many times the frame the finished pictures are: `1` (default), `2`, `3`, `4`; `2` renders
640x640 px from a 320x320 model. Resolution as a decoder loop bound, not a longer sequence
(IDEA 24): the model's bytes, tokens, memory and step time do not change. The codec is fitted with
the sample pictures cached at the OUTPUT size (`out_scale^2` times the cache: 8,192 pictures at
640x640 is 10 GB, the form's hint says how much), the encoder reads them pooled down to the frame,
and the decoder (`PatchDecoder` at `patch x out_scale` px per cell) learns to paint the full size
from real pixels at that size, scored by L1 and PSNR against them like any codec fit. The extra
detail is therefore the decoder's, learned from your photos, not the model's; blur in a 2x picture
is the codec ceiling (`recon.png`) at 2x. The codec carries the scale (`ImageCodec.out_scale`, in
its saved config) and is named for it (`images_320x320_p20x4_x2_codec`), so a 1x and a 2x codec
over the same pictures coexist; a run that names an existing codec adopts its scale. The corpus is
built from every picture decoded at the frame (the 2x cache is not reused for it, so one decode
path feeds the encoder). Generation, the probe pictures and the prompt panel all come out at the
output size; the decoded edge is capped at 1920 px (`MAX_OUTPUT_EDGE`), which the form flags.

### probe_every

Image trainer. Steps between picture probes (`image_probe.dump`: samples, fill test, formation,
attention, all of the Models tab) run on the eager model with no weights saved; default 100, `0`
= only at checkpoints. Each probe is ~10-20 s on an M2, so 100 costs a few percent of a run and
the first pictures appear at step 100 instead of the first checkpoint (500). Checkpoints still
probe. The Models tab and the prompt panel list probes and checkpoints separately: the scrubber
walks probes, drawing with words needs a checkpoint.

### caption_dropout

Image trainer. Share of training draws whose whole context is padded (default 0.1): the model sees
the picture with no separator and no caption, exactly the window an unprompted generation gives it,
so it learns to draw from nothing as well as from words. This is what classifier-free guidance
needs and what makes the prompt panel's *caption influence* (with words vs without) a fair
comparison. The context is otherwise the record's own bytes only: the loader pads everything before
the separator that opens the record (`image_grid.make_record_loader`), because the bin holds the
previous picture's code tail there and generation never does; training and generation now see the
same layout.

### planes

Residual VQ depth of the codec. Each plane adds one byte per patch and quantizes what the planes
above it missed; planes are emitted coarse to fine, so a prefix of the byte string decodes to a valid
lower-fidelity picture (anytime output). Costs bytes per image linearly.

### patch

Pixels per code cell, on both the encoder and the decoder (they share the grid). Smaller means more
detail and more bytes per image. Must divide both `height` and `width`.

### caption_bytes

Context budget ahead of the image. `seq` is derived as `image_code_bytes + caption_bytes` rounded up
to 64 when `seq` is 0; a caption longer than the budget is simply cut from the front of the window,
never the image.

### codec_epochs

Passes over the set when fitting a new codec (ignored when `codec` names an existing one). Pictures
are decoded once into `data/image_cache/`, so epochs after the first are pure tensor work; the fit
reports held-out L1 and PSNR per epoch and saves after each.

### codec_images

Pictures the codec is fitted on when `codec` is blank: the first `codec_images` names in content-hash
order, which is a random sample, so the pixel cache holds that many frames and no more (8,192 x
320x320 = 2.5 GB). 0 fits on the whole set. A codec does not improve past ~10k pictures; the corpus
still holds every picture, streamed past the fitted codec.

### recipe

Preset bundles (`TRAIN_RECIPES` in `index.js`) that prefill the form; applying one overwrites the affected fields and nothing else. Every value remains individually editable after applying.

### optimizer

`adamw` (default) or `muon`. Muon routes 2-D non-embedding weights through Muon and the rest through AdamW behind one optimizer surface. A checkpoint saved under one optimizer resumes under the other with a fresh optimizer state: the restore is skipped and logged (`optimizer state restore skipped`), moments start from zero, so pair the switch with a warmup or a low constant lr. Muon measured 1.60x fewer bytes to equal quality at 10M (see successes.md).

### trunk

Architecture selector: `dense` | `patched` | `hybrid` | `looped` | `recurrent` | `memory` | `hybrid_moe`. `dense` is the canonical exportable trunk; `hybrid` exports via the v13 engine format; the rest serve through PyTorch. `hybrid` composes boundary patching with a constant-state recurrent global mixer — the best measured quality/wall-clock trade (successes.md). `hybrid_moe` adds a mixture-of-experts global FFN (top-2 of 8 quarter-width experts + 1 shared).

### precision

Text trainer: `fp32` or `bf16`. bf16 halves activation memory and is the default where supported;
unsupported devices resolve to fp32 with a warning, not an error.

Image trainer: `auto` (default), `fp16`, `bf16`, `fp32`. `auto` is the half precision this GPU is
MEASURED fastest at when the run starts (`hardware.half_precision_probe`: a square matmul in fp16,
bf16 and fp32, cached per process). On an M2 that is fp16, because bf16 matmuls run at half the
fp16 rate there (3.2 vs 1.5 TFLOPS) and a bf16 step is 20-25% slower; on a GPU that is fast at bf16
it stays bf16. fp16 trains under `torch.amp.GradScaler` (loss scaling, so small gradients survive
the format; an overflowed step is skipped and the scale comes down). Asking for `bf16` on a GPU
that measures fp16 faster warns with the two rates. The CPU resolves everything to fp32. The run
log prints `precision: auto -> fp16 (measured fp16 3.2 TFLOPS, bf16 1.5, fp32 2.8)` and
`config.json` records `precision_resolved`; Muon's orthogonalization uses the same measured dtype.

### compile

Image trainer: `auto` (default), `on`, `off`. `torch.compile` fuses the model's many small kernels
into a few; a 20m picture model on an M2 is launch-bound enough that it measured 1.54x per step
(3.15 -> 2.37 s at batch 16), the first step taking ~9 s longer to compile. `auto` compiles on a
GPU and never on the CPU. Only the training forward is compiled; evaluation and the checkpoint probe
run the eager model on the same weights. A graph that fails to compile, at build or at the first
step, falls back to eager with a line in the run log and the run continues. The memory plan takes
30% off the budget when compiling (a compiled graph measured 1.27x the eager peak). `progress.json`
notes `compiled`; the live view shows a chip.

### batch_size

Sequences per step. Size it to the box: memory ceiling from the planner, then pick the measured best-throughput batch (not the max that fits — throughput collapses once memory pressure bites). Moderate jumps only, or retune LR.

### seq

Context window in bytes. Directly costs throughput and activation memory; fixed shapes only on MPS.

### n_chunks

Chunks per optimizer step (gradient accumulation). Raises tokens/step at near-zero memory cost because activation memory is bounded by `bptt_window`, not chunk count. Fewer optimizer updates at a fixed token budget degrades convergence — don't maximize blindly.

### bptt_window

How many chunks gradients flow back through when state is carried. The real memory knob for recurrent/hybrid runs.

### freeze_blocks

Resume-only. N freezes the token and position embeddings (the LM head is tied to the token embedding and freezes with it) and the first N entries of `model.blocks`; the optimizer is built over the trainable parameters only. Backward stops at block N because nothing below it requires grad; freezing the blocks alone does not buy this, since a trainable embedding below them makes backward walk every block for its input gradient. 0 trains everything. Measured on cardinal-01 (2026-09-02, 200M hybrid, 20 blocks, AdamW, batch 7, activation checkpointing on, idle box), seconds per step:

| freeze | embeddings | step | params training |
|---:|---|---:|---:|
| 0 | trainable | 94.5 s | 270M |
| 10 | trainable (blocks only) | 83.3 s | 136M |
| 10 | frozen | 58.6 s | 134M |
| 15 | trainable (blocks only) | 78.5 s | 68M |
| 15 | frozen | 47.5 s | 66M |
| 15, activation checkpointing off | frozen | 40.0 s | 66M |

A count that leaves no block trainable is refused at launch. Binding under the freeze, measured the same day on cardinal-01 (lab 2026-09-02, fast consolidation on cardinal): `freeze_blocks 15`, AdamW 3e-5 constant, `fact_sft:0.75,mixed_chat:0.25`, batch 7, 200 steps (2.7 h) took a 200M model from 0/100 to 72/100 closed-book on 50 novel facts (87-91/100 by step 400-500, 6.8 h) with replayed mixed_chat val +2.4% at step 200 and +0.8% at step 450 on a 32-draw `val_eval` (the trainer's own 4-draw rows read -5.7% on windows that favoured the run: a forgetting call at the +2% level needs several times the val file in draws, never the trainer's row), where the full-parameter E4 recipe (Muon 5e-6) needed 79-98 MB of drill for the same recall. The count is model-shape-specific, so it is a per-install sleep setting (`sleep_freeze_blocks`), not a global default.

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
