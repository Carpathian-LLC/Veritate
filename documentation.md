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

### readers layer

`veritate_mri/readers/` is the data-access layer: routes call readers, readers own all `os.listdir`/`open`. Missing files return `None` or empty, never raise.

Key readers: `paths.py` (every canonical path; `corpus_search_dirs()` returns `(data/corpus, trainers/corpus)`, canonical first, and `_corpus_file_path()` resolves existing-file-wins — anything listing or globbing corpora walks both), `trainers.py` (builds the trainer manifest from `trainer_sizes.json`; default size comes from that file's `default_size` key), `bin.py` (`.bin` header parsing; `RETIRED_VERSIONS = {10}` hard-rejects with a clear error; `act_boost` returns `None` for v13), `wiki.py` (serves this file to the dashboard).

### settings store

`mri_settings.json` via `veritate_mri/runtime/settings.py::DEFAULTS`. `PUBLIC_AI_ENDPOINT`/`PUBLIC_AI_KEY` are injected live and never persisted. Load strips unknown stale keys once.

### auth

Two independent gates. Dashboard password: off unless `VERITATE_DASHBOARD_PASSWORD` is set; public surface is `/`, `/login`, `/logout`, `/static`, `/chat`, `/hybrid`; without `VERITATE_SECRET_KEY` sessions reset every restart. API bearer key: gates `/generate`, `/agent/stream`, `/v1/*` only; key and counters live in `mri_settings.json`.

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
- **Hybrid chat** (`/hybrid/chat`): local byte model slides context (no summarize); remote summarizes. `mri:true` is an opt-in telemetry flag and never changes sampling.
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

Single-page app: `veritate_mri/web/index.html` + `index.js`. Tabs: Chat, Generation, Models, Training, Wiki, Logs, Settings. Hash-based navigation; the `valid` array in `index.js` is the sole allowlist of routable tabs — an unknown tab silently falls back to `generation`. Adding a tab: add to `valid`, add an `activateTab` branch, and update the tutorial selectors (renaming a tab breaks the tutorial spotlight silently). `MINIMAL_HIDE_TABS` hides tabs in minimal mode.

Data transport: polling (train CSV 5s, runs 30s) plus SSE for streams. SSE has no HTTP status on failure — the log view re-polls a snapshot to detect stalls.

Conventions:

- Canvas charts: `fitCanvas` returns early on a detached canvas; `ResizeObserver` coalesces redraws; drawers are idempotent.
- Dropdowns anchor to the same horizontal edge as their trigger.
- Every `localStorage` call is wrapped in try/catch; localStorage is shared per origin, so two dashboards on one machine collide.
- Standalone modules (prune, tutorial) use the IIFE pattern.
- HUD meters hide when their source field is null — null ≠ 0; never fabricate a reading. macOS has no first-party CPU temperature API.
- Models tab: `modelShape` is the one module-global driving dimension-scaled UI; a `0` field means "not known yet".
- Generation tab: chat stop markers are matched **without** the closing bracket because byte models reproduce markers approximately. Read-ahead and speculative prefetch are separate controls with different risk profiles. The capability gate stays live via a 15s poll.
- Training tab: only multi-size manifests show the size dropdown. Auto-tune writes to the machine-local tuning store, never the manifest.
- Corpus mix planner: profiles are server data, never hardcoded. Ticking the Training-tab corpus picker rebuilds the field from checked stems joined `+`, discarding weights — re-accept the mix to restore them.
- RAG train panel: corpus stem fixed to `rag_ui`; requires a configured teacher; one job at a time (409 on a second start).

## training

### the trainer

There is exactly one trainer: `veritate_mri/training/veritate_trainer.py`, ordinary tracked platform code. Sizes and tuned defaults are data in `veritate_mri/data/trainer_sizes.json` (34 sizes, 5m–1t): `shared_defaults` → the size's `defaults` → user settings, resolved identically by `readers/trainers.py` and the trainer so the offered set never drifts from the supported set. `default_size` in the same file prefills the Training form. Adding or retuning a size is a JSON edit, nothing else. No tunable is ever a literal in the trainer.

The dashboard builds the trainer's manifest record synthetically (`_native_record()` in `readers/trainers.py`); third-party trainer plugins may exist as `trainers/<id>/{manifest.json, trainer.py}` bundles discovered by `_walk()`, must import only `veritate_core.plugin`, and require explicit user permission to create. The prune panel's generated plugins are the one sanctioned way new `trainers/<id>/` dirs appear.

Plugin import surface (`veritate_core.plugin`): `save`, `paths`, `model`, `qat`, `hardware`, `multicorpus`, `oom_recovery`, `bench`, `mem_planner`, `mem_executor`, `get_teacher_client`. Nothing else. `hardware.resolve_precision` downgrades bf16 to fp32 on unsupported devices; forced-device fallback warns rather than raises.

### launching runs

Launch via `POST /trainers/run` with `id="native/trainer"` and `args` (never a hand-rolled launcher — that bypasses the runner, the heartbeat, and the env plumbing below). `trainer_runner.start()` enforces one training instance globally and sets `VERITATE_PLUGIN_ID`, `VERITATE_DEVICE`, and thread caps on spawn. `NEGATABLE_BOOL_FLAGS` emit `--no-<flag>`; other manifest-default-true booleans cannot be disabled through the API — raise batch instead of chasing act_ckpt.

Load-bearing launch facts:

- **`model_type` is mandatory and silently defaults to `language`.** Values: `language` | `code` | `statistical` | `other`. It is not a manifest field; it rides only the `VERITATE_MODEL_TYPE` env var set by `trainer_runner` (the trainer's `parse_known_args` discards a bare `--model_type` CLI flag). An absent value on a statistical model produces meaningless language-eval panels. To fix a mislaunched run, set `training_args.model_type` in the model's `config.json`; save reads it fresh each checkpoint and never overwrites it (`model_type` is deliberately excluded from `RUN_ARG_KEYS`).
- **`loss_mask`**: `require_loss_mask_decision` refuses to start on a ChatML-dense mix without an explicit `--loss_mask`. Forgetting it fails silently otherwise — loss falls, the run looks healthy, the model can't answer. Role-masked (assistant-only) loss is opt-in and costs no throughput.
- **Resume does not restore model shape.** `apply_resume_overrides` reads `cfg["training_args"]`, which flat old configs lack — always pass `--size` explicitly on a continue. `total_steps` is absolute, not additional. An omitted `lr_schedule` on resume silently inherits the base's decayed `wsd` tail. Old `config.json` files must keep resuming; trainer changes stay backwards-compatible with them.
- **`resolve_val_path()` follows the heaviest-weighted corpus**, not the first listed.
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
- Recurrent: chunkwise-parallel training over 64-byte chunks (verified vs per-token recurrence to 4e-11). `state_rule`: `gla` (default) | `delta` | `pinned` (both non-default rules are falsified — failures.md). `state_carry=chunks` carries state across windows; `bptt_window` is the memory knob.
- Memory: fast weights are per-sequence state, never in the checkpoint; `forward()` resets per call; `forward_carry()` persists across windows. Not exportable.
- MoE (`hybrid_moe`): 8 routed quarter-width experts + 1 shared, top-2, aux-loss-free bias balancing plus sequence-wise aux loss; MPS-safe routing (iterative argmax, no sort/bool indexing, fp32 bookkeeping).

Exportability (`export.py`): `dense` → v9/v11/v12 int8; `hybrid` → v13 fp16/fp32/int8; everything else refuses early with `ValueError` naming the variant and serves through the PyTorch brain instead. Shape resolution falls back `shape` → `training_args` → top-level flat config. A non-canonical checkpoint pushed through the int8 writer would produce a silent-garbage bin — the refusal is the guard.

### MPS rules

Fixed or bucketed tensor shapes only (dynamic padding recompiles kernels; measured 23x slowdown). In model forward code avoid bool-tensor advanced indexing, int64 `sort`, and in-place writes on freshly indexed tensors — replace with `F.embedding` lookups, int32 sort, out-of-place masks. Prefer op patterns the canonical dense path already exercises. Keep tensors on device; every `.to(device)` round-trip stalls. 8-bit AdamW silently falls back to fp32 on MPS — do not chase it. Stability-smoke any test-time-learning module at real width, not toy width.

## corpus

### locations

Canonical: `data/corpus/` (`paths.CORPUS_ROOT`), holding `<stem>_train.bin` / `<stem>_val.bin` pairs and nothing else — no code, no manifests. Builders and downloads write here via `paths.corpus_train_path()` / `corpus_val_path()`. `trainers/corpus/` is a read-only mirror still resolved second by `corpus_search_dirs()`; it is deletable only when no run holds its bins open and the user approves.

### framing

Record separator across all modes: literal `<|endoftext|>` bytes. Chat mode is ChatML: `<|im_start|>{role}\n...<|im_end|>`; inference hard-stops on `<|im_end|>`. Agent mode is Hermes function calling: `<tool_call>{"name","arguments"}</tool_call>` / `<tool_response>...</tool_response>`, single-line schema-strict JSON. At byte level these markers are literal learned bytes, not reserved token ids. `models/<name>/config.json::capabilities` declares trained tiers (autocomplete < chat < agent, additive; `mark()` never regresses a trained tier; a missing key reads as autocomplete-only). Never invent a framing; these are the canonical ones.

### builders and library

All builders are deterministic from a fixed seed, write to `data/corpus/`, and split val before anything can straddle train/val.

- Code corpus: two-phase `stage` (network → jsonl cache) / `build` (offline, deterministic). Sources pinned (stack-edu, curated tarballs, oa-stackexchange, syntax-gated textbook docs). Per-document stable-hash val bucket (1-in-50), dedup before split.
- Authoring pipeline (teacher-driven): `corpus_spec.json` holds all recipe data; gates run first-failure-wins (JSON parse → schema → turn count → marker → NUL → length → dash policy → banned phrase → exact dup sha1 → opening-cap repeat → simhash near-dup). `simhash64` uses blake2b, not builtin `hash()` (process-salted). `build_sft_corpus.build()` always carves ≥1 conversation into val. A catalog entry writes straight into the shipped `corpus_catalog.json` — rebuilding a published stem breaks its recorded sha256.
- RAG corpus: held-out facts (`TEST_FRAC`) never enter the bins; the test set measures in-context copy on unseen facts. Teacher-required. Reruns with the same stem overwrite bins and the held-out set, and teacher sampling is unseeded, so reruns are not byte-identical.
- Curriculum corpus: generated child-concept corpus for 10M-class models. A procedurally generated corpus has an entropy floor set by generator complexity, not byte count — never use one to measure model capacity or growth.
- Bigram index: `<stem>_train_bigrams.npz` sidecars for the writing-health PMI probe; on-demand builds are capped and cover only the head of a large corpus — capped and `--all` (uncapped) PMI values are not comparable. Tokens are lowercase `[a-z][a-z']*` only.
- Corpus library: catalog merge order local → remote → user sources, later wins. Five install formats (raw_bytes, raw_bytes_zip, zip_bundle, hf_dataset, native); `coming_soon` gates unpublished tiers; `uninstall()` removes only `data/corpus/` copies. Behavior-corpus ladder: chat 50MB/500MB/5GB, agent 15MB/150MB/1.5GB, mcp 2MB/15MB/150MB/1.5GB — behavior data scales with task diversity, not parameter count; Chinchilla governs knowledge volume separately. Tiers over GitHub's 100MB limit ship as `zip_bundle` on Carpathian COS.

### sizing

Chinchilla in bytes: `bytes = batch × seq × n_chunks × total_steps`; divide by 4.55 (prose) or 4.12 (code) for token equivalents before sizing any run. "Correct output shape containing invented words" is the diagnostic signature of undertraining, not an instruction-tuning bug.

## engine

### formats

`.bin` versions v3–v13; compatibility is forward-only, and the full table lives with the loader (`veritate_engine/src/veritate.h`). Load-bearing points: v9 int8 non-MoE dense; v11 adds ternary (5-trit-per-byte base-3 packing, per-row `int32 gamma_q24`, 256-byte LUT decode) and MoE layout (top-1 routing only; per-expert up/down when `n_experts>1`, uniform per-tensor scale); v12 = MTP; v13 = hybrid trunk. v10 is rejected at load. Dense paths v3–v12 require head_dim 64; v13 is head_dim-generic. No Mamba-2 hot path; no speculative decode for v13; MoE with `n_experts>1` + non-INT8 quant refuses at load.

v13 specifics: strictly additive format with its own loader/forward (`hybrid.c`); computes fp32 end-to-end (hybrid checkpoints are not QAT-trained), weights stored fp32/fp16 dequantized exactly at load; ~28MB fixed decode state at h=768/seq=1024 regardless of conversation length; boundary table baked from Python `chr(b).isalnum()` at export, not re-derived in C. `VERITATE_PREFILL_BATCH` defaults off — measured regression on Apple Silicon. Worker count auto-calibrates per box at load; parity holds at any thread count.

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

One route module per concern in `veritate_mri/routes/`. High-traffic groups: `/trainers` + `/trainers/run|stop|tune_defaults|sysprobe`, `/runs/*` (incl. timelines and eval-deep), `/models/*`, `/corpus/*` (incl. `/corpus/mix/plan`), `/settings` + `/settings/notices`, `/sys_metrics`, `/backends`, `/wiki*`, `/addons`, `/atlas/*` (concept/neuron/lifetime/circuit/concepts_inverted — pure derivations over hook dumps), `/teacher/*`, `/mesh/*` (machine-to-machine), `/models/git/*` (models repo sync), `/app/*` (updater). The authoritative settings-key list is `DEFAULTS` in `veritate_mri/runtime/settings.py` — treat the code as the reference; notable keys beyond the obvious: `speculative_enabled`, `speculative_bytes`, `speculative_chunk_bytes`, `speculative_pause_ms`, `read_ahead_enabled`, `api_read_ahead_enabled`, `api_generate_ahead_enabled`, `corpus_compose_chunk_bytes`, `corpus_compose_val_ratio`, `corpus_compose_seed`, `trainer_sizes_path`, `corpus_mix_max_epochs`, `pytorch_load_mode`, `warm_models`, `device_preference`, `api_key`.

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
