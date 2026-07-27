---
title: platform API reference
date: 2026-07-27
tags: [api, extensions, reference]
summary: Every HTTP route the platform server serves, with method, parameters, and response shape.
---

# platform API reference

The complete HTTP contract for the platform server (`veritate_mri/app.py`). This is the stable surface an extension codes against: an extension is a self-contained page that reaches the platform only through these routes.

For the outward-facing surface an outside client uses to reach this box's models, and its optional key gate, see the external API entry. Those routes are listed here in full as well.

Routes are cited by module name, not by line number, so a reference stays correct as the file moves. Every route below is registered in `veritate_mri/routes/<module>.py` unless stated otherwise.

## conventions

- **Base URL:** `http://0.0.0.0:8001`. `--port` overrides the port.
- **Content type:** request bodies are JSON, read with `request.get_json(silent=True)`, except the knowledge-base upload which is `multipart/form-data`. Responses are JSON except CSV, file downloads, HTML pages, and SSE streams.
- **Dashboard auth:** off unless `VERITATE_DASHBOARD_PASSWORD` is set (`auth_routes.py`). When set, `/`, `/login`, `/logout`, `/favicon.ico`, and anything under `/static`, `/chat`, or `/hybrid` stay open; every other route needs a session. An unauthenticated GET redirects to `/login`, an unauthenticated non-GET returns `401 {"ok": false, "error": "authentication required"}`.
- **API-key gate:** separate and independent, covering `/generate`, `/agent/stream`, and every path under `/v1/` (`api_auth_routes.py`). Off until a key is set. Detail in the external API entry.
- **Errors:** every uncaught exception and every Flask HTTP error is serialized to JSON, never an HTML error page. HTTP errors return `{"ok": false, "error": "<message>", "status": <code>}`; uncaught exceptions return `{"ok": false, "error": "<message>"}`. Route-local handlers return `{"ok": false, "error": ...}` with the status noted per endpoint. Treat any non-2xx as failure and read `error`.
- **SSE:** streaming routes set `Content-Type: text/event-stream`, emit `data: <json>` frames, send `: keepalive` comment lines to hold the connection open, and close on client disconnect. Consume with `EventSource` or a streaming fetch reader.

## the subset most extensions need

| purpose | endpoint |
|---|---|
| list loadable models with metadata | `GET /pytorch-models` |
| list models offered for chat | `GET /hybrid/models` |
| current model metadata and shape | `GET /meta` |
| token generation with introspection | `GET /generate` |
| conversational chat with retrieval | `POST /hybrid/chat` |
| backend load, unload, status | `GET /backends`, `POST /backends/pytorch` |
| export a checkpoint to a `.bin` | `POST /export/<name>` |
| read and write settings | `GET /settings`, `POST /settings` |
| read the version ledger | `GET /versions` |
| list training runs | `GET /runs`, `GET /run/<name>/csv` |

Read-only routes are safe to poll. Mutating routes change server or disk state. The `/lifecycle/*`, git-sync, `/engine/build`, and trainer-launch routes drive the platform itself and are not an extension's business.

## models

| method and path | module | params | response |
|---|---|---|---|
| `GET /pytorch-models` | models_routes | none | `{models: [{name, step, is_current, plugin, n_params, hidden, layers, description, mtime, capabilities, engine}]}`, newest first |
| `GET /v1/models` | models_routes | none | `{object:"list", data:[{id, object:"model", created, owned_by:"veritate", n_params, hidden, layers, capabilities, is_current, engine}]}` |
| `POST /models/fork` | models_routes | `source`, `new_name` | `{ok, ...}`; `400` on fork error |
| `POST /models/open_folder` | models_routes | none | `{ok, path}` |
| `GET /models/git/status` | models_routes | none | sync-status dict for the models repo |
| `POST /models/git/sync` | models_routes | `actions`, `branch` | sync-result dict with per-file actions |
| `POST /models/git/check` | models_routes | none | check-result dict |
| `GET /models/git/files` | models_routes | none | per-file table plus per-directory provenance |
| `GET /models/git/progress` | models_routes | none | live byte counter for a running sync |

`engine` is the backend the box will serve that model on: `c` when a `.bin` export and a built engine binary exist, `pytorch` otherwise.

## runs

| method and path | module | params | response |
|---|---|---|---|
| `GET /runs` | runs_routes | none | `{runs:[{name, mtime, size, n_rows, capabilities}]}` |
| `GET /run/<path:name>/csv` | runs_routes | none | `text/csv` body; `404` if absent |
| `GET /run/<path:name>/config` | runs_routes | none | the run's `config.json`; `404` if absent |
| `GET /run/<path:name>/probes` | runs_routes | none | `{name, model_dir, steps:[{step, probe, lens}]}`; `404` |
| `GET /run/<path:name>/classroom` | runs_routes | none | `{name, model_dir, items:[{kind, step, file}]}`; `404` |
| `GET /run/<path:name>/coactivation/<int:step>` | runs_routes | none | `{step, n_tokens, threshold, pairs, nodes}`; `404` |
| `GET /run/<path:name>/learning_rate/<int:step>` | runs_routes | none | `{step, prior_step, neurons:[{layer, neuron, delta, now, prev}]}`; `404` |
| `GET /run/<path:name>/surprise` | runs_routes | none | `{steps, tokens, prompt, surprise, median}`; `404` |
| `GET /run/<path:name>/eval_deep` | runs_routes | none | `{name, results:[{suite, step, file, mtime, n, acc, elapsed_s, by_subject, by_rule, accuracy_letter, accuracy_text}]}`; `404` |
| `GET /run/<path:name>/eval_deep/status` | runs_routes | `step` | `{running, error, suites, progress, ...}` |
| `POST /run/<path:name>/eval_deep` | runs_routes | `suite`, `step`, `limit`, `mmlu_mode`, `ifeval_max_new`, `threads` | `{name, step, suites, files, report}`; `400` bad suite or step, `404` no model, `500` run failure |
| `GET /eval_sets` | runs_routes | none | `{ok, running, started, finished, error, builders:[{builder, rc}], known_builders}` |
| `POST /eval_sets` | runs_routes | none | starts the eval-set rebuild; `{ok, running, known_builders}`; `409` already running |
| `GET /run/<path:name>/timeline` | runs_routes | none | `{name, prompt, max_new, checkpoints:[{step, file, n_frames, output_text, train_loss, val_loss, precision, quant_kl_bits}], description}`; `404` |
| `GET /timelines` | runs_routes | none | `{timelines:[{name, mtime, n_checkpoints, n_pt_checkpoints, has_hooks, prompt, source}]}` |
| `GET /timeline/<path:name>/<path:fname>` | runs_routes | none | one timeline artifact as JSON or bytes; `400` bad name, `404` missing |

## training

| method and path | module | params | response |
|---|---|---|---|
| `GET /train/discovery` | train_routes | none | `{corpora:[...], models:[{name, steps:[int]}]}` |
| `GET /corpus/<path:stem>/usage` | train_routes | none | usage dict; `400` bad stem, `404` not found |
| `GET /train_stream` | train_routes | none | SSE. an `event: ready` frame, then one `data:` frame per training step |
| `GET /trainers` | trainers_routes | none | `{trainers:[...], running:{...}}` |
| `POST /trainers/run` | trainers_routes | `id`, `args` | `{ok, ...}`; `400` missing id, `409` model name already exists, `500` launch error |
| `POST /trainers/stop` | trainers_routes | none | stop-result dict |
| `POST /trainers/tune_defaults` | trainers_routes | `id`, `args`, `measured`, `sysprobe` | `{ok, saved, upload}`; `400` missing id. Writes machine-local tuning, never the upstream manifest |
| `POST /trainers/sysprobe` | trainers_routes | `disk_dir` | `{ok, sysprobe}`. Runs the hardware benchmark suite: disk write, CPU throughput and bandwidth, GPU throughput on every accelerator, memory headroom |
| `GET /core_trainers` | trainers_routes | `flow` | `{trainers:[...]}` |
| `GET /trainers/git/status` | trainers_routes | none | git-status dict for the trainers checkout |
| `POST /trainers/git/sync` | trainers_routes | `actions`, `branch` | sync-result dict |
| `POST /trainers/git/check` | trainers_routes | none | check-result dict |
| `GET /trainers/git/files` | trainers_routes | none | per-file table with three-state classification |
| `POST /trainers/open_folder` | trainers_routes | none | `{ok, path}` |

`args` on `POST /trainers/run` carries the training settings: `name`, `size`, `corpus`, `total_steps`, `batch_size`, `seq`, and the rest. Each setting has its own wiki entry under `settings/`.

## corpus

| method and path | module | params | response |
|---|---|---|---|
| `GET /corpus/library/catalog` | corpus_routes | none | catalog dict |
| `POST /corpus/library/install` | corpus_routes | install params | install-result dict |
| `POST /corpus/library/install_deps` | corpus_routes | none | deps-result dict |
| `POST /corpus/library/uninstall` | corpus_routes | `stem` | uninstall-result dict |
| `POST /corpus/library/catalog_url` | corpus_routes | `url` | result dict |
| `POST /corpus/library/sources/add` | corpus_routes | source params | add-result dict |
| `POST /corpus/library/sources/remove` | corpus_routes | `stem` | remove-result dict |
| `GET /corpus/mix/profiles` | corpus_routes | none | `{ok, path, profiles:[{name, label, topics, stems}]}`; `400` on an unreadable profiles file |
| `POST /corpus/mix/plan` | corpus_routes | `stems` (required, unique), `target_bytes` (required), `profile`, `max_epochs`, `model_params`, `weights` | `{ok, spec, sources:[{stem, label, topic, weight, bytes_drawn, bytes_available, epochs}], warnings, bytes_planned, inputs}`; `400` bad body, unknown profile, or no data |
| `POST /corpus/open_folder` | corpus_routes | none | `{ok, path}` |

A mix profile is a named intent (pretraining, instruction tuning) that supplies target topic proportions. `spec` from a plan is what the trainer's `--corpus` argument accepts.

## generation

`/generate` and `/agent/stream` are the primary inference surface. Both are SSE and both are covered by the API-key gate.

| method and path | module | params | response |
|---|---|---|---|
| `GET /generate` | backends_routes | `prompt`, `temperature` (0.7), `top_k` (40, clamped 1..256), `max_new` (200, capped 4096), `backend` (`c` default, or `pytorch`), `ablate_layer`, `ablate_neuron`, `addons` (csv), `fast` (`kv`, `mtp`, `mtp-verify`, `adaptive`), `constrained`, `adaptive_threshold`, `rep_window`, `rep_penalty`, `no_repeat_ngram`, `rag` (corpus path, loopback only), `rag_k` (max 16), `rag_compress`, `prefetch_id` (claims a `/prefetch` draft) | SSE, described below |
| `GET /agent/stream` | backends_routes | `prompt` (required), `max_turns` (6, max 16), `best_of_n` (1, max 8), `temperature`, `top_k`, `seed`, `corpus` (loopback only), `fs_root` (loopback only), `tools` (csv) | SSE. an `agent_meta` frame, then per-turn frames, then `done`. `400` missing prompt, bad path, or no usable tools; `403` loopback-only param from a remote client |
| `POST /prefetch` | backends_routes | `prompt` (the exact wire prompt submit will send; omit to stand down), plus the decode knobs submit will use: `temperature`, `top_k`, `ablate_layer`, `ablate_neuron`, `addons`, `rep_window`, `rep_penalty`, `no_repeat_ngram` | `{speculating, buffered, draft_id, stats}`, plus `reason` when the feature is off or the C engine is unloaded |
| `GET /prefetch` | backends_routes | none | `{speculating, buffered, draft_id, prompt, text, stats}` for the live draft; reading never consumes it |
| `GET /addons` | backends_routes | none | `{addons:[...]}` |
| `GET /meta` | backends_routes | none | current model and engine configuration |
| `GET /neuron/<int:layer>/<int:nid>` | backends_routes | none | `{layer, neuron, stories, affinity, predecessors, successors, stats, label, pytorch_loaded, pytorch_last_error}` |
| `GET /backends` | backends_routes | none | `{pytorch:{loaded, pending, model, step, last_error}, c:{loaded, pending, exe, model_bin, model_dir, blocked_reason, blocked_model, build, bins_available}}` |
| `POST /backends/pytorch` | backends_routes | `action` (`load`, `unload`), `model`, `step`, `threads` | same shape as `GET /backends`; `400` no models or unsupported trunk, `500` load error |
| `POST /backends/c` | backends_routes | `action` (`load`, `unload`), `model` | same shape as `GET /backends`; `400` invalid action |

The `/generate` stream opens with a `meta` frame (`checkpoint, n_params, layers, heads, ffn, vocab, seq, hidden, has_memory, prompt, prompt_bytes, backend, c_exe, c_model, c_model_dir, c_model_path`), an optional `rag` frame when retrieval ran, then one `kind:"token"` frame per generated byte (`byte, argmax_byte, T, fwd_ms, entropy_bits, surprise_bits, ffn_full, ffn_top, ffn_argmax, ffn_downsample, decisiveness, dla_picked, dla_argmax, dla_cand, ablation, margin, entropy, lens_consistency, residual_stab, confidence, attn, info_flow, res, contrib, lens, cand, memory, backend`), then `kind:"stop"` with a reason, then `done`.

When a `/prefetch` draft matches the request exactly, the stream carries a `kind:"prefetch"` frame (`bytes`) after `meta`, then one `kind:"fast_byte"` frame per prefetched byte carrying `prefetched:true`. Those bytes were generated untraced, so they have no telemetry; the `kind:"token"` frames resume at the first live byte.

A non-numeric `temperature`, `top_k`, `max_new`, or ablation index yields a `kind:"error"` SSE frame followed by `done` rather than a 500, so an `EventSource` client stays parseable. `400` covers an invalid `fast`, `rag_k`, or `rag_compress` and a missing corpus. `403` covers `rag=` from a non-loopback client.

## chat

The chat page is served at `/chat` and embedded by the dashboard chat tab. `/hybrid/chat` answers with a local byte model or a configured teacher, with optional retrieval over an uploaded knowledge base and conversation memory.

| method and path | module | params | response |
|---|---|---|---|
| `GET /hybrid/health` | hybrid_routes | none | `{ok, has_corpus, n_files}` |
| `GET /hybrid/models` | hybrid_routes | none | `{models:[{id, label, group, provider, model}]}` |
| `POST /hybrid/kb/upload` | hybrid_routes | `multipart/form-data` field `file` | `{ok, filename, n_files, n_chunks}`; `400` no file, `413` over 64 MB, `500` save error |
| `POST /hybrid/chat` | hybrid_routes | `message` (required), `model`, `backend`, `use_rag`, `use_logs`, `kb_scope` (`all`, `platform`, `user`), `k`, `history`, `summary`, `mri` | `{ok, answer, model, backend, confident, sources, memory, context}`; `400` empty message, `503` model or provider unavailable, `500` error |
| `POST /hybrid/chat/stream` | hybrid_routes | same body as `/hybrid/chat` | SSE. `{kind:"delta", text}` frames, then one `{kind:"done", ...}` frame. A mid-stream failure arrives as `{kind:"error", error}` with the status already 200 |
| `POST /v1/chat/completions` | hybrid_routes | `model`, `messages` (required, ends on a user turn), `stream`, `temperature`, `max_tokens`, `top_k`, `rep_window`, `rep_penalty`, `no_repeat_ngram`, `mri` | OpenAI `chat.completion`, or an SSE stream of `chat.completion.chunk` frames ending in `data: [DONE]` |
| `POST /v1/chat/mri` | hybrid_routes | same body as `/v1/chat/completions`; `stream` defaults to true | OpenAI-shaped frames that always carry per-byte telemetry; `400` on a non-local model, `503` model unavailable |

`mri: true` on a local model adds the per-byte telemetry frames to the response: interleaved as extra SSE frames when streaming, under a top-level `mri` key when not. A remote model exposes no telemetry and the flag has no effect there. `/v1/chat/mri` is the always-on variant of the same behavior and rejects a remote model with `400`.

## hallucination

| method and path | module | params | response |
|---|---|---|---|
| `POST /hallucination/analyze` | hallucination_routes | `model` (required), `backend`, `prompt` or `message`, `use_rag`, `kb_scope`, `k`, `max_new`, `temperature`, `frames` | `{ok, answer, overall, spans, provenance, context, confidence_source}`; `400` missing model or prompt, `503` backend unavailable |

The route generates an answer through the same backends as `/generate`, then scores it: per-byte confidence rolled up to word, sentence, and paragraph spans with byte-exact offsets; overlap against the retrieved context; divergence between the answer with and without context; and nearest training passages as a similarity proxy, not a causal attribution.

`overall.verdict` is one of `refused`, `grounded`, `likely_hallucinated`, `partially_grounded`, `low_confidence`, or `ungrounded_ok`. `grounded_fraction` and `context_divergence` are null when no context was retrieved. Passing `frames` grades an answer the client already streamed, with no regeneration.

## teacher

Teacher routes configure and drive a remote frontier model used for synthetic-data generation and chat fallback. Providers: carpathian, openai, anthropic, gemini, xai, deepseek, mistral, groq, openrouter, ollama, lm_studio, llama_cpp.

| method and path | module | params | response |
|---|---|---|---|
| `GET, POST /teacher` | teacher_routes | POST `teacher_provider`, `teacher_api_key`, `teacher_model`, `teacher_base_url`, `teacher_configs` | `{providers, configured, provider, model, base_url, has_api_key, configs, max_concurrency, max_tokens, temperature}`; `400` invalid value |
| `POST /teacher/test` | teacher_routes | `provider`, `model`, `base_url`, `api_key` | `{ok, model, latency_ms}` or `{ok:false, error}` |
| `POST /teacher/models` | teacher_routes | `provider`, `base_url`, `api_key` | `{models:[str]}`; `400` no provider |
| `POST /teacher/complete` | teacher_routes | `prompt` (required), `system`, provider overrides, `max_tokens`, `temperature` | `{ok, text, provider, model}`; `400` no prompt or no teacher, `502` provider error |
| `POST /teacher/synth/start` | teacher_routes | `prompts` (required), `format`, `seed_ids`, `job_id`, `output_dir`, provider overrides | `{job_id, output_dir}`; `400` bad prompts, `409` job running |
| `POST /teacher/synth/stop` | teacher_routes | `job_id` | `{job_id, stopping}`; `404` unknown job |
| `GET /teacher/synth/jobs` | teacher_routes | none | `{jobs:[{job_id, completed, categories, seeds, running}]}` |
| `GET /teacher/synth/status` | teacher_routes | `job_id` | `{job_id, running, completed, failed, skipped_dup, last_error, error_summary, aborted, authoring, output_path}`; `404` |
| `GET /teacher/synth/samples` | teacher_routes | `job_id`, `limit` (default 20, max 100) | `{job_id, samples:[{id, response}]}`; `404` |
| `POST /teacher/synth/delete` | teacher_routes | `job_id` | `{job_id, deleted}`; `400` missing, `409` running, `404` unknown |
| `POST /teacher/synth/build_corpus` | teacher_routes | `job_id`, `stem` | `{stem, train_bin, val_bin, n_records, n_train, n_val}`; `400` bad stem, `404` no samples |
| `GET, POST /teacher/authoring/spec` | teacher_routes | POST the full spec object, `genres` and `gates` required | the stored spec; `400` on a spec missing either key |
| `POST /teacher/authoring/import` | teacher_routes | `source_dir` (required, must exist), `job_id` | `{job_id, output_dir, files:[{file, accepted, rejected}], accepted_total, ngram_ratio, ngram_floor, ngram_below_floor}`; `400` bad directory, `409` job still running |
| `POST /teacher/authoring/start` | teacher_routes | `genres` (required), `target_mb` (required), `ngram_distinct_floor`, `max_concurrency`, `job_id` | `{job_id, output_dir, calls, total_calls, max_concurrency}`; `400` no genre, no target, or no teacher, `409` job running |
| `POST /teacher/authoring/build` | teacher_routes | `job_id`, `stem` (required), `label`, `description`, `recommended_min_params`, `recommended_max_params` | `{stem, zip_path, zip_bytes, zip_sha256, family_counts, manifest, catalog_entry, next_steps}`; `400` bad stem or no records, `404` no samples |
| `GET /teacher/seeds` | teacher_routes | none | `{version, seeds:[...], total_count}` |
| `GET /teacher/seeds/<seed_id>` | teacher_routes | none | `{id, count, prompts:[dict]}`; `404` unknown |

`/teacher/authoring/import` reads every `.jsonl` file in `source_dir` and appends the records that pass the authoring quality gate to a job's sample file, so externally generated text enters through the same gate as generated text. Records that fail the gate are counted and dropped.

## retrieval training

| method and path | module | params | response |
|---|---|---|---|
| `POST /rag/build_corpus` | rag_routes | `n_facts` (> 0) | `{ok, phase:"build_corpus", stem:"rag_ui"}`; `400` bad count, `409` already running |
| `POST /rag/train` | rag_routes | `source` (required), `name`, `steps`, `n_facts` | `{ok, phase, name, auto_built}`; `400` no source, `409` already running |
| `POST /rag/stop` | rag_routes | none | `{ok, running}` |
| `GET /rag/status` | rag_routes | none | `{ok, status, phase, started_at, finished_at, exit_code, running, log}` |

## interpretability

Read-only neuron and concept introspection over a run's dump artifacts.

| method and path | module | params | response |
|---|---|---|---|
| `GET /atlas/concept` | atlas_routes | `model`, `step`, `substring`, `top_k` | neurons that fire for a concept; `400` bad model |
| `GET /atlas/neuron/<int:layer>/<int:neuron>` | atlas_routes | `model`, `step`, `top_k` | concepts one neuron tracks; `400` bad model |
| `GET /atlas/lifetime/<int:layer>/<int:neuron>` | atlas_routes | `model` | how a neuron changed across checkpoints; `400` bad model |
| `GET /atlas/circuit` | atlas_routes | `layer`, `top_k` | circuit graph for a layer |
| `GET /atlas/concepts_inverted` | atlas_routes | `model`, `step` | inverted concept index; `400` bad model |

## settings

| method and path | module | params | response |
|---|---|---|---|
| `GET, POST /settings` | settings_routes | POST a patch of any settings keys | the merged settings object; `400` on an invalid value |
| `GET /settings/notices` | settings_routes | none | `{notices:[...]}` pending build notices |
| `POST /settings/api-key` | settings_routes | `action` (`rotate`, `clear`) | the merged settings object; `400` bad action |
| `POST /ai/ask` | settings_routes | `kind`, `payload` | varies by `kind` |

Settings keys are defined in `DEFAULTS` in `veritate_mri/runtime/settings.py`: `pytorch_load_mode`, `pytorch_idle_unload_secs`, `warm_models`, `hud_enabled`, `hud_position`, `hud_detailed`, `mri_compact_frames`, `temperature_unit`, `heartbeat_enabled`, `heartbeat_send_errors`, `consent_modal_seen`, `analytics_advanced_enabled`, `share_current_training`, `diagnostics_logs_enabled`, `device_preference`, `update_channel`, `auto_reload_on_update`, `extensions`, `ai_enabled`, `ai_endpoint_user`, `ai_api_key_user`, `last_acknowledged_build`, `device_name`, `corpus_catalog_url`, `corpus_user_sources`, `corpus_mix_max_epochs`, `corpus_mix_default_profile`, `corpus_mix_profiles_path`, `teacher_provider`, `teacher_model`, `teacher_base_url`, `teacher_api_key`, `teacher_configs`, `teacher_max_concurrency`, `teacher_max_tokens`, `teacher_temperature`, `mesh_role`, `mesh_hub_address`, `mesh_auth_token`, `tutorial_enabled`, `tutorial_completed`, `api_key`, `api_key_request_count`, `api_key_last_used_at`.

The `extensions` flag controls the extension nav links and the marketplace entry in the dashboard. The `/extensions/*` routes register regardless of it.

## engine and export

| method and path | module | params | response |
|---|---|---|---|
| `GET /engine/status` | engine_routes | none | `{status, error, c_subprocess_running, c_exe, ...}` |
| `POST /engine/build` | engine_routes | `force` | build-state dict |
| `GET /c-engines` | engine_routes | none | `{engines:[{version, path, is_current, mtime, size}]}` |
| `GET /c-models` | engine_routes | none | `{models:[{name, bin_path, is_current, mtime, size, precision, bin_version, training, activation, act_boost, qat_enabled, description}]}` |
| `POST /c-config` | engine_routes | `exe`, `model` | selected-configuration dict; `400` not found, `500` respawn failed |
| `GET /pruning/report` | pruning_routes | `model` (required), `step`, `samples` | `{ok, model, step, corpus, samples, n_params, n_params_after, size_mb_before, size_mb_after, dead_pct, per_layer, plan}`; `400` bad model or corpus |
| `POST /pruning/generate_plugin` | pruning_routes | `model`, `step`, `plan`, `samples` | `{ok, plugin_id, plugin_dir}`; `400` invalid or unsupported trunk |
| `POST /export/<name>` | pruning_routes | `step` | `{ok, path, bytes, ...}`; `400` no checkpoints or export error, `404` no model |

## system

| method and path | module | params | response |
|---|---|---|---|
| `GET /sys_metrics` | sys_routes | none | CPU, memory, and temperature snapshot |
| `GET /sys/mode` | sys_routes | none | `{minimal: bool}` |
| `POST /sys/mode/relaunch` | sys_routes | `minimal` | restart-result dict |
| `GET /sys/specs` | sys_routes | none | saved hardware specs, or `{detected:false}` |
| `POST /sys/detect` | sys_routes | none | detect-result dict, including dependency status |
| `POST /system/install_dep` | trainers_routes | `pkg` (required), `index_url` | `{ok, method, needs_elevation, stdout, stderr}`; `400` missing package. Installs a missing Python package, escalating if needed, and blocks until the install returns |
| `POST /system/install_helper` | trainers_routes | `helper` (required) | `{ok, ...}`; `400` missing helper. Installs a native OS helper such as a temperature sensor reader |
| `GET /heartbeat/status` | sys_routes | none | heartbeat tier and last-send status |
| `POST /heartbeat/send` | sys_routes | none | `{ok, ...}` |
| `GET /heartbeat/preview` | sys_routes | none | the payload a heartbeat would send |
| `GET /app/update_status` | sys_routes | none | update channel and pending-update state |
| `POST /app/update_check` | sys_routes | none | check-result dict |
| `POST /app/update_pull` | sys_routes | `force`, `ignore_training`, `reload` | pull-result dict |
| `GET /app/local_edits` | sys_routes | none | files diverging from the pulled baseline |
| `POST /app/update_channel` | sys_routes | `channel` | switch-result dict |
| `GET /versions` | sys_routes | none | the `versions.json` body; `404` if missing |
| `GET /logs/snapshot` | logs_routes | `after`, `limit` | `{latest_seq, entries:[...]}` |
| `GET /logs/stream` | logs_routes | none | SSE. one JSON log entry per frame, with keepalives |
| `POST /lifecycle/restart` | lifecycle_routes | none | restart-result dict |
| `POST /lifecycle/kill` | lifecycle_routes | none | kill-result dict |
| `POST /lifecycle/soft_reload` | lifecycle_routes | none | soft-reload-result dict |

## mesh

Distributed-training mesh control. Roles are `off`, `node`, `hub`, and `both`.

| method and path | module | params | response |
|---|---|---|---|
| `GET /mesh/status` | mesh_routes | none | `{role, hub_address, has_token, node_registered, last_heartbeat, current_job, hub_nodes}` |
| `GET /mesh/token` | mesh_routes | none | `{has_token, token}`, unmasked only for a loopback caller |
| `POST /mesh/token/regenerate` | mesh_routes | none | `{ok, token}` |
| `POST /mesh/test_connection` | mesh_routes | `hub_address`, `auth_token` | `{ok, status, error, response_ms}` |
| `POST /mesh/role` | mesh_routes | `role` | `{ok, role, restart_required}`; `400` invalid role |

Hub and node routes register only when the role selects them, from `veritate_mesh/hub.py` and `veritate_mesh/node.py`. All of them require a bearer token.

| method and path | module | role |
|---|---|---|
| `POST /mesh/register` | veritate_mesh/hub.py | hub |
| `POST /mesh/heartbeat` | veritate_mesh/hub.py | hub |
| `GET /mesh/job/next` | veritate_mesh/hub.py | hub, long-polls and returns `204` when the queue is empty |
| `POST /mesh/job/<job_id>/progress` | veritate_mesh/hub.py | hub |
| `POST /mesh/job/<job_id>/result` | veritate_mesh/hub.py | hub |
| `GET /mesh/hub/nodes` | veritate_mesh/hub.py | hub |
| `GET /mesh/hub/jobs` | veritate_mesh/hub.py | hub |
| `POST /mesh/hub/submit` | veritate_mesh/hub.py | hub |
| `GET /mesh/node/status` | veritate_mesh/node.py | node |

## extensions

Platform routes that list installed extensions, read the marketplace catalog, install and uninstall, and manage each extension's optional datasets. All disk work belongs to the `extensions` package; these routes are a thin wrapper.

| method and path | module | params | response |
|---|---|---|---|
| `GET /extensions` | extensions_routes | none | `{ok, extensions:[{id, name, version, nav_label, route, experimental}]}` |
| `GET /extensions/catalog` | extensions_routes | none | `{ok, catalog:[{id, name, version, author, description, installed, ...}]}` |
| `POST /extensions/install` | extensions_routes | `id` (required) | `{ok, extension:{id, installed:true}}`; `400` missing id, `404` no source |
| `POST /extensions/uninstall` | extensions_routes | `id` (required) | `{ok}`; `400` missing id |
| `GET /extensions/<ext_id>/data` | extensions_routes | none | `{ok, datasets:[{source, label, description, url, approx_gb, schema, present, files, size_gb, downloadable}]}` |
| `POST /extensions/<ext_id>/data/download` | extensions_routes | `source` (required) | `{ok, source, files, size_gb}` or `{ok:false, error}`; `400` missing source |
| `POST /extensions/<ext_id>/data/delete` | extensions_routes | `source` (required) | `{ok, source, deleted, reclaimed_gb}` or `{ok:false, error}`; `400` missing source |

An installed extension also contributes its own routes, registered at startup by `extensions/registry.py::register_all`. Those appear at runtime and are documented by the extension that owns them, not here.

## wiki

| method and path | module | params | response |
|---|---|---|---|
| `GET /wiki` | wiki_routes | none | `{categories:[{name, n_entries}]}` |
| `GET /wiki/<category>` | wiki_routes | none | `{category, entries:[...]}`; `404` unknown category |
| `GET /wiki/<category>/<slug>` | wiki_routes | none | the entry as JSON, including `body_html`; `404` unknown entry |
| `GET /wiki/<category>/<slug>/page` | wiki_routes | none | a standalone styled HTML render of the entry, for opening in a new tab; `404` unknown entry |

Category names must match `^[a-z0-9_]+$` and slugs `^[a-z0-9_\-]+$`. Anything else is treated as not found.

## pages and auth

| method and path | module | response |
|---|---|---|
| `GET /`, `GET /app` | app.py | the dashboard, `index.html` |
| `GET /chat` | app.py | the standalone chat page, `hybrid.html` |
| `GET /static/<path:filename>` | Flask static | assets from `veritate_mri/web/` |
| `GET, POST /login` | auth_routes | the login page; POST checks the `password` form field and redirects to `/app` on a match |
| `GET /logout` | auth_routes | clears the session and redirects to `/` |
