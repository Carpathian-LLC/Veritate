# rest api reference

Complete HTTP contract for the Veritate platform server (`veritate_mri/app.py`). This is the stable surface that downloadable extensions code against: extensions are self-contained pages that talk to the platform only through these routes.

- **Base URL:** `http://0.0.0.0:8001` (default port; `--port` overrides it, `veritate_mri/app.py:183`).
- **Content type:** request bodies are JSON (`request.get_json(silent=True)`) unless noted (knowledge-base upload is `multipart/form-data`). Responses are JSON unless noted (CSV, file downloads, SSE).
- **Auth:** off by default. A password gate activates only when `VERITATE_DASHBOARD_PASSWORD` is set (`veritate_mri/routes/auth_routes.py:33`). When enabled, the public surface (`/`, `/login`, `/logout`, `/favicon.ico`, `/static/*`, `/chat*`, `/hybrid*`) stays open; every other route requires a session. Unauthenticated `GET` redirects to `/login`; unauthenticated non-`GET` returns `401 {"ok": false, "error": "authentication required"}` (`veritate_mri/routes/auth_routes.py:49`).
- **Error convention:** every uncaught exception and every Flask HTTP error is serialized to JSON, never an HTML error page. Shape is `{"ok": false, "error": "<message>", "status": <code>}` for HTTP errors and `{"ok": false, "error": "<message>"}` for uncaught exceptions (`veritate_mri/app.py:86`). Route-local handlers also return `{"ok": false, "error": ...}` with `400`/`404`/`409`/`500`/`502`/`503` as noted per endpoint. The `safe_route` wrapper (`veritate_mri/routes/_common.py:26`) gives the same `500` envelope. Treat any non-2xx as failure and read `error`.
- **SSE:** streaming endpoints set `Content-Type: text/event-stream`, emit `data: <json>\n\n` frames, send `: keepalive\n\n` comment lines to hold the connection, and close on client disconnect. Consume with `EventSource` or a streaming fetch reader.
- **Note on CLI:** the platform never surfaces shell commands as user-facing errors; error strings are plain language.

## endpoints an extension will typically use

The market reference extension (`veritate_mri/routes/market_routes.py`) demonstrates the pattern. Most extensions need only this subset:

| purpose | endpoint |
|---|---|
| list loadable models with metadata | `GET /pytorch-models` |
| list models for chat (local + remote) | `GET /hybrid/models` |
| current model metadata + shape | `GET /meta` |
| token generation with introspection (SSE) | `GET /generate` |
| conversational chat with RAG/teacher | `POST /hybrid/chat` |
| backend load/unload + status | `GET /backends`, `POST /backends/pytorch` |
| export a checkpoint to a `.bin` | `POST /export/<name>` |
| read/write user settings | `GET /settings`, `POST /settings` |
| read version ledger | `GET /versions` |
| list trained runs | `GET /runs`, `GET /run/<name>/csv` |
| extension catalog / install / remove | `GET /market/extensions/catalog`, `POST /market/extensions/download`, `POST /market/extensions/delete` |

Read-only endpoints (`GET` on `/pytorch-models`, `/meta`, `/runs`, `/run/*`, `/settings`, `/versions`, `/backends`, the `/atlas/*` family, `/wiki*`, `/market/*` reads) are safe to poll. Mutating endpoints (`/backends/pytorch`, `/trainers/run`, `/export/*`, `/settings` POST, `/lifecycle/*`, the git-sync routes) change server or disk state. The `/lifecycle/*`, git-sync, `/engine/build`, and trainer-launch routes are platform-internal; an extension should not drive them.

## models and runs

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /pytorch-models` | `models_routes.py:70` | none | `{models: [{name, step:int, is_current:bool, plugin:str, n_params:int\|null, hidden, layers, description:str, mtime:float, capabilities:dict}]}`, sorted newest first | model picker list with shape + capability metadata |
| `POST /models/fork` | `models_routes.py:57` | body `source:str`, `new_name:str` | `{ok, ...}`; `400 {ok:false,error}` on `ForkError` | copy latest checkpoint of `source` into a new model dir |
| `POST /models/open_folder` | `models_routes.py:66` | none | `{ok, path}` | open the models root in the OS file browser |
| `GET /models/git/status` | `models_routes.py:32` | none | sync-status dict | git status of the models repo |
| `POST /models/git/sync` | `models_routes.py:36` | body `actions:dict`, `branch:str` (both optional) | sync-result dict | pull/push the models repo |
| `POST /models/git/check` | `models_routes.py:43` | none | check-result dict | verify models repo integrity |
| `GET /models/git/files` | `models_routes.py:47` | none | per-file table + per-dir provenance | file-level diff state for the models repo |
| `GET /models/git/progress` | `models_routes.py:52` | none | byte-counter dict | live progress of an active models sync |
| `GET /runs` | `runs_routes.py:256` | none | `{runs: [{name, mtime, size, n_rows, capabilities}]}` | list training runs under `models/` |
| `GET /run/<path:name>/csv` | `runs_routes.py:274` | none | `text/csv` body; `404` if absent | per-step training metrics CSV |
| `GET /run/<path:name>/config` | `runs_routes.py:306` | none | `application/json` config; `404` if absent | the run's `config.json` |
| `GET /run/<path:name>/probes` | `runs_routes.py:281` | none | `{name, model_dir, steps:[{step, probe, lens}]}`; `404` | probe + logit-lens artifacts per step |
| `GET /run/<path:name>/classroom` | `runs_routes.py:294` | none | `{name, model_dir, items:[{kind, step, file}]}`; `404` | classroom-suite artifacts |
| `GET /run/<path:name>/coactivation/<int:step>` | `runs_routes.py:314` | none | `{step, n_tokens, threshold, pairs, nodes}`; `404` | neuron coactivation graph at a step |
| `GET /run/<path:name>/learning_rate/<int:step>` | `runs_routes.py:320` | none | `{step, prior_step, neurons:[{layer, neuron, delta, now, prev}]}`; `404` | per-neuron weight deltas between steps |
| `GET /run/<path:name>/surprise` | `runs_routes.py:326` | none | `{steps, tokens, prompt, surprise, median}`; `404` | token-surprise atlas over training |
| `GET /run/<path:name>/eval_deep` | `runs_routes.py:332` | none | `{name, results:[{suite, step, file, mtime, n, acc, elapsed_s, by_subject, by_rule, accuracy_letter, accuracy_text}]}`; `404` | cached deep-eval results |
| `GET /run/<path:name>/eval_deep/status` | `runs_routes.py:377` | `step` (optional) | eval-state dict `{running, error, suites, progress, ...}` | status of a running/finished deep eval |
| `POST /run/<path:name>/eval_deep` | `runs_routes.py:388` | body `suite:str\|list`, `step:int\|"latest"`, `limit:int`, `mmlu_mode:str`, `ifeval_max_new:int`, `threads:int` | `{name, step, suites, files, report}`; `400` bad suite/step, `404` no model, `500` run failure | trigger a deep evaluation on a checkpoint |
| `GET /run/<path:name>/timeline` | `runs_routes.py:545` | none | `{name, prompt, max_new, checkpoints:[{step, file, n_frames, output_text, train_loss, val_loss, precision, quant_kl_bits}], description}`; `404` | full per-checkpoint timeline payload |
| `GET /timelines` | `runs_routes.py:497` | none | `{timelines:[{name, mtime, n_checkpoints, n_pt_checkpoints, has_hooks, prompt, source}]}` | timeline index (compat) |
| `GET /timeline/<path:name>/<path:fname>` | `runs_routes.py:522` | none | JSON dict or binary file by `fname`; `400` bad name, `404` missing | fetch a single timeline artifact (compat) |

## training

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /train/discovery` | `train_routes.py:28` | none | `{corpora:[...], models:[{name, steps:[int]}]}` | available corpora + models with checkpoints |
| `GET /corpus/<path:stem>/usage` | `train_routes.py:41` | none | usage dict; `400` bad stem, `404` not found | per-stem corpus usage stats |
| `GET /train_stream` | `train_routes.py:50` | none | **SSE.** `event: ready` then `data: <payload>` frames; `: keepalive` between writes | live per-step training feed (tier 4) |
| `GET /trainers` | `trainers_routes.py:26` | none | `{trainers:[...], running:{...}}` | trainer catalog + current run state |
| `POST /trainers/run` | `trainers_routes.py:33` | body `id:str`, `args:{name, size, resume, base_ckpt, ...}` | `{ok, ...}`; `400` missing id, `409` model name exists, `500` error | start a trainer run |
| `POST /trainers/stop` | `trainers_routes.py:57` | none | stop-result dict; `500` error | stop the current trainer run |
| `POST /trainers/tune_defaults` | `trainers_routes.py:61` | body `id:str`, `args:dict`, `measured:dict` | `{ok, manifest_updated:bool}`; `400` missing id, `500` error | write auto-tune results into a trainer manifest |
| `GET /core_trainers` | `trainers_routes.py:77` | `flow` (optional) | `{trainers:[...]}`; `500` error | core trainer index, optionally filtered by flow |
| `GET /trainers/git/status` | `trainers_routes.py:85` | none | git-status dict; `500` | trainers repo git status |
| `POST /trainers/git/sync` | `trainers_routes.py:89` | body `actions:dict`, `branch:str` | sync-result dict; `500` | sync the trainers repo (canonical upstream) |
| `POST /trainers/git/check` | `trainers_routes.py:98` | none | check-result dict; `500` | verify trainers repo integrity |
| `GET /trainers/git/files` | `trainers_routes.py:102` | none | per-file table; `500` | file-level diff state for the trainers repo |
| `POST /trainers/open_folder` | `trainers_routes.py:107` | none | `{ok, path}`; `500` | open the trainers root in the OS file browser |
| `GET /corpus/library/catalog` | `corpus_routes.py:28` | none | catalog dict | corpus-library catalog |
| `POST /corpus/library/install` | `corpus_routes.py:32` | body (install params) | install-result dict | install a corpus from the library |
| `POST /corpus/library/install_deps` | `corpus_routes.py:37` | none | deps-result dict | pip-install HuggingFace dataset deps |
| `POST /corpus/library/uninstall` | `corpus_routes.py:43` | body `stem:str` | uninstall-result dict | remove an installed corpus |
| `POST /corpus/library/catalog_url` | `corpus_routes.py:48` | body `url:str` | result dict | set a custom catalog URL |
| `POST /corpus/library/sources/add` | `corpus_routes.py:53` | body (source params) | add-result dict | add a user-defined corpus source |
| `POST /corpus/library/sources/remove` | `corpus_routes.py:58` | body `stem:str` | remove-result dict | remove a user-defined corpus source |
| `POST /corpus/open_folder` | `corpus_routes.py:63` | none | `{ok, path}` | open the corpus root in the OS file browser |

## inference / generate

`/generate` and `/agent/stream` are the primary inference surface. Both are SSE. `/generate` drives either the C engine (`backend=c`, default) or the PyTorch brain (`backend=pytorch`) and emits per-byte MRI telemetry.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /generate` | `backends_routes.py:666` | `prompt:str`, `temperature:float=0.7`, `top_k:int=40`, `max_new:int=200`, `backend:str=c`, `ablate_layer:int=-1`, `ablate_neuron:int=-1`, `addons:str` (csv), `fast:str` (`kv`\|`mtp`\|`mtp-verify`\|`adaptive`), `constrained:str`, `adaptive_threshold:float=0.8`, `rag:str` (corpus path), `rag_k:int=3` (max 16), `rag_compress:str` (`off`\|`crude`\|`word_ppl[:keep_frac]`) | **SSE.** First a `meta` frame (`checkpoint, n_params, layers, heads, ffn, vocab, seq, hidden, has_memory, prompt, prompt_bytes, backend, c_exe, c_model, c_model_dir, c_model_path`); an optional `rag` frame (`backend, corpus, top_k, hits, prefix_bytes, prefix_text, compress`); then `kind:"token"` frames (`byte, argmax_byte, T, fwd_ms, entropy_bits, surprise_bits, ffn_full, ffn_top, ffn_argmax, ffn_downsample, decisiveness, dla_picked, dla_argmax, dla_cand, ablation, margin, entropy, lens_consistency, residual_stab, confidence, attn, info_flow, res, contrib, lens, cand, memory, backend`); ends with `kind:"stop"` (`reason`) then `done`. `400` invalid `fast`/`rag_k`/`rag_compress`/missing corpus, `500` stream error | byte-by-byte generation with full interpretability telemetry, optional ablation, RAG prefix, decode addons, and constrained decoding |
| `GET /agent/stream` | `backends_routes.py:851` | `prompt:str` (required), `max_turns:int=6` (max 16), `best_of_n:int=1` (max 8), `temperature:float=0.7`, `top_k:int=40`, `seed:int=0`, `corpus:str` (path), `fs_root:str` (path), `tools:str` (csv) | **SSE.** `agent_meta` frame (`tools, max_turns, best_of_n`), then per-turn agent frames, then `done`. `400` missing prompt / bad path / no usable tools | stream a full agent trace (tool calls + reasoning) |
| `GET /addons` | `backends_routes.py:658` | none | `{addons:[...]}`; `500` | list available decode addons |
| `GET /meta` | `backends_routes.py:602` | none | model + engine metadata: `checkpoint, n_params, layers, heads, ffn, vocab, seq, hidden, has_memory, prompt_bytes, c_backend_available, c_exe, c_exe_path, c_engine_version, c_engine_label, c_engine_perf_ms_per_byte, c_model, c_model_dir, c_model_path, c_model_precision, c_model_bin_version, c_model_training, c_model_activation, c_model_description, c_model_act_boost, c_model_qat_enabled, c_model_capabilities, pytorch_capabilities` | current model + backend configuration |
| `GET /neuron/<int:layer>/<int:nid>` | `backends_routes.py:566` | none | `{layer, neuron, stories, affinity, predecessors, successors, stats, label, pytorch_loaded, pytorch_last_error}` | neuron activation patterns + relationships |
| `GET /backends` | `backends_routes.py:475` | none | `{pytorch:{loaded, pending, model, step, last_error}, c:{loaded, pending, exe, model_bin, model_dir, blocked_reason, blocked_model, build, bins_available}}` | status of both inference backends |
| `POST /backends/pytorch` | `backends_routes.py:479` | body `action:"load"\|"unload"`, `model:str`, `step:int`, `threads:int` | same shape as `GET /backends`; `400` no models / non-vanilla, `500` load error | load/unload/swap the PyTorch brain |
| `POST /backends/c` | `backends_routes.py:542` | body `action:"load"\|"unload"`, `model:str` | same shape as `GET /backends`; `400` invalid action | load/unload the C engine model |

## interpretability / atlas

Read-only neuron + concept introspection over a trained model's dump artifacts.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /atlas/concept` | `atlas_routes.py:28` | `model`, `step`, `substring`, `top_k` | concept-to-neuron mapping; `400` bad model | neurons that fire for a concept substring |
| `GET /atlas/neuron/<int:layer>/<int:neuron>` | `atlas_routes.py:38` | `model`, `step`, `top_k` | neuron-to-concept associations; `400` bad model | concepts a single neuron tracks |
| `GET /atlas/lifetime/<int:layer>/<int:neuron>` | `atlas_routes.py:47` | `model` | neuron lifetime/evolution dict; `400` bad model | how a neuron changed across training |
| `GET /atlas/circuit` | `atlas_routes.py:54` | `layer`, `top_k` | circuit-graph dict | circuit graph for a layer |
| `GET /atlas/concepts_inverted` | `atlas_routes.py:61` | `model`, `step` | inverted concept index; `400` bad model, `500` error | inverted concept-to-neuron index |

## settings

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /settings` | `settings_routes.py:33` | none | full settings object (see keys below) | read all settings |
| `POST /settings` | `settings_routes.py:33` | body = patch of any settings keys | merged settings object; `400 {error}` on invalid value | update settings; flipping `pytorch_load_mode` to `"always"` eager-loads the brain |
| `GET /settings/notices` | `settings_routes.py:72` | none | `{notices:[...]}` | pending build-notice messages |
| `POST /ai/ask` | `settings_routes.py:76` | body `kind:str`, `payload:dict` | varies by `kind` | in-app AI assistant dispatch |

Settings keys (`veritate_mri/runtime/settings.py:51`): `pytorch_load_mode` (`on_demand`\|`always`), `pytorch_idle_unload_secs`, `hud_enabled`, `hud_position`, `hud_detailed`, `temperature_unit` (`C`\|`F`\|`K`), `heartbeat_enabled`, `heartbeat_send_errors`, `consent_modal_seen`, `analytics_advanced_enabled`, `diagnostics_logs_enabled`, `device_preference`, `update_channel` (`stable`\|`experimental`), `auto_reload_on_update`, `extensions`, `ai_enabled`, `ai_endpoint_user`, `ai_api_key_user`, `last_acknowledged_build`, `device_name`, `corpus_catalog_url`, `corpus_user_sources`, `teacher_provider`, `teacher_model`, `teacher_base_url`, `teacher_api_key`, `teacher_configs`, `teacher_max_concurrency`, `teacher_max_tokens`, `teacher_temperature`, `mesh_role` (`off`\|`node`\|`hub`\|`both`), `mesh_hub_address`, `mesh_auth_token`, `tutorial_enabled`, `tutorial_completed`. The `extensions` flag gates extension UI surfaces (the per-extension nav links and the Marketplace entry); the routes themselves register regardless.

## teacher

Teacher endpoints configure and drive a remote frontier LLM used for synthetic-data generation and chat fallback. Providers: carpathian, openai, anthropic, gemini, xai, deepseek, mistral, groq, openrouter, ollama, lm_studio, llama_cpp.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET, POST /teacher` | `teacher_routes.py:231` | POST body `teacher_provider`, `teacher_api_key`, `teacher_model`, `teacher_base_url`, `teacher_configs` | `{providers, configured:bool, provider, model, base_url, has_api_key:bool, configs, max_concurrency, max_tokens, temperature}`; `400` on invalid value | read / update teacher config |
| `POST /teacher/test` | `teacher_routes.py:259` | body `provider`, `model`, `base_url`, `api_key` | `{ok, model, latency_ms}` or `{ok:false, error}` | test a provider connection |
| `POST /teacher/models` | `teacher_routes.py:282` | body `provider`, `base_url`, `api_key` | `{models:[str]}`; `400` no provider | list a provider's available models |
| `POST /teacher/synth/start` | `teacher_routes.py:296` | body `prompts:[dict]` (required), `format`, `seed_ids`, `job_id`, `output_dir`, teacher overrides | `{job_id, output_dir}`; `400` bad prompts, `409` job running | start a synthetic-data generation job |
| `POST /teacher/synth/stop` | `teacher_routes.py:420` | body `job_id` | `{job_id, stopping:bool}`; `404` unknown job | request a graceful stop |
| `GET /teacher/synth/jobs` | `teacher_routes.py:342` | none | `{jobs:[{job_id, completed, categories, seeds, running}]}` | list synth jobs |
| `GET /teacher/synth/status` | `teacher_routes.py:447` | `job_id` | `{job_id, running, completed, failed, skipped_dup, last_error, error_summary, aborted, output_path}`; `404` | synth job status + stats |
| `GET /teacher/synth/samples` | `teacher_routes.py:432` | `job_id`, `limit` (default 20, max 100) | `{job_id, samples:[{id, response}]}`; `404` | recent sample outputs |
| `POST /teacher/synth/delete` | `teacher_routes.py:361` | body `job_id` | `{job_id, deleted:bool}`; `400` missing, `409` running, `404` unknown | delete a finished job |
| `POST /teacher/synth/build_corpus` | `teacher_routes.py:381` | body `job_id`, `stem` | `{stem, train_bin, val_bin, n_records, n_train, n_val}`; `400` bad stem, `404` no samples | build train/val `.bin` corpora from a job |
| `GET /teacher/seeds` | `teacher_routes.py:406` | none | `{version, seeds:[...], total_count}` | list seed-prompt catalogs |
| `GET /teacher/seeds/<seed_id>` | `teacher_routes.py:413` | none | `{id, count, prompts:[dict]}`; `404` unknown | prompts for one seed catalog |

## hybrid / chat

The public chat surface (`/` and `/chat` serve `hybrid.html`). `/hybrid/chat` answers with the local byte-model or a teacher, with optional RAG over an uploaded knowledge base and conversation memory.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /hybrid/health` | `hybrid_routes.py:459` | none | `{ok, has_corpus:bool, n_files:int}` | chat health + KB status |
| `GET /hybrid/models` | `hybrid_routes.py:463` | none | `{models:[{id, label, group, provider?, model?}]}` | chat model list (local + remote) |
| `POST /hybrid/kb/upload` | `hybrid_routes.py:467` | `multipart/form-data` file field `file` | `{ok, filename, n_files, n_chunks}`; `400` no file, `413` >64MB, `500` save error | upload a text file to the RAG knowledge base |
| `POST /hybrid/chat` | `hybrid_routes.py:482` | body `message:str` (required), `model`, `backend` (`pytorch`\|`c`), `use_rag:bool`, `use_logs:bool`, `kb_scope` (`all`\|`platform`\|`user`), `k:int`, `history:[{role, content}]`, `summary:str` | `{ok, answer, model, backend, confident:bool, sources:[{text, score}], memory:{summary, turns}, context:{turns, chars, char_limit, pct}}`; `400` empty message, `503` model/provider unavailable, `500` error | conversational chat with RAG + teacher fallback + memory |

## rag

Build and train a RAG-grounded SFT model over the knowledge base.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `POST /rag/build_corpus` | `rag_routes.py:118` | body `n_facts:int` (>0) | `{ok, phase:"build_corpus", stem:"rag_ui"}`; `400` bad n_facts, `409` running | build a RAG corpus from the KB |
| `POST /rag/train` | `rag_routes.py:131` | body `source:str` (required), `name:str`, `steps:int=1500`, `n_facts:int=200` | `{ok, phase, name, auto_built:bool}`; `400` no source, `409` running | train a RAG SFT model (auto-builds corpus if missing) |
| `POST /rag/stop` | `rag_routes.py:154` | none | `{ok, running:bool}`; `500` | stop the running RAG job |
| `GET /rag/status` | `rag_routes.py:169` | none | `{ok, status, phase, started_at, finished_at, exit_code, running:bool, log:str}`; `500` | RAG job status + log tail |

## wiki

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /wiki` | `wiki_routes.py:22` | none | `{categories:[...]}` | list wiki categories |
| `GET /wiki/<category>` | `wiki_routes.py:26` | none | `{category, entries:[...]}`; `404` unknown | list entries in a category |
| `GET /wiki/<category>/<slug>` | `wiki_routes.py:33` | none | entry dict; `404` unknown | a single wiki entry |

## mesh

Distributed-training mesh control. Roles: off, node, hub, both. Hub/node routes register only when the role is set (`veritate_mri/app.py:162`).

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /mesh/status` | `mesh_routes.py:38` | none | `{role, hub_address, has_token:bool, node_registered:bool, last_heartbeat, current_job, hub_nodes}` | mesh status + config |
| `GET /mesh/token` | `mesh_routes.py:69` | none | `{has_token:bool, token}` (token unmasked only to localhost) | read the mesh auth token |
| `POST /mesh/token/regenerate` | `mesh_routes.py:78` | none | `{ok, token}`; `400` on invalid | regenerate the mesh auth token |
| `POST /mesh/test_connection` | `mesh_routes.py:87` | body `hub_address`, `auth_token` (both optional) | `{ok, status, error, response_ms}` | test connection to a mesh hub |
| `POST /mesh/role` | `mesh_routes.py:114` | body `role:str` | `{ok, role, restart_required:bool}`; `400` invalid role | set the mesh role |

## engine and pruning

C-engine build/config and model export (used to produce the `.bin` an extension or the C backend consumes).

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /engine/status` | `engine_routes.py:30` | none | `{status, error, c_subprocess_running, c_exe, ...}` | C-engine build + subprocess status |
| `POST /engine/build` | `engine_routes.py:39` | body `force:bool` | build-state dict | trigger a C-engine build |
| `GET /c-engines` | `engine_routes.py:44` | none | `{engines:[{version, label, perf_ms_per_byte, path, exists, is_current, mtime, size}]}` | list built C-engine binaries |
| `GET /c-models` | `engine_routes.py:64` | none | `{models:[{name, bin_path, is_current, mtime, size, precision, bin_version, training, activation, act_boost, qat_enabled, description}]}` | list C-model `.bin` files |
| `POST /c-config` | `engine_routes.py:93` | body `exe:str`, `model:str` (both paths, optional) | `{ok, c_exe_path, c_exe, c_model_path, c_model, c_model_dir, c_model_precision, c_model_bin_version, c_model_training, c_model_activation, c_model_act_boost, c_model_qat_enabled}`; `400` not found / no exe, `500` respawn failed | select + respawn the C engine and model |
| `GET /pruning/report` | `pruning_routes.py:30` | `model` (required), `step`, `samples=32` | `{ok, model, step, corpus, samples, n_params, n_params_after, size_mb_before, size_mb_after, dead_pct, per_layer:[{layer, alive, total, alive_frac, keep}], plan:{layer:keep_frac}}`; `400` bad model/corpus, `500` | analyze FFN neuron activity, recommend a pruning plan |
| `POST /pruning/generate_plugin` | `pruning_routes.py:114` | body `model`, `step`, `plan:dict`, `samples=16` | `{ok, plugin_id, plugin_dir}`; `400` invalid/non-vanilla, `500` | generate a width-pruning trainer from a plan |
| `POST /export/<name>` | `pruning_routes.py:169` | body `step:int` (optional, defaults latest) | `{ok, path, bytes, ...}`; `400` no checkpoints / export error, `404` no model, `500` | export a PyTorch checkpoint to a C `.bin` |

## system, logs, lifecycle

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /sys_metrics` | `sys_routes.py:41` | none | metrics snapshot dict | live CPU/mem/etc. snapshot |
| `GET /sys/mode` | `sys_routes.py:45` | none | `{minimal:bool}` | whether power-save mode is on |
| `POST /sys/mode/relaunch` | `sys_routes.py:50` | body `minimal:bool` | restart-result dict | relaunch with the minimal flag toggled |
| `GET /sys/specs` | `sys_routes.py:62` | none | specs dict or `{detected:false}` | saved hardware specs |
| `POST /sys/detect` | `sys_routes.py:66` | none | detect-result dict | detect + save hardware specs |
| `GET /heartbeat/status` | `sys_routes.py:70` | none | heartbeat-status dict | analytics heartbeat status |
| `POST /heartbeat/send` | `sys_routes.py:74` | none | `{ok, ...}` | send a heartbeat now |
| `GET /heartbeat/preview` | `sys_routes.py:79` | none | payload preview dict | preview the heartbeat payload |
| `GET /app/update_status` | `sys_routes.py:83` | none | update-status dict | app self-update status |
| `POST /app/update_check` | `sys_routes.py:87` | none | check-result dict | check for app updates |
| `POST /app/update_pull` | `sys_routes.py:91` | body `force:bool`, `ignore_training:bool`, `reload:bool` | pull-result dict (`+reload_error` on failed reload) | pull an app update, optionally restart |
| `GET /app/local_edits` | `sys_routes.py:104` | none | local-edits dict | files diverging from the pulled baseline |
| `POST /app/update_channel` | `sys_routes.py:109` | body `channel:str` | switch-result dict | switch update channel |
| `GET /versions` | `sys_routes.py:115` | none | `application/json` body of `versions.json`; `404` if missing | version ledger (build, engine, mri, format, plugins) |
| `GET /logs/snapshot` | `logs_routes.py:27` | `after:int=0`, `limit:int` | `{latest_seq:int, entries:[...]}` | log entries after a sequence number |
| `GET /logs/stream` | `logs_routes.py:34` | none | **SSE.** one JSON log entry per `data:` frame, 15s keepalive | live log stream |
| `POST /lifecycle/restart` | `lifecycle_routes.py:24` | none | restart-result dict | restart the server |
| `POST /lifecycle/kill` | `lifecycle_routes.py:28` | none | kill-result dict | kill the server process |
| `POST /lifecycle/soft_reload` | `lifecycle_routes.py:32` | none | soft-reload-result dict | soft-reload the server |

## market (reference extension)

The market page (`/market` serves `market.html`) is the canonical downloadable-extension reference. It is fully isolated: it reads only external-data CSVs and Veritate checkpoints (`veritate_mri/routes/market_routes.py:6`). All routes are wrapped in `safe_route`, so failures return `{ok:false, error}` with `500` unless a route sets a narrower status. The page is surfaced under the `extensions` settings flag, but the routes register unconditionally.

| method + path | def | params | response | purpose |
|---|---|---|---|---|
| `GET /market/veritate_models` | `market_routes.py:24` | none | `{ok, models:[...]}` | list market-capable Veritate models |
| `GET /market/veritate_hindcast` | `market_routes.py:31` | `model` (required), `source=crypto`, `symbol=BTCUSDT`, `base=1m`, `n=1500` (300-20000) | hindcast result `+ {ok, symbol, model, step}`; `400` bad model, `404` no data / too few bars | backtest a model over historical bars |
| `GET /market/veritate_benchmark` | `market_routes.py:54` | same as hindcast | benchmark result `+ {ok, symbol, model, step}`; `400`/`404` | scored benchmark over historical bars |
| `GET /market/veritate_data_report` | `market_routes.py:77` | `source=crypto` | report dict `+ {ok, source}` | data-availability report |
| `GET /market/veritate_live` | `market_routes.py:88` | `model` (required), `symbol=BTCUSDT`, `source=crypto` | prediction `+ {ok, symbol, model, last_close, last_t, expected_move_bps}`; `400` bad model, `404` no data, `500` predict fail, `502` live-feed error | next-bar prediction on live/recent data |
| `GET /market/instruments` | `market_routes.py:122` | `source=crypto` | `{ok, source, instruments:[...]}` | list instruments for a source |
| `GET /market/extensions/catalog` | `market_routes.py:130` | none | catalog dict | downloadable-extension catalog |
| `POST /market/extensions/download` | `market_routes.py:137` | body `source:str` | download-result dict | install an extension by source id |
| `POST /market/extensions/delete` | `market_routes.py:144` | body `source:str` | delete-result dict | remove an installed extension |

## static and auth

| method + path | def | response | purpose |
|---|---|---|---|
| `GET /`, `GET /chat` | `app.py:70` | `hybrid.html` | public chat page |
| `GET /app` | `app.py:76` | `index.html` | the dashboard |
| `GET /market` | `app.py:81` | `market.html` | market extension page |
| `GET /static/<path>` | Flask static (`app.py:53`) | static asset | dashboard + page assets, served from `veritate_mri/web/` at `/static` |
| `GET, POST /login` | `auth_routes.py:57` | GET `login.html`; POST form field `password`, redirects to `/app` on match else `/login?e=1` | password login (only meaningful when auth is enabled) |
| `GET /logout` | `auth_routes.py:67` | redirect `/` | clear the session |
