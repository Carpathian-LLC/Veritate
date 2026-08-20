# chat extraction handoff

The chat feature was removed from Veritate 2026-08-20 to become a standalone project. Veritate is training-specific; the Generation tab is the only in-platform way to talk to a model. This file is the complete inventory of what chat was, what was deleted (all recoverable from git history at this commit's parent), and the API contract a standalone chat client needs to stand itself up against a running Veritate server.

## what the feature was

A ChatGPT-style conversation UI, reachable two ways: a standalone page at `/chat` and a Chat tab inside the dashboard (an inlined port of the same page). User-facing behavior:

- Model picker: any local trained model (PyTorch or Veritate C engine), any configured teacher provider model (`teacher:<provider>:<model>`), the public Carpathian cloud model (`cloud`), or `veritate_docs` (cloud model grounded on the shipped platform KB).
- Streaming replies with a per-model context gauge (ring showing % of the model's window used).
- Conversation memory held client-side and compacted server-side: local byte models slide (drop oldest whole turns, no summary), remote models fold old turns into a running summary keeping the last 6 verbatim.
- BM25 grounding: "use my uploaded files" (user KB uploads), "read my recent training logs" (dashboard log ring tail), scoped platform/user/all.
- Slash commands, "new chat", transcript persistence in localStorage.

## deleted components

Frontend (files deleted):

- `veritate_mri/web/chat_tab.js` (358 lines) — the Chat tab logic, IIFE, ids prefixed `ch-`, wired by `chatTabInit()` on first tab activation. localStorage keys: `veritate_chat_settings_v1` (model/engine/RAG toggles), `veritate_chat_convo_v1` (`{summary, turns}` conversation memory).
- `veritate_mri/web/chat_tab.css` (97 lines) — styles scoped under `#chatMount`.
- `veritate_mri/web/hybrid.html` (516 lines) — the standalone `/chat` page; same features, self-contained.

Frontend (edits):

- `veritate_mri/web/index.html` — Chat tab button, the `CHAT TAB` body block (`#chatMount`, former lines 93–177), and the chat_tab css/js includes.
- `veritate_mri/web/index.js` — "chat" removed from the `valid` tab array; `chatTabInit` activation branch.
- `veritate_mri/web/index.css` — chat-tab viewport-lock rules (`body:has(.tab-body[data-tab="chat"].active)`).

Server (edits, no file deleted):

- `veritate_mri/app.py` — the `/chat` page route (served hybrid.html).
- `veritate_mri/routes/hybrid_routes.py` — chat endpoints and chat-only helpers: `POST /hybrid/chat` (buffered turn: answer + compacted memory + context meter + sources), `POST /hybrid/chat/stream` (SSE `delta` frames then one `done` frame), `GET /hybrid/health`, `GET /hybrid/models` (picker enumeration incl. live teacher-provider model listing), `POST /hybrid/kb/upload` (user KB file upload, 64 MB cap); helpers `_chat_prepare`/`_chat_result`/`_ChatCtx`, memory compaction (`_compact`, `_fit_tail`, `_history_in`, summary constants), context meter, `_system_text`/`_fit_local_system` (persona-free local system budgeting), `remote_models`/`_provider_model_names`, `training_log_lines`, KB upload/save, `VERITATE_DOCS_ID`.
- `veritate_mri/routes/auth_routes.py` — `/chat` and `/hybrid` dropped from `PUBLIC_PREFIXES`.

Tests: `tests/mri/test_chat_compaction.py` deleted; `/hybrid/chat(/stream)` route tests removed from `tests/mri/test_openai_chat.py` and `tests/mri/test_mri_optin.py`.

## what did NOT move (stays in Veritate)

The entire backend. A standalone chat project is a client of a running Veritate server; it re-implements the UI and its conversation memory, not generation.

- `/generate` (`routes/backends_routes.py`) with all modes including `fast=stream` + `state_id`, and its experience-log wiring (`inference/experience.py` — every completed exchange appends to `data/experience/YYYYMMDD.jsonl`).
- The PyTorch backend (`inference/backends/pytorch.py`): `stream_fast`, `_stream_fast_streaming`, state persistence under `data/stream_states/`.
- OpenAI-compatible serving: `POST /v1/chat/completions` and `POST /v1/chat/mri` (`routes/hybrid_routes.py`), plus the shared ChatML framing (`render_local_open`, `_render_local`, `fit_chat_history`), model routing (`_resolve_route`: local/teacher/cloud), and the BM25 KB retrieval stack (`retrieve`, consumed by the hallucination detector).
- The Generation tab (chat/agent/autocomplete modes) — the in-platform conversational surface. Its localStorage key `veritate_chat_state_id_v1` (per-conversation stream-state id, rotated by "clear chat") stays.
- All chat-formatted corpora and builders (mixed_chat, veritate_chat, fact_sft, recall_curr, build_experience_corpus, build_fact_sft, build_recall_corpus) — training data, not UI.

## API contract for the standalone client

All against a running Veritate server (default `http://<host>:8001`). If an API key is set in Settings, `/generate` and `/v1/*` need `Authorization: Bearer <key>`.

### prompt format (ChatML, byte-literal)

Byte-level models learn the markers as literal bytes. One turn:

```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

Prior turns are prepended as `<|im_start|>{role}\n{content}<|im_end|>` joined by newlines. Optional grounding rides as a `context: {facts}\n` block before the final user turn. Replies hard-stop on `<|im_end|>` (matched without the closing bracket — byte models reproduce markers approximately); stop markers: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>`, `\ncontext:`. Fit history by dropping oldest whole turns, never by byte-slicing (a sliced turn makes the model answer the wrong question).

### option A: `/v1/chat/completions` (recommended)

OpenAI-compatible; the server does the ChatML rendering and history fitting. `model` = local model name, `teacher:<provider>:<model>`, or `cloud`. Supports `stream:true` (true per-token SSE for local models), `temperature`/`max_tokens`/`top_k` plus `rep_window`/`rep_penalty`/`no_repeat_ngram` extensions, and `mri:true` for per-byte telemetry frames. `/v1/chat/mri` is the telemetry-first sibling.

### option B: `/generate` (raw, what the Generation tab uses)

Client renders the ChatML prompt itself and opens `GET /generate?...` as an SSE stream (the Generation tab uses `EventSource`). Relevant query parameters: `prompt`, `backend` (`c`|`pytorch`), `temperature`, `top_k`, `max_new`, repetition params (`rep_window`/`rep_penalty`/`no_repeat_ngram` — send them for chat-framed prompts), and:

- **`fast=stream`** (PyTorch backend, recurrent-mixer models with seq a multiple of 256): unbounded-context decoding over carried recurrent state; the prompt is never truncated.
- **`state_id=<[A-Za-z0-9._-]{1,64}>`** — per-conversation persisted state. Loopback-only (server rejects non-local callers). After every call the carried states + pending buffer save to `data/stream_states/<id>.pt`; the next call sends ONLY the new bytes (the new user turn + assistant open marker) and continues byte-exactly. `state_reset=1` starts the id over. A state is bound to the exact checkpoint that wrote it; a checkpoint mismatch errors (client should rotate to a fresh id on a "stream state" error, as the Generation tab does). One id per conversation; rotate on "new chat".
- Every completed exchange is recorded server-side in the experience log regardless of caller.

### what the standalone project must build itself

Conversation memory (client-side turns + compaction policy), context gauge, model picker (enumerate local models via `/models` / `/pytorch-models`; teacher enumeration was `/hybrid/models`, now deleted — reimplement or list manually), KB upload + grounding if wanted (the server keeps BM25 retrieval internally but no longer exposes upload or chat-side retrieval endpoints), and transcript persistence. The deleted `chat_tab.js` / `hybrid.html` / hybrid_routes chat helpers at this commit's parent are the reference implementation for all of it.
