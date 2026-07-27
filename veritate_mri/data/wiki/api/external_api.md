---
title: external API and the key gate
date: 2026-07-27
tags: [api, reference]
summary: The programmatic surface an outside client uses to reach this box's models, and the optional API key that protects it.
---

# external API and the key gate

The surface an outside client (another server, an app, an OpenAI-compatible library) uses to reach the models on this box. This is what the dashboard **Settings, API access** key protects.

Two shapes:

- **OpenAI-compatible**: `/v1/models`, `/v1/chat/completions`, `/v1/chat/mri`. Point any OpenAI client at the base URL.
- **Native SSE**: `/generate` and `/agent/stream`. Byte-level generation and agent traces with full telemetry.
- **Work ahead of a request**: `/prefill` reads a prompt into the engine while it is still being built, so the request that follows skips the prefill. `/prefetch` goes further and generates the reply itself. Reading ahead guesses nothing and wastes nothing; generating ahead guesses the prompt is finished and discards the reply when it is wrong.

The complete platform contract, including everything an on-box extension calls, is in the platform API reference entry.

## base url

- Default `http://<box-ip>:8001`. `--port` overrides the port. On the same host, `http://127.0.0.1:8001`.
- Requests and responses are JSON except the SSE endpoints, which stream `text/event-stream`.
- Errors are always JSON, never an HTML page: `{"ok": false, "error": "<message>"}`, with a `status` field on HTTP errors. Treat any non-2xx as failure and read `error`.

## the key gate

Off until a key is set. On a trusted LAN the endpoints stay open. Set a key before exposing the box beyond the LAN.

- **Enable:** dashboard **Settings, API access, generate key**, or `POST /settings/api-key {"action":"rotate"}`. Clear it with `{"action":"clear"}`.
- **Use:** when a key is set, every protected request carries `Authorization: Bearer <key>`. A missing or wrong key returns `401 {"ok": false, "error": "invalid or missing api key"}`.
- **What is protected:** the exact paths `/generate`, `/agent/stream`, `/prefetch`, and `/prefill`, plus **every path beginning with `/v1/`**. The gate in `veritate_mri/routes/api_auth_routes.py` is a prefix test, not a list of endpoints, so any `/v1/` route added later is covered the moment it exists. Today that is `/v1/models`, `/v1/chat/completions`, and `/v1/chat/mri`.
- **What is not protected:** the dashboard, `/chat`, the `/hybrid/*` routes, and `/static`. The bare path `/v1` with no trailing slash is not protected either.

The dashboard password gate is a separate mechanism and runs first. When `VERITATE_DASHBOARD_PASSWORD` is set, `/v1/*` is not on its public list, so a programmatic client also needs a dashboard session for those paths. A bearer key alone is enough only when the dashboard password is unset.

```bash
curl http://<box-ip>:8001/v1/models \
  -H "Authorization: Bearer <key>"
```

## GET /v1/models

List the models this box can serve.

```json
{"object": "list",
 "data": [{"id": "<name>", "object": "model", "created": 0, "owned_by": "veritate",
           "n_params": 0, "hidden": 0, "layers": 0, "capabilities": {}, "is_current": true,
           "engine": "c"}]}
```

`engine` is the backend the box will run that model on: `c` for the compiled Veritate C engine, used whenever the model has a `.bin` export and the engine binary is built, or `pytorch` for a model whose trunk is not exportable. The box decides this; a client does not choose it. The C engine is faster and runs off the GPU, so a `pytorch` model may respond more slowly, especially while the box is training.

## POST /v1/chat/completions

OpenAI-compatible chat completions.

- Body: `model` (an id from `/v1/models`, or `cloud` for the configured teacher), `messages` (a list of `{role, content}` ending on a user turn), `stream`, and optional `temperature`, `max_tokens`, `top_k`, `rep_window`, `rep_penalty`, `no_repeat_ngram`, `mri`.
- Per-request generation overrides are bounded on the way in: `temperature` to 0 through 2, `rep_penalty` to 0 through 2, `rep_window` to 0 through 4096, `no_repeat_ngram` to 0 through 64. `top_k` and `max_tokens` are clamped later by the engine's own limits. Omit a key to keep the box default.
- Non-stream response: a standard OpenAI `chat.completion` with `choices[].message.content`, `finish_reason:"stop"`, and `usage`.
- Stream response: `text/event-stream` of `chat.completion.chunk` frames (a role frame, content-delta frames, a `finish_reason:"stop"` frame), then `data: [DONE]`. Local trained models stream a true delta per generated token; the cloud teacher chunks its buffered answer.
- Errors: `400` bad body, `503` model or provider unavailable, `500` stream error.

```bash
curl http://<box-ip>:8001/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "hello"}]}'
```

## POST /v1/chat/mri

The same request body and model routing as `/v1/chat/completions`, with per-byte MRI telemetry always on. MRI frames are the model's own internals for each generated byte, so a client gets reply text and interpretability data in one stream. Local trained byte models only.

- `stream` defaults to true here.
- Stream response: a role frame, then one `chat.completion.chunk` per MRI frame carrying that frame under a top-level `mri` key and the byte's text in `choices[0].delta.content`, then a `finish_reason:"stop"` frame, then `data: [DONE]`. A plain OpenAI client reads the content deltas and ignores the extra key.
- Non-stream response: the standard `chat.completion` plus the full frame list under a top-level `mri` key.
- Errors: `400` bad body or a remote model, which has no telemetry; `503` model unavailable.

## GET /generate

Byte-by-byte generation with per-byte telemetry. SSE.

- Params: `prompt` (required), `temperature` (0.7), `top_k` (40), `max_new` (200, capped at 4096), `backend` (`c` by default, or `pytorch`), `prefetch_id` (claims a `/prefetch` draft). The full parameter list, covering ablation, repetition control, decode addons, and constrained decoding, is in the platform API reference.
- Stream: a `meta` frame with the model and engine shape, then one `kind:"token"` frame per byte carrying entropy, surprise, confidence, and attention fields, then `kind:"stop"` and `done`.
- A bad numeric parameter yields a `kind:"error"` frame followed by `done` rather than a 500, so an `EventSource` client stays parseable.
- `rag=` is loopback only. A remote client passing it gets `403`.

## POST /prefill

Read a prompt into the engine before the request that carries it arrives. Answering costs reading the prompt and then writing the reply; reading is the part that must finish before a single reply byte exists. Sending it early moves that cost off the request.

- Body: either `messages` or `prompt`, plus `temperature`, `top_k`, and (with `messages`) `max_tokens`. Omit both to stand down.
- **A chat client sends `messages`.** The same OpenAI-shaped array `/v1/chat/completions` takes, ending on the user turn being typed, and the box renders the chat prefix with the template it will submit through, fitting the conversation to the context exactly as submit fits it. Send `max_tokens` here whenever you send it on submit: it sizes the reply room that fit is measured against, so a mismatch keeps a different number of turns. A client that renders ChatML itself has to reproduce the box's framing exactly, and a client that gets it wrong sees no error, just no speedup.
- Response: `{"reading": true, "read_bytes": 214, "stats": {"reads": 12, "bytes": 3324}}`. A `reason` field appears when the box declined (not allowed for this caller, C engine not loaded).
- **A `prompt` body must be a strict prefix of what you will submit.** For a chat-framed prompt that means the conversation plus the user header plus the text so far, and NOT the closing `<|im_end|>` scaffold: the scaffold sits after the text, so including it moves it on every update and matches nothing. The box restores the longest matching prefix, so a prompt that diverges simply restores less.
- Nothing is buffered and nothing is claimed. There is no id and no matching step: the next `/generate` carrying the prompt benefits automatically. A caller that never follows through has only paid for reading.
- The box reads one prompt at a time; a new one supersedes it. A real request always wins the engine.
- Controlled by **Settings, API access, API: work ahead of a request**. Allowed by default for API callers.

Measured on a 200M model: a 552-byte prompt takes 857 ms to the first reply byte cold, and 136 ms when it was read ahead while being typed, for about 1 s of engine time spread over 106 s of typing.

The intended client loop: debounce the input, and on each pause POST the conversation with the draft as its last user turn. Then submit normally. There is nothing else to do.

## POST /prefetch

Speculate a reply for a prompt a user is still typing. Off by default. Enable for API callers under **Settings, API access, API: work ahead of a request**; the dashboard's own switch is **generate ahead**, next to its generate button.

Prefer `/prefill` unless the client is certain it can tell when a prompt is finished. A wrong guess here discards a whole generation; `/prefill` cannot be wrong.

- Body: `prompt`, the exact wire prompt the client will send on submit, plus every decode knob submit will use: `temperature`, `top_k`, `ablate_layer`, `ablate_neuron`, `addons`, `rep_window`, `rep_penalty`, `no_repeat_ngram`. Omit `prompt` to stand down.
- Response: `{"speculating": true, "buffered": 0, "budget": 1024, "draft_id": 7, "state": "running", "stats": {...}}`. `state` is why the draft stopped: `running`, `done` (the reply finished), `budget` (hit the byte cap), `busy` (the engine never came free), or `error`. A `reason` field appears when the box declined (not allowed for this caller, C engine not loaded).
- The box keeps one draft. A new prompt supersedes the previous one; a stand-down drops it.
- On submit, pass `prefetch_id=<draft_id>` to `GET /generate` to claim the buffer. Passing the id asserts that nothing the client would send has changed since the draft; the box also requires the prompt to match. A miss generates normally, so a wrong or stale id costs nothing. The buffer is claimed by `/generate` only; the chat endpoints ignore it.
- `GET /prefetch` reports the live draft and the reply so far without consuming it.

The intended client loop: debounce the input, and on each pause POST the current prompt and keep the `draft_id` alongside the exact body you sent. On submit, echo the id only when the body you would send now is identical. A buffer is only ever flushed once.

## GET /agent/stream

A full agent trace: tool calls plus reasoning. SSE.

- Params: `prompt` (required), `max_turns` (6, max 16), `best_of_n` (1, max 8), `temperature` (0.7), `top_k` (40), `seed` (0), `tools` (comma separated).
- Stream: an `agent_meta` frame, then per-turn frames, then `done`.
- `corpus=` and `fs_root=` are loopback only. A remote client passing either gets `403`.

## MRI telemetry from the chat endpoints

`/v1/chat/completions` serves text only by default. Setting `"mri": true` in the body adds the same per-byte frames `/generate` emits: entropy, surprise, per-layer activations, attention, logit lens, candidates, and confidence.

- **Local trained models only.** A remote model exposes no per-byte telemetry, so the flag has no effect there.
- **The frame schema is not redefined.** Frames are the exact objects `/generate` yields: `kind:"meta"`, one `kind:"token"` per generated byte, then `kind:"stop"`. Fields pass through unfiltered.
- **Output bytes are unaffected.** The flag selects telemetry, never sampling.

When streaming, each MRI frame rides as its own SSE line: a valid `chat.completion.chunk` with an empty `delta` and the frame under a top-level `mri` key. An OpenAI client that reads only `choices[].delta.content` ignores both the empty delta and the unknown key.

```text
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":null}],"mri":{"kind":"token","byte":72,"entropy_bits":1.2,"surprise_bits":0.4,"confidence":0.91}}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: [DONE]
```

When not streaming, the response is the standard `chat.completion` object with one added top-level `mri` key holding the frame list.

## size caveat

MRI frames are large. A per-byte frame from the C engine carries every layer's downsampled activation buckets, per-head attention, and per-layer logit-lens tables, on the order of hundreds of kilobytes per generated byte. A short reply is several megabytes. Request telemetry only when consuming it.

The `mri_compact_frames` setting makes the engine summarize each byte itself, shrinking the wire payload substantially while the dashboard renders the same view.

For a dedicated interpretability stream, `/generate` is the native full-telemetry endpoint with ablation, retrieval prefix, decode addons, and constrained decoding. The `mri` flag exists so an OpenAI-compatible client can obtain the same frames without leaving the chat API.

The chat-page routes `/hybrid/chat` and `/hybrid/chat/stream` take the same `"mri": true` flag with the same meaning.
