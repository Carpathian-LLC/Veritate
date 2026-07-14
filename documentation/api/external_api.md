# external api reference

The box's programmatic API: the surface an outside client (another server, an app, an OpenAI-compatible library) uses to reach the models on this box. This is the surface the dashboard **Settings → API access** key gates.

Two shapes:

- **OpenAI-compatible** (`/v1/models`, `/v1/chat/completions`, `/v1/chat/mri`) — point any OpenAI client at the base URL.
- **Native SSE** (`/generate`, `/agent/stream`) — byte-level generation and agent traces with full telemetry.

For the complete platform contract that on-box extensions code against (models, runs, training, settings, and everything else), see [internal_api.md](internal_api.md).

## base url

- Default `http://<box-ip>:8001` (`--port` overrides). On the same host, `http://127.0.0.1:8001`.
- Requests and responses are JSON unless noted. SSE endpoints stream `text/event-stream`.
- Errors are always JSON, never an HTML page: `{"ok": false, "error": "<message>"}` (HTTP errors also carry `"status"`). Treat any non-2xx as failure and read `error`.

## auth

Off by default. On a trusted LAN the four endpoints stay open. Set a key when you expose the box beyond the LAN.

- **Enable:** dashboard **Settings → API access → generate key**, or `POST /settings/api-key {"action":"rotate"}`. Clear it with `{"action":"clear"}`.
- **Use:** when a key is set, every request to `/v1/*`, `/generate`, and `/agent/stream` must carry `Authorization: Bearer <key>`. Missing or wrong key returns `401 {"ok": false, "error": "invalid or missing api key"}`.
- **Scope:** only these four endpoints are gated. Dashboard pages, `/chat`, `/hybrid/*`, and `/static` are never gated by this key.

```bash
curl http://<box-ip>:8001/v1/models \
  -H "Authorization: Bearer <key>"
```

## endpoints

### GET /v1/models

List the models this box can serve.

```json
{"object": "list",
 "data": [{"id": "<name>", "object": "model", "created": 0, "owned_by": "veritate",
           "n_params": 0, "hidden": 0, "layers": 0, "capabilities": {}, "is_current": true,
           "engine": "c"}]}
```

`engine` is the backend the box will run that model on for chat/generation: `"c"` (the compiled Veritate C engine, used whenever the model has a `.bin` export and the engine binary is built) or `"pytorch"` (the fallback for non-exportable trunks). The box sets this; a client does not choose it. The C engine is faster and runs off the GPU, so a `"pytorch"` model may respond more slowly, especially while the box is also training.

### POST /v1/chat/completions

OpenAI-compatible chat completions.

- Body: `model` (a model `id` from `/v1/models`, or `cloud` for the configured teacher), `messages` (list of `{role, content}`, must end with a user turn), `stream` (bool), optional `temperature`, `max_tokens`, `top_k`, `mri` (bool, opt-in per-byte MRI telemetry, see [receiving MRI telemetry](#receiving-mri-telemetry)).
- Non-stream response: standard OpenAI `chat.completion` (`choices[].message.content`, `finish_reason:"stop"`, `usage`).
- Stream response: `text/event-stream` of `chat.completion.chunk` frames (a role frame, content-delta frames, a `finish_reason:"stop"` frame), then `data: [DONE]`. Local trained models stream true per-token as generated; the cloud teacher chunks its buffered answer.
- Errors: `400` bad body, `503` model or provider unavailable, `500` stream error.

```bash
curl http://<box-ip>:8001/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "hello"}]}'
```

### POST /v1/chat/mri

OpenAI-shaped sibling of `/v1/chat/completions` that also streams per-byte MRI (model-internals) telemetry, so a client gets reply text and interpretability frames in one stream. Local trained byte models only.

- Body: identical to `/v1/chat/completions` (`model`, `messages`, `stream`, optional `temperature`, `max_tokens`, `top_k`). `stream` defaults to `true` here.
- Stream response: `text/event-stream` of `chat.completion.chunk` frames. A role frame, then one chunk per MRI frame carrying that frame under a top-level `mri` key plus the assistant text delta for its byte in `choices[0].delta.content`, then a `finish_reason:"stop"` frame, then `data: [DONE]`. A plain OpenAI client reads the content deltas and ignores `mri`.
- Non-stream response: the standard OpenAI `chat.completion` plus the full per-byte MRI frame list under a top-level `mri` key.
- Errors: `400` bad body or a non-local (cloud/teacher) model, which has no MRI; `503` model unavailable.

```bash
curl http://<box-ip>:8001/v1/chat/mri \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "hello"}]}'
```

### GET /generate  (SSE)

Byte-by-byte generation with per-byte interpretability telemetry.

- Params: `prompt` (required), `temperature` (0.7), `top_k` (40), `max_new` (200, capped 4096), `backend` (`c` default, or `pytorch`). Full param list (ablation, repetition control, decode addons, constrained decoding) is in [internal_api.md](internal_api.md).
- Stream: a `meta` frame (model + engine shape), then `kind:"token"` frames (one per byte, with entropy/surprise/confidence/attention fields), then `kind:"stop"` and `done`. A bad numeric param yields a `kind:"error"` SSE frame + `done`, not a 500, so the `EventSource` stays parseable.
- `rag=` is loopback-only: a non-loopback client passing it gets `403`.

### GET /agent/stream  (SSE)

Stream a full agent trace (tool calls plus reasoning).

- Params: `prompt` (required), `max_turns` (6, max 16), `best_of_n` (1, max 8), `temperature` (0.7), `top_k` (40), `seed` (0), `tools` (csv).
- Stream: an `agent_meta` frame, then per-turn agent frames, then `done`.
- `corpus=` and `fs_root=` are loopback-only: a non-loopback client passing either gets `403`.

## receiving MRI telemetry

`/v1/chat/completions` serves text only by default (fast, non-traced). Set `"mri": true` in the request body to also receive the full per-byte **MRI frames** — the same interpretability objects the dashboard MRI view and `/generate` emit (entropy, surprise, per-layer FFN activations, attention, logit-lens, candidates, confidence). Absent or `false` is the default and is byte-for-byte the current behavior.

- **Local trained models only.** A remote model (`cloud`, a teacher) exposes no per-byte telemetry, so the flag is ignored: the response is identical to leaving it off.
- **Frame schema is not redefined.** The frames are the exact parsed objects `/generate` yields (`kind:"meta"`, then one `kind:"token"` per generated byte, then `kind:"stop"`); fields are passed through unfiltered. See the `/generate` `kind:"token"` field list in [internal_api.md](internal_api.md).
- **Output bytes are unaffected.** The flag selects telemetry, never sampling parameters. Same generation path as `/generate`'s traced/full modes.

### streaming (`stream:true`)

The OpenAI-format text chunks stay standard `chat.completion.chunk` frames. Each MRI frame rides as its **own** SSE `data:` line: a valid `chat.completion.chunk` with an empty `delta` and the frame under a top-level `mri` key. An off-the-shelf OpenAI client that reads only `choices[].delta.content` ignores the empty-delta chunk and the unknown `mri` key, so it keeps working; an MRI-aware client reads `chunk["mri"]`.

```text
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"chat80m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"chat80m","choices":[{"index":0,"delta":{},"finish_reason":null},"mri":{"kind":"meta","checkpoint":"chat80m","layers":12,"heads":12,"ffn":3072,"vocab":256,"seq":1024,"hidden":768,"prompt_bytes":[…]}}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"chat80m","choices":[{"index":0,"delta":{},"finish_reason":null},"mri":{"kind":"token","byte":72,"argmax_byte":72,"entropy_bits":1.2,"surprise_bits":0.4,"confidence":0.91,"ffn_full":[…],"ffn_top":[…],"attn":[…],"lens":[…],"cand":[…],"res":[…],"contrib":[…],"backend":"pytorch"}}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"chat80m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"chat80m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

(`mri` and `choices` are shown on one object per frame; the JSON is a single line per `data:`.)

### non-streaming (`stream:false`)

The response is the standard `chat.completion` object with one added top-level `mri` key holding the frame list (`meta`, one `token` per byte, `stop`):

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,"model":"chat80m",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],
 "usage":{…},
 "mri":[{"kind":"meta",…},{"kind":"token","byte":72,…},{"kind":"stop","reason":"<|end|>"}]}
```

### size caveat and the native alternative

MRI frames are **large**: the C engine's per-byte frame carries every layer's downsampled FFN buckets, per-head attention, and per-layer logit-lens tables, on the order of hundreds of KB per generated byte. A short reply is multiple MB. Request telemetry only when consuming it. For a dedicated interpretability stream, [`/generate`](#get-generate-sse) remains the native full-telemetry SSE endpoint (byte-level, with ablation, RAG prefix, decode addons, and constrained decoding); the `mri` flag exists so an OpenAI-compatible client can obtain the same frames without leaving the chat API.

The chat-page-facing `/hybrid/chat` and its `/hybrid/chat/stream` twin take the same `"mri": true` flag with the same semantics (streamed frames ride as native `kind:"token"`/`kind:"meta"`/`kind:"stop"` SSE frames alongside the `kind:"delta"` text frames; the buffered form attaches the frame list under a top-level `mri` key). See [internal_api.md](internal_api.md).
