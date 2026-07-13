# external api reference

The box's programmatic API: the surface an outside client (another server, an app, an OpenAI-compatible library) uses to reach the models on this box. This is the surface the dashboard **Settings → API access** key gates.

Two shapes:

- **OpenAI-compatible** (`/v1/models`, `/v1/chat/completions`) — point any OpenAI client at the base URL.
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
           "n_params": 0, "hidden": 0, "layers": 0, "capabilities": {}, "is_current": true}]}
```

### POST /v1/chat/completions

OpenAI-compatible chat completions.

- Body: `model` (a model `id` from `/v1/models`, or `cloud` for the configured teacher), `messages` (list of `{role, content}`, must end with a user turn), `stream` (bool), optional `temperature`, `max_tokens`, `top_k`.
- Non-stream response: standard OpenAI `chat.completion` (`choices[].message.content`, `finish_reason:"stop"`, `usage`).
- Stream response: `text/event-stream` of `chat.completion.chunk` frames (a role frame, content-delta frames, a `finish_reason:"stop"` frame), then `data: [DONE]`. Local trained models stream true per-token as generated; the cloud teacher chunks its buffered answer.
- Errors: `400` bad body, `503` model or provider unavailable, `500` stream error.

```bash
curl http://<box-ip>:8001/v1/chat/completions \
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
