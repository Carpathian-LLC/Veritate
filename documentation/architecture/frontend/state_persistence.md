# State persistence

## What it is

Where frontend state lives across page reloads (localStorage) versus where it lives for the session only (in-memory) versus where it lives on the server.

## localStorage

Client-side persistence. Survives reloads, lost on private-browsing or storage clears.

- **Chat history** at [index.js:1490–1497](../../../veritate_mri/web/index.js#L1490) — generation tab stores the last 100 messages under a `CHAT_KEY`.
- **Run picks** at [index.js:2420–2424](../../../veritate_mri/web/index.js#L2420) — training tab caches the selected run name and other UI state.
- **Trainer form values** at [index.js:7729](../../../veritate_mri/web/index.js#L7729) — trainer config inputs (batch size, lr, epochs) cached so reloads don't lose work.

Every localStorage call is wrapped in `try/catch`. Missing storage (private browsing, quota exceeded) silently falls back to no-op.

## In-memory state (session only)

- `meta` — model metadata fetched once at startup ([index.js:2306–2307](../../../veritate_mri/web/index.js#L2306)). Stale until the user loads a different model.
- `frames` — generation telemetry populated by `/generate` response stream; cleared on new prompt.
- `learningState` — checkpoint timelines loaded on-demand when the learning tab activates; cached in memory.
- `trainLastText` — last training CSV text; updated by polling.

## Server-side state

What the frontend reads but does not own:

- `/settings` — dashboard preferences (theme, polling cadence, device prefs). Stored in `data/mri_settings.json`.
- `/runs` — derived from `models/<name>/train.csv` files.
- `/heartbeat/status` — read from `data/heartbeat_state.json`.

## Dependencies

- [data_flow.md](data_flow.md) — the fetch wrappers that read server state.
- Backend [settings_routes.py](../../../veritate_mri/routes/settings_routes.py) for the settings round-trip.

## Pitfalls

- localStorage is shared per origin. Two dashboards on the same machine (e.g., two ports) collide on chat history and run picks. Namespace keys if you add another instance.
- Clearing localStorage doesn't clear in-memory state — a page reload is required for a true clean state.
- Don't store secrets in localStorage. The `ai_api_key` field in settings lives server-side for this reason.
