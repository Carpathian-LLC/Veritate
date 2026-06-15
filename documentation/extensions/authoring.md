# external extension authoring

Guide for building a self-contained project on top of Veritate without modifying
platform code. An **extension** here is a third-party UI + logic bundle that runs
beside the dashboard, calls the platform HTTP API, and/or loads a model the user
trained. It is distinct from an internal **plugin** (= a trainer; see
[plugins/contract.md](../plugins/contract.md)). The two terms must not be conflated:
plugin = trainer contract, extension = user-authored add-on.

## what exists today vs the target

The "drop a folder, get an isolated tab" experience is **partially implemented**.
Read this section before assuming a capability is present.

| Capability | Status |
| --- | --- |
| Self-contained UI that runs beside the dashboard and is deletable as a unit | yes (standalone-module IIFE, or a separate page) |
| Calls the platform HTTP API read-only | yes (any registered route) |
| Loads a user-trained model and runs it | yes, server-side only (a route loads the checkpoint; the browser never loads weights) |
| A separate isolated page on its own route | yes, but the route is hardcoded in `app.py` (e.g. `/market`) |
| An experimental on/off gate in the dashboard | yes (the experimental settings toggle) |
| **Zero-edit folder drop that auto-surfaces a tab** | **not yet implemented** — a tab still requires editing `index.html` and the `valid` allowlist |
| **Runtime discovery of extension folders** | **not yet implemented** — there is no folder scan, no extension manifest, no dynamic registration |
| **Sandbox/isolation enforcement (iframe, CSP, capability scoping)** | **not yet implemented** — isolation today is by convention (namespaced CSS, IIFE scope, read-only API use), not enforced |

Build to what exists. Where this guide says "not yet implemented," do not write
code that assumes the missing mechanism.

## the two extension shapes available now

### shape A: standalone module (in-dashboard panel or tab)

A self-contained JavaScript IIFE that mounts itself into the dashboard. Full
mechanics in [architecture/frontend/standalone_modules.md](../architecture/frontend/standalone_modules.md).
The IIFE keeps all state out of the global namespace, mounts into a host element,
and can be deleted as one unit. `prune.js` and `tutorial.js` are the live examples.

Cost of this shape today: surfacing a **tab** still requires three edits to
`index.html` (a `<script>`/`<link>` in `<head>`, a `.tab` element, a `.tab-body`
host) plus adding the tab name to the `valid` allowlist in `index.js`
(see [architecture/frontend/tab_system.md](../architecture/frontend/tab_system.md)).
Until folder discovery ships, a tab is not a zero-edit drop.

### shape B: separate page on its own route (full isolation)

A standalone HTML page served at its own route, calling only `/<prefix>/*` on the
canonical server. The Market LLM page is the reference implementation: a single
`web/market.html` with no build step, no framework, no shared JS with the main
dashboard, served at `/market`. This is the closest existing match to the
"100 percent isolated" target — the page shares nothing with `index.html` except
the HTTP API and the CSS token set. The route is currently hardcoded in
[app.py:81](../../veritate_mri/app.py#L81); a drop-in route registry is not yet
implemented.

## isolation contract (what an extension must not touch)

These boundaries hold for both shapes and mirror the trainer-plugin forbidden list
(plugins/contract.md):

- **Read-only on canonical state.** Do not mutate training, chat, or RAG state.
  Load checkpoints and read corpora through the existing routes; never write into
  `models/<name>/` or `trainers/corpus/` from an extension.
- **No platform-internal imports in server code.** If an extension needs a
  server-side route, it goes through the same `register(app)` pattern as every
  other route ([architecture/backend/routes.md](../architecture/backend/routes.md))
  and reads disk through [readers](../architecture/backend/readers.md), never
  `open()` directly. Browser code calls HTTP only.
- **Namespace your CSS and DOM ids** (e.g. `.myext-*`) so nothing collides with
  `index.css` / `index.html`.
- **Fail soft.** A missing optional dependency must never break dashboard startup;
  the market routes lazy-import their package for exactly this reason
  ([architecture/backend/market_routes.md](../architecture/backend/market_routes.md)).

## calling the platform API

Every dashboard feature is a thin browser caller over a Flask route. Each route
module owns one concern and exports `register(app)`; the full inventory is in
[architecture/backend/routes.md](../architecture/backend/routes.md). Routes return
JSON (dicts auto-jsonified); the global error handler also returns JSON, so a
caller can always `r.json()`. Useful prefixes for an extension:

- `/models/*` — model listing, config, checkpoints (discover what the user trained).
- `/runs/*` — training run data and CSVs.
- `/generate`, `/meta` — PyTorch inference brain ([architecture/backend/inference_brain.md](../architecture/backend/inference_brain.md)).
- `/hybrid/*` — RAG chat ([architecture/backend/hybrid_chat.md](../architecture/backend/hybrid_chat.md)).
- `/sys/*` — system metrics, `/versions`.

Read [architecture/frontend/data_flow.md](../architecture/frontend/data_flow.md)
for the existing fetch conventions (caching, status reporting, empty states).

## loading a user-trained model

Models live under `models/<name>/` (gitignored, machine-local;
[training/storage.md](../training/storage.md)). An extension loads a model
**server-side** in a route, not in the browser. The market subsystem is the
reference: a route lazily loads the latest checkpoint of a named byte model via the
model registry and `veritate_core.load.load_from_state_dict`, runs inference on CPU
(no MPS contention with a live training run), and returns plain JSON
([market/market_platform.md](../market/market_platform.md)). To list models a user
can pick from, call `/models/*` or follow the market pattern of a dedicated
`*_models` endpoint that filters to models with at least one checkpoint.

## the experimental gate

An extension under active development can hide behind the experimental settings
flag. The flag is a boolean in `data/mri_settings.json` (default `false`,
machine-local), read/written via `GET`/`POST /settings`
([architecture/backend/settings.md](../architecture/backend/settings.md)). Today it
gates one thing: the visibility of the Market LLM nav link
([index.html:81](../../veritate_mri/web/index.html#L81)); the `/market` route stays
reachable regardless. The gate hides nav entry points; it does not sandbox or
unload anything. Treat it as a "show in the UI" switch, not an isolation boundary.

## naming: the eventual rename of "Experiments"

The dashboard tab/feature now labeled **Experiments / experimental** will be
renamed. It cannot become "plugins": that term is already the internal trainer
contract ([plugins/contract.md](../plugins/contract.md)) and reusing it would
collide. This guide uses **extension** for the user-authored add-on concept.
Candidate names for the eventual rename, with the one-line tradeoff each:

- **Extensions** — matches the browser/IDE mental model of optional add-ons; clear
  separation from internal "plugins"; slightly generic.
- **Apps** — strongest signal of "self-contained, isolated, drop-in"; risks
  implying an app store / packaging system that does not exist yet.
- **Labs** — keeps the experimental, in-progress connotation of the current name;
  weaker at conveying "third-party, isolated bundle."

No code or route is renamed by this doc. `/market`, the `experimental` settings
key, and the `#navMarket` id stay as-is until a rename ships in code.

## see also

- [plugins/contract.md](../plugins/contract.md) — the internal plugin (trainer) contract; do not confuse with extensions.
- [architecture/frontend/standalone_modules.md](../architecture/frontend/standalone_modules.md) — shape A mechanics.
- [architecture/frontend/tab_system.md](../architecture/frontend/tab_system.md) — why a tab needs the `valid` allowlist edit today.
- [architecture/backend/routes.md](../architecture/backend/routes.md) — the full HTTP API surface to call.
- [architecture/backend/settings.md](../architecture/backend/settings.md) — the experimental gate.
