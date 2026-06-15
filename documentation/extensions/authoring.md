# authoring an extension

Build a self-contained page that runs beside the Veritate dashboard without
modifying platform code. An **extension** is a directory containing a manifest, an
optional entry point, a self-contained page, and optional server modules. It is
discovered, registered at startup, and installed/uninstalled through the
marketplace. It talks to the platform **only** through the documented HTTP API
([../api/rest_api.md](../api/rest_api.md)) plus the server-side model-loading
surface.

An extension is distinct from an internal **plugin** (a trainer; see
[../../developer_documentation/plugins/contract.md](../../developer_documentation/plugins/contract.md)).
The terms are not interchangeable: plugin = trainer contract, extension =
self-contained add-on page.

## directory layout

An extension is one directory, keyed by its `id`:

```
extensions/
  registry.py            platform: discovery + registration + install/uninstall
  catalog.json           platform: the marketplace listing
  canonical/<id>/        bundled install source (canonical extensions)
  installed/<id>/        active extensions: discovered + registered at startup
    manifest.json        required: identity, page, entry point (see manifest.md)
    register.py          optional: exposes register(app); adds server routes
    page/index.html      the self-contained page (own HTML/CSS/JS, no build step)
    server/              optional: extra server-side modules for register.py
```

`canonical/<id>/` is the bundled install source; `install` copies it into
`installed/<id>/`. A user-created extension is placed directly in
`installed/<id>/`. Only `installed/` is scanned at startup. See
[marketplace.md](marketplace.md) for the install flow and
[entry_point.md](entry_point.md) for the registration sequence.

## manifest

`manifest.json` is the only required file. It names the extension, points at the
page, and names the entry-point module. Full schema in
[manifest.md](manifest.md). Minimal page-only example:

```json
{
  "id": "hello",
  "name": "Hello",
  "version": "0.1.0",
  "author": "user",
  "kind": "extension",
  "description": "A static page that calls the platform API.",
  "page": { "route": "/ext/hello", "file": "page/index.html", "nav_label": "Hello" }
}
```

Keep the page route, `api_prefix`, and any server routes under `/ext/<id>/` so
nothing collides with platform routes or another extension.

## the entry point (register.py)

If the extension needs its own server routes, add a `register` field to the
manifest naming a module that exposes `register(app)`:

```python
def register(app):
    @app.route("/ext/hello/data")
    def hello_data():
        return {"ok": True, "items": [...]}
```

At startup the registry imports this module and calls `register(app)` with the
Flask app (`extensions/registry.py:67`). Every route it adds must live under the
extension's `api_prefix`. Server code may read disk through the model `readers` and
load checkpoints through `veritate_core.load`; it must not import `veritate_mri`
internals. Full registration sequence in [entry_point.md](entry_point.md).

A page-only extension omits `register`. A server-only extension omits `page`.

## the self-contained page

The page (`page/index.html`) is a single document with its own HTML, CSS, and JS,
served at `page.route`. No build step, no framework requirement, no shared code
with `index.html`. From the browser it talks to the platform over the HTTP API and
to its own `api_prefix` routes by URL.

Every dashboard feature is itself a thin browser caller over a Flask route, so the
same calls are available to an extension. Routes return JSON (dicts are
auto-jsonified), and the global error handler also returns JSON, so a caller can
always `r.json()` and treat any non-2xx as failure
([../api/rest_api.md](../api/rest_api.md)). Endpoints an extension commonly uses:

- `GET /pytorch-models` — list loadable models with shape + capability metadata.
- `GET /meta` — current model metadata + shape.
- `GET /generate` — byte-by-byte generation with introspection (SSE).
- `POST /hybrid/chat` — conversational chat with RAG/teacher.
- `GET /runs`, `GET /run/<name>/csv` — training run data + per-step CSV.
- `GET /settings`, `POST /settings` — user settings.
- `GET /versions` — the version ledger.

The full endpoint inventory, parameters, and response shapes are in
[../api/rest_api.md](../api/rest_api.md). That reference is the contract: anything
not in it is not part of the stable surface.

## loading a user-trained model

Models live under `models/<name>/` (gitignored, machine-local). An extension loads
a model **server-side**, in one of its `register(app)` routes, never in the
browser. The reference pattern: a route lazily loads the latest checkpoint of a
named byte model through the model registry and `veritate_core.load`, runs
inference on CPU (no MPS contention with a live training run), and returns plain
JSON. To present a model picker, call `GET /pytorch-models` and filter to models
with at least one checkpoint.

## isolation rules

Isolation today is **by convention**, not enforced (no iframe, CSP, or capability
sandbox yet). Honor these boundaries:

- **API-only from the browser.** The page reaches the platform only through the
  documented HTTP API and the extension's own `api_prefix` routes.
- **No platform-internal imports in server code.** A server route is added through
  `register(app)`. It reads disk through the model `readers` and loads checkpoints
  through `veritate_core.load`. It never imports `veritate_mri` internals and never
  `open()`s platform files directly.
- **Read-only on canonical state.** Do not mutate training, chat, or RAG state.
  Never write into `models/<name>/` or `trainers/corpus/` from an extension.
- **Namespace everything.** Page route, `api_prefix`, server routes, CSS classes,
  and DOM ids all under an extension-specific prefix (e.g. `/ext/<id>/`,
  `.<id>-*`).
- **Fail soft.** A missing optional dependency must never break dashboard startup.
  Lazy-import optional packages inside the route that needs them; one failing
  extension is logged and skipped, never aborting the others or server startup
  (`extensions/registry.py:95`).

## what exists vs what is future

| capability | status |
|---|---|
| directory-based extension with manifest, entry point, page | yes |
| startup discovery of `installed/<id>/` + automatic registration | yes (`extensions/registry.py:42`) |
| page mounted at its own route, namespaced | yes (`extensions/registry.py:74`) |
| server routes via `register(app)` under `api_prefix` | yes |
| marketplace catalog + install/uninstall endpoints | yes ([marketplace.md](marketplace.md)) |
| install from the bundled `canonical/` source | yes (`extensions/registry.py:128`) |
| extensions nav + marketplace gated by the settings flag | yes |
| **remote-URL download of an extension** | future — install copies from `canonical/` only |
| **enforced sandbox (iframe / CSP / capability scoping)** | not yet — isolation is by convention |
| **hot reload of an extension without a server restart** | not yet — installs activate on the next start |

Build to what exists. Where this guide says "future" or "not yet," do not write
code that assumes the missing mechanism.

## lifecycle: restart to activate

Routes and pages mount once, at server start, in `register_all`
(`veritate_mri/app.py:163`). An install or uninstall changes disk immediately but
takes effect only on the **next server start**. Editing an extension's files while
the server runs has no effect until restart. See [marketplace.md](marketplace.md).

## see also

- [manifest.md](manifest.md) — the `manifest.json` schema.
- [entry_point.md](entry_point.md) — discovery, `register(app)`, the startup lifecycle.
- [marketplace.md](marketplace.md) — catalog, install/uninstall, restart-to-activate.
- [../api/rest_api.md](../api/rest_api.md) — the platform API contract.
- [../../developer_documentation/plugins/contract.md](../../developer_documentation/plugins/contract.md) — the internal plugin (trainer) contract; do not confuse with extensions.
