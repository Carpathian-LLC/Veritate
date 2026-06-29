# extension entry point

How an installed extension is discovered, registered, and mounted onto the
platform server. The registry (`extensions/registry.py`) owns the whole lifecycle.

## the one rule

An extension reaches the platform **only** through the documented HTTP API
([../api/rest_api.md](../api/rest_api.md)) and, server-side, the model `readers` +
`veritate_core.load` surface. It never imports platform internals
(`veritate_mri.*` modules, runtime state, route objects). Everything below exists
to mount the extension; none of it grants access to internals.

## discovery

At startup the registry scans `extensions/installed/`. Each immediate subdirectory
that contains a `manifest.json` is one extension; the manifest is parsed and the
absolute directory path is attached as `_dir` (`extensions/registry.py:42`). A
subdirectory with no manifest, or a manifest that fails to parse, is logged and
skipped (`extensions/registry.py:49`, `extensions/registry.py:55`). Discovery is
ordered by directory name.

## registration

`register_all(app)` is called once, after every platform route is registered
(`veritate_mri/app.py:163`). For each discovered manifest it calls
`_register_one(app, manifest)` (`extensions/registry.py:62`), which does three
things in order:

1. **Server path.** If the extension has a `server/` directory, it is prepended to
   `sys.path` so the entry-point module can import its own siblings
   (`extensions/registry.py:64`). This adds the extension's `server/` dir only; it
   does not expose platform internals.
2. **Entry point.** If the manifest has a `register` field, the named module is
   loaded by file path under a namespaced module name (`ext_<id>_register`) and its
   `register(app)` function is called with the Flask app
   (`extensions/registry.py:67`). This is where the extension adds its own routes.
3. **Page route.** If the manifest has `page.route` and `page.file`, the route is
   mounted under the endpoint `ext_page_<id>` and serves the page file via
   `send_from_directory` (`extensions/registry.py:74`).

One extension failing during registration is caught, logged, and skipped; it never
aborts the other extensions or server startup (`extensions/registry.py:95`).

## the `register(app)` function

The entry-point module (named by the manifest `register` field, conventionally
`register.py`) exposes a single function:

```python
def register(app):
    @app.route("/ext/<id>/data")
    def ext_data():
        return {"ok": True, "items": [...]}
```

Contract:

- It takes the Flask `app` and adds routes to it. Nothing else.
- Every route it adds lives under the manifest's `api_prefix` (e.g. `/ext/<id>/*`).
  The registry does not enforce this; a route outside the prefix risks colliding
  with a platform route or another extension.
- Server-side it may read disk through the model `readers` and load checkpoints
  through `veritate_core.load`. It must not write into `models/<name>/` or
  `trainers/corpus/`, and must not import `veritate_mri` internals.
- It returns plain JSON (Flask jsonifies dicts). The platform's global error
  handler also returns JSON, so callers can always read `r.json()`
  ([../api/rest_api.md](../api/rest_api.md)).

A page-only extension omits `register`; a server-only extension omits `page`.

## the page route + api_prefix convention

The page (`page.file`, conventionally `page/index.html`) is served at `page.route`
(conventionally `/ext/<id>`). It is a self-contained HTML/CSS/JS document with no
build step and no shared code with the dashboard. From the browser it calls the
platform HTTP API and its own `api_prefix` routes by URL.

Keeping the page route, the `api_prefix`, and the server routes all under
`/ext/<id>/` namespaces the extension so nothing collides with platform routes or
another extension.

## startup lifecycle

Routes and pages are mounted **once, at server start**, in `register_all`. There is
no hot-reload and no per-request discovery. Consequences:

- An install or uninstall takes effect on the **next server start**, not
  immediately ([marketplace.md](marketplace.md)).
- Editing an extension's files while the server runs has no effect until restart.

## not yet available

- **No sandbox.** Isolation is by convention (namespaced routes/page, API-only
  access, no internal imports), not enforced by an iframe, CSP, or capability
  scoping. A misbehaving server route runs with full process access; review
  extension server code before installing.
- **No remote download.** Installation copies from the bundled `canonical/` source
  only ([marketplace.md](marketplace.md)). Remote-URL download is a documented
  future capability, not present in v1.

## see also

- [manifest.md](manifest.md) — the fields this lifecycle consumes.
- [marketplace.md](marketplace.md) — install/uninstall and restart-to-activate.
- [authoring.md](authoring.md) — building the page and server modules.
- [../api/rest_api.md](../api/rest_api.md) — the API surface an entry point and page call.
