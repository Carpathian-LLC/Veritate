---
title: extension entry point
date: 2026-07-27
tags: [extensions, reference]
summary: How an extension is discovered, how register(app) is called, and what mounts at startup.
---

# extension entry point

How an extension is discovered, registered, and mounted onto the platform server. `extensions/registry.py` owns the whole lifecycle.

## the one rule

An extension reaches the platform only through the documented HTTP API and, on the server side, the model readers and the `veritate_core.load` surface. It never imports platform internals. Everything below exists to mount the extension; none of it grants access to internals.

## discovery

At startup `discover()` scans `extensions/canonical/` and then `extensions/installed/`. Each immediate subdirectory holding a `manifest.json` is one extension. The manifest is parsed and two keys are attached: `_dir`, the absolute directory path, and `_source`, `canonical` or `installed`.

Ordering matters twice. Directories are read in sorted order within each root, and `installed/` is read second, so an id present in both roots resolves to the `installed/` copy. Ids listed in `extensions/disabled.json` are filtered out at the end of discovery.

A subdirectory with no manifest, or a manifest that fails to parse, is logged and skipped.

## registration

`register_all(app)` runs once, after every platform route is registered. Its first act is to install the live disable gate as a `before_request` hook. It then walks the discovered manifests and calls `_register_one(app, manifest)` for each, which does three things in order:

1. **Server path.** If the extension has a `server/` directory, it is prepended to `sys.path` so the entry-point module can import its own siblings by bare name. This exposes the extension's own directory only.
2. **Entry point.** If the manifest has a `register` key, that file is loaded by path under the module name `ext_<id>_register` and its `register(app)` function is called with the Flask app. This is where the extension adds its own routes.
3. **Page route.** If the manifest has both `page.route` and `page.file`, the route is added under the endpoint `ext_page_<id>` and serves the page file.

Before and after that sequence the registry snapshots `app.url_map` and records every rule that appeared, mapping each new URL rule to the extension id. That record is what lets the disable gate know which routes belong to which extension, so an extension never declares its routes anywhere.

One extension failing during registration is caught, logged, and skipped. It never aborts another extension or server startup.

## the register(app) function

The entry-point module exposes one module-level function:

```python
def register(app):
    @app.route("/ext/hello/data")
    def hello_data():
        return {"ok": True, "items": []}
```

The contract:

- The signature is exactly `register(app)`, one positional argument. Nothing else is passed: no registry, no manifest, no configuration object, no directory path. An extension that needs its own paths derives them from `__file__`.
- The return value is ignored.
- The module is loaded by file path, not imported as a package, so relative imports fail. Sibling modules import by bare name, which works because `<extension>/server` was prepended to `sys.path` immediately before the load.
- Every route it adds should live under the manifest's `api_prefix`. The registry does not enforce this; a route outside the prefix risks colliding with a platform route or another extension.
- Server code may read disk through the model readers and load checkpoints through `veritate_core.load`. It must not write into `models/<name>/` or `data/corpus/`, and must not import `veritate_mri` internals.
- Routes return plain JSON. Flask serializes a returned dict, and the platform's global error handler also returns JSON, so a caller can always read the body.

A page-only extension omits `register`. A server-only extension omits `page`.

## the page route and the prefix convention

The page file is served at `page.route`, conventionally `/ext/<id>`. It is a self-contained HTML document with no build step and no shared code with the dashboard. From the browser it calls the platform HTTP API and its own routes by URL.

Keeping the page route, the `api_prefix`, and the server routes all under `/ext/<id>/` namespaces the extension so nothing collides.

## startup lifecycle

Routes and pages mount once, at server start. There is no hot reload and no per-request discovery. Two consequences follow, and they are not symmetric.

- **Disabling and re-enabling an extension that was present at the last startup is live.** Its routes are already registered, and the gate decides per request whether to serve or return 404.
- **An extension that was absent at the last startup has no registered routes at all.** Installing it copies its files, and its routes and page appear at the next server start.

Editing an extension's files while the server runs has no effect until a restart.

## what is not available

- **No sandbox.** Isolation is by convention: namespaced routes, API-only access from the page, no internal imports. A server route runs with full process access, so extension server code is worth reading before installing it.
- **No remote code download.** Install copies from the bundled `canonical/` source. Extension datasets are a separate mechanism and do download from a hosted URL.
