---
title: authoring an extension
date: 2026-07-27
tags: [extensions, authoring]
summary: Build a self-contained page that runs beside the dashboard without touching platform code.
---

# authoring an extension

An extension is one directory containing a manifest, an optional entry point, a self-contained page, and optional server modules. The registry discovers it, mounts it at startup, and the marketplace installs and removes it.

## directory layout

```
extensions/
  registry.py            platform: discovery, registration, install, uninstall
  data.py                platform: optional per-extension dataset downloads
  catalog.json           platform: the marketplace listing
  disabled.json          platform: ids the live gate blocks (created on first uninstall)
  canonical/<id>/        bundled install source
  installed/<id>/        install target and the home of user-authored extensions
    manifest.json        required: identity, page, entry point
    register.py          optional: exposes register(app), adds server routes
    page/index.html      the self-contained page
    server/              optional: extra modules importable by register.py
    data_catalog.json    optional: large downloadable datasets
```

`discover()` scans **both** `canonical/` and `installed/`, canonical first. An id present in both resolves to the `installed/` copy, which is what makes a local edit override a bundled extension. A directory without a `manifest.json` is skipped.

## manifest

`manifest.json` is the only required file. It names the extension, points at the page, and names the entry-point module. Every key is in the manifest entry of this wiki. Minimal page-only example:

```json
{
  "id": "hello",
  "name": "Hello",
  "version": "0.1.0",
  "description": "A static page that calls the platform API.",
  "page": { "route": "/ext/hello", "file": "page/index.html", "nav_label": "Hello" }
}
```

Keep the page route, the `api_prefix`, and every server route under `/ext/<id>/` so nothing collides with a platform route or another extension.

## the entry point

An extension that needs its own server routes adds a `register` key naming a module that exposes `register(app)`:

```python
def register(app):
    @app.route("/ext/hello/data")
    def hello_data():
        return {"ok": True, "items": []}
```

The registry loads that module by file path and calls the function with the Flask app. Full sequence in the entry point entry.

A page-only extension omits `register`. A server-only extension omits `page`.

## the page

`page/index.html` is a single document with its own HTML, CSS, and JavaScript, served at `page.route`. No build step, no framework requirement, no shared code with the dashboard. From the browser it calls the platform HTTP API and its own routes by URL.

Every dashboard feature is itself a browser caller over a Flask route, so the same calls are available to an extension. Routes return JSON and the global error handler also returns JSON, so a caller can always read the body and treat any non-2xx as failure. The endpoints most extensions need:

- `GET /pytorch-models` to list loadable models with shape and capability metadata.
- `GET /meta` for the current model's metadata and shape.
- `GET /generate` for byte-by-byte generation with introspection.
- `POST /hybrid/chat` for conversational chat, or `POST /hybrid/chat/stream` for the streaming twin.
- `GET /runs` and `GET /run/<name>/csv` for training run data.
- `GET /settings` and `POST /settings` for settings.
- `GET /versions` for the version ledger.

## loading a trained model

Models live under `models/<name>/`, machine-local and outside version control. An extension loads a model on the server, inside one of its `register(app)` routes, never in the browser. The pattern is to lazily load the latest checkpoint of a named model through the model readers and `veritate_core.load`, run inference on the CPU so a live training run keeps the accelerator, and return plain JSON. For a model picker, call `GET /pytorch-models` and filter to models with at least one checkpoint.

## isolation rules

Isolation is by convention. There is no iframe, no content-security policy, and no capability sandbox. A server route added by an extension runs with the same process access as platform code, so extension server code is worth reading before installing it. Honor these boundaries:

- **API only from the browser.** The page reaches the platform through the documented HTTP API and the extension's own routes, nothing else.
- **No platform-internal imports in server code.** Read disk through the model readers, load checkpoints through `veritate_core.load`, never import `veritate_mri` internals and never open platform files directly.
- **Read-only on canonical state.** Do not mutate training, chat, or retrieval state. Never write into `models/<name>/` or `trainers/corpus/`.
- **Namespace everything.** Page route, `api_prefix`, server routes, CSS classes, and DOM ids all carry the extension prefix.
- **Fail soft.** Lazy-import optional packages inside the route that needs them. One extension failing during registration is logged and skipped and never aborts another extension or server startup.

## what exists

| capability | state |
|---|---|
| directory-based extension with manifest, entry point, page | present |
| startup discovery of `canonical/` and `installed/` | present |
| page mounted at its own namespaced route | present |
| server routes through `register(app)` | present |
| marketplace catalog with install and uninstall routes | present |
| install by copying from the bundled `canonical/` source | present |
| live uninstall and reinstall with no server restart | present, for extensions registered at the last startup |
| extensions nav and marketplace gated by the settings flag | present |
| download of extension code from a remote URL | not present; install copies from `canonical/` only |
| enforced sandbox | not present; isolation is by convention |
| activating a brand-new extension without a restart | not present; its routes mount at the next startup |

Build to what exists. Where this entry says a mechanism is not present, do not write code that assumes it.
