# manifest.json

Every extension directory contains a `manifest.json` at its root. The registry
reads it to discover the extension, mount its page route, and call its entry point
(`extensions/registry.py:42`, `extensions/registry.py:62`). It is the only file the
registry requires; a directory under `installed/` with no `manifest.json` is
skipped (`extensions/registry.py:49`).

## schema

```json
{
  "id": "market",
  "name": "Market LLM",
  "version": "1.0.0",
  "author": "veritate",
  "kind": "extension",
  "description": "Byte-model market comparison ground.",
  "experimental": true,
  "page": {
    "route": "/ext/market",
    "file": "page/index.html",
    "nav_label": "Market"
  },
  "api_prefix": "/ext/market",
  "register": "register.py"
}
```

## fields

| field | type | required | meaning |
|---|---|---|---|
| `id` | string (slug) | yes | unique identifier. Names the `installed/<id>/` and `canonical/<id>/` directories, the catalog key, and the page endpoint (`ext_page_<id>`). Lowercase, no spaces. |
| `name` | string | yes | display name shown in the marketplace and dashboard. |
| `version` | string (semver) | yes | extension version, e.g. `1.0.0`. |
| `author` | string | yes | `veritate` for bundled canonical extensions, or a user identifier. |
| `kind` | string | yes | always `"extension"`. Reserved for distinguishing future bundle types. |
| `description` | string | yes | one-line summary for the marketplace listing. |
| `experimental` | bool | no (default `false`) | marks the extension as experimental; surfaced through `list_installed` (`extensions/registry.py:110`). |
| `page` | object | no | the self-contained page. Omit for a server-only extension (routes but no UI). See below. |
| `page.route` | string | required if `page` present | the URL the page is served at, e.g. `/ext/<id>`. |
| `page.file` | string | required if `page` present | path to the HTML file relative to the extension root, e.g. `page/index.html`. |
| `page.nav_label` | string | no | label for the dashboard nav entry. Omit to register the page route without a nav link. |
| `api_prefix` | string | no | the URL prefix the extension's own server routes own, e.g. `/ext/<id>`. Convention, not enforced: the registry does not police it, but every route an extension registers must live under this prefix to avoid colliding with platform routes. |
| `register` | string | no | path to the entry-point module relative to the extension root, e.g. `register.py`. The module must expose `register(app)`. Omit for a page-only extension with no server routes. |

The registry adds a transient `_dir` key (absolute path to the extension
directory) to the in-memory manifest after reading it (`extensions/registry.py:57`).
Do not write `_dir` into the file; it is set at discovery time.

## minimal manifests

Page only, no server routes:

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

Server only, no page (registers routes under its prefix):

```json
{
  "id": "feed",
  "name": "Feed",
  "version": "0.1.0",
  "author": "user",
  "kind": "extension",
  "description": "Background data routes only.",
  "api_prefix": "/ext/feed",
  "register": "register.py"
}
```

## see also

- [entry_point.md](entry_point.md) — how `register` and `page` are consumed at startup.
- [marketplace.md](marketplace.md) — how `id`, `name`, `version`, `description` appear in the catalog.
- [../api/rest_api.md](../api/rest_api.md) — the platform API the page and routes call.
