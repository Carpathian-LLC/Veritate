---
title: manifest.json
date: 2026-07-27
tags: [extensions, manifest, reference]
summary: Every key the registry reads from an extension manifest, with types and defaults.
---

# manifest.json

Every extension directory holds a `manifest.json` at its root. The registry reads it to discover the extension, mount its page route, and call its entry point. A directory without one is skipped.

## how it is read

`discover()` in `extensions/registry.py` reads the file with a plain JSON parse and nothing else. There is no schema validation, no type checking, and no required-key enforcement at read time. A file that fails to parse is logged and its directory is skipped; a file that parses but omits a key falls through to the default in the table below.

The practical consequence: `id` is the one key an extension cannot omit. It is the discovery key, the directory name in `installed/`, the catalog key, and part of the page endpoint name. Discovery falls back to the directory name when `id` is absent, but registration reads `manifest["id"]` directly, so an extension with a `register` or `page` key and no `id` fails to mount. The failure is logged and skipped, not fatal to the server.

## keys

| key | type | required | meaning |
|---|---|---|---|
| `id` | string slug | yes | unique identifier. Lowercase, no spaces. Names the directory, the catalog key, and the page endpoint `ext_page_<id>`. |
| `name` | string | no | display name in the marketplace and nav. Falls back to `id`. |
| `version` | string | no | semver string shown in the marketplace. |
| `description` | string | no | one-line summary for the catalog listing. |
| `author` | string | no | credit line shown in the marketplace listing. |
| `experimental` | bool | no, default `false` | marks the extension experimental in the installed list. |
| `page` | object | no | the self-contained page. Omit for a server-only extension. |
| `page.route` | string | with `page` | the URL the page is served at, for example `/ext/hello`. |
| `page.file` | string | with `page` | path to the HTML file relative to the extension root. |
| `page.nav_label` | string | no | nav entry label. Falls back to `name`, then `id`. |
| `api_prefix` | string | no | the URL prefix this extension's routes own. A convention the registry does not enforce. |
| `register` | string | no | path to the entry-point module relative to the extension root. The module exposes `register(app)`. |

The page route mounts only when `page.route` and `page.file` are both present. Either one alone mounts nothing.

The registry injects two keys into the in-memory manifest after reading it: `_dir`, the absolute path to the extension directory, and `_source`, either `canonical` or `installed`. Do not write either into the file.

## minimal manifests

Page only, no server routes:

```json
{
  "id": "hello",
  "name": "Hello",
  "version": "0.1.0",
  "description": "A static page that calls the platform API.",
  "page": { "route": "/ext/hello", "file": "page/index.html", "nav_label": "Hello" }
}
```

Server only, no page:

```json
{
  "id": "feed",
  "name": "Feed",
  "version": "0.1.0",
  "description": "Background data routes only.",
  "api_prefix": "/ext/feed",
  "register": "server/register.py"
}
```

## data_catalog.json

A separate optional file at the extension root, read by `extensions/data.py`. It declares large datasets an operator can download on demand, surfaced per extension in the marketplace and managed through the `/extensions/<id>/data` routes. Omit the file for an extension with no supplemental data. The download and delete mechanism belongs to the platform; which datasets exist and where they are hosted belongs to the extension.

```json
{
  "datasets": [
    {"source": "notes", "label": "Sample notes", "description": "A small text set.",
     "url": null, "approx_gb": 0.4, "schema": "date,title,body"}
  ]
}
```

| key | type | meaning |
|---|---|---|
| `source` | string slug | dataset id. Names the local cache directory and is the body field for download and delete. Unique within the catalog. |
| `label` | string | display name in the listing. |
| `description` | string | one-line summary. |
| `url` | string or null | the hosted archive. `null` marks the entry as not yet downloadable. |
| `approx_gb` | number | approximate on-disk size, shown before download. |
| `schema` | string | the dataset's column layout. |

`catalog(ext_id)` annotates each entry at read time with `present`, `files`, `size_gb`, and `downloadable`. Those are runtime fields; do not write them into the file.

Downloaded data lands in `extensions/installed/<id>/data/extension_data/<source>`. That path sits under `installed/` so it is never copied on install and survives an uninstall.
