# manifest.json

Every extension directory contains a `manifest.json` at its root. The registry
reads it to discover the extension, mount its page route, and call its entry point
(`extensions/registry.py:42`, `extensions/registry.py:71`). It is the only file the
registry requires; a directory with no `manifest.json` is
skipped (`extensions/registry.py:53`).

## schema

```json
{
  "id": "market",
  "name": "Market LLM",
  "version": "1.0.0",
  "author": "veritate",
  "kind": "extension",
  "description": "Byte-LLM market forecasting: hindcast, benchmark, and live decision support over price-series corpora.",
  "experimental": true,
  "page": {
    "route": "/market",
    "file": "page/index.html",
    "nav_label": "Market LLM"
  },
  "api_prefix": "/market",
  "register": "register.py"
}
```

This is the shipped canonical Market manifest (`extensions/canonical/market/manifest.json`).
New extensions should pick an unused `api_prefix` and page `route` (e.g.
`/ext/<id>`) to avoid colliding with platform routes.

## fields

| field | type | required | meaning |
|---|---|---|---|
| `id` | string (slug) | yes | unique identifier. Names the `installed/<id>/` and `canonical/<id>/` directories, the catalog key, and the page endpoint (`ext_page_<id>`). Lowercase, no spaces. |
| `name` | string | yes | display name shown in the marketplace and dashboard. |
| `version` | string (semver) | yes | extension version, e.g. `1.0.0`. |
| `author` | string | yes | `veritate` for bundled canonical extensions, or a user identifier. |
| `kind` | string | yes | always `"extension"`. Reserved for distinguishing future bundle types. |
| `description` | string | yes | one-line summary for the marketplace listing. |
| `experimental` | bool | no (default `false`) | marks the extension as experimental; surfaced through `list_installed` (`extensions/registry.py:109`). |
| `page` | object | no | the self-contained page. Omit for a server-only extension (routes but no UI). See below. |
| `page.route` | string | required if `page` present | the URL the page is served at, e.g. `/ext/<id>`. |
| `page.file` | string | required if `page` present | path to the HTML file relative to the extension root, e.g. `page/index.html`. |
| `page.nav_label` | string | no | label for the dashboard nav entry. Omit to register the page route without a nav link. |
| `api_prefix` | string | no | the URL prefix the extension's own server routes own, e.g. `/ext/<id>`. Convention, not enforced: the registry does not police it, but every route an extension registers must live under this prefix to avoid colliding with platform routes. |
| `register` | string | no | path to the entry-point module relative to the extension root, e.g. `register.py`. The module must expose `register(app)`. Omit for a page-only extension with no server routes. |

The registry adds a transient `_dir` key (absolute path to the extension
directory) to the in-memory manifest after reading it (`extensions/registry.py:61`).
Do not write `_dir` into the file; it is set at discovery time.

## data_catalog.json (optional)

Separate from `manifest.json`. An extension may ship a `data_catalog.json` at its
root declaring large optional supplemental datasets the operator can download on
demand. It is read by the generic data module (`extensions/data.py:39`), surfaced
per-extension in the marketplace, and managed through `GET /extensions/<id>/data`,
`POST /extensions/<id>/data/download`, and `POST /extensions/<id>/data/delete`
([../api/rest_api.md](../api/rest_api.md)). Omit the file for an extension with no
supplemental data. The mechanism is platform-owned; the catalog (which datasets,
where hosted) is owned by the extension.

```json
{
  "datasets": [
    {"source": "crypto", "label": "Crypto majors (1m)", "description": "200 major Binance USDT pairs at 1-minute.", "url": null, "approx_gb": 34.0, "schema": "time,open,high,low,close,volume,trades,taker_buy"},
    {"source": "stocks", "label": "US Stocks (daily)", "description": "S&P 500 daily adjusted OHLCV bars.", "url": null, "approx_gb": 0.43, "schema": "date,open,high,low,close,adjclose,volume"}
  ]
}
```

| field | type | required | meaning |
|---|---|---|---|
| `source` | string (slug) | yes | dataset id. Names the local cache dir `extensions/installed/<id>/data/extension_data/<source>` and is the body field for download/delete. Unique within the catalog. |
| `label` | string | yes | display name shown in the marketplace listing. |
| `description` | string | yes | one-line summary of the dataset's contents. |
| `url` | string \| null | yes | hosted archive to download. `null` = placeholder ("coming soon"): the entry lists but is not downloadable until a URL is set (`extensions/data.py:97`). |
| `approx_gb` | number | yes | approximate on-disk size in GB, shown before download so the operator can judge the cost. |
| `schema` | string | yes | the dataset's column layout (CSV header), e.g. `date,open,high,low,close,adjclose,volume`. |

`catalog(ext_id)` annotates each entry at read time with `present:bool`,
`files:int`, `size_gb:float` (measured locally), and `downloadable:bool`
(`bool(url)`) (`extensions/data.py:79`). These are runtime fields; do not write them
into the file. The full annotated response shape is in
[../api/rest_api.md](../api/rest_api.md). The market extension's
`data_catalog.json` is the reference example.

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
