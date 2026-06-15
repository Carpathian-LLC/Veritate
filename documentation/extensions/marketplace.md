# extension marketplace

The marketplace lists installable extensions, installs them from the bundled
source, and uninstalls them. The dashboard nav + marketplace UI appear when the
extensions settings flag is on. The registry (`extensions/registry.py`) does the
disk work; the HTTP routes (`veritate_mri/routes/extensions_routes.py`) expose it.

## the catalog

`extensions/catalog.json` is the marketplace listing. Shape:

```json
{
  "extensions": [
    {
      "id": "market",
      "name": "Market LLM",
      "version": "1.0.0",
      "author": "veritate",
      "description": "Byte-LLM market forecasting: hindcast, benchmark, and live decision support over price-series corpora.",
      "experimental": true,
      "builtin": true
    }
  ]
}
```

`load_catalog()` reads this file and annotates each entry with an `installed`
boolean computed from whether the `id` is currently discovered
(`extensions/registry.py:130`). The catalog is the listing; it does not contain the
extension code.

## canonical vs user-created

- **Canonical extensions** ship bundled under `extensions/canonical/<id>/`. This is
  the install source: `install` copies `canonical/<id>` into `installed/<id>`
  (`extensions/registry.py:134`). Authored by `veritate`. The Market LLM extension
  (`extensions/canonical/market/`) is the reference canonical extension. Canonical
  extensions are discovered and active without an explicit install
  (`extensions/registry.py:42`); install copies one into `installed/` so a user
  edit overrides the bundled copy.
- **User-created extensions** are placed directly under `extensions/installed/<id>/`
  by the author. They are discovered and registered at startup exactly like an
  installed canonical extension. They do not need a catalog entry to run; the
  catalog is only the marketplace listing.

There is no remote download in v1. `install` resolves an `id` to
`canonical/<id>` and copies it; if there is no canonical source it raises and the
endpoint returns `404` (`extensions/registry.py:136`). Remote-URL download is a
documented future capability, not present today.

## endpoints

Full request/response detail lives in [../api/rest_api.md](../api/rest_api.md).

| method + path | purpose |
|---|---|
| `GET /extensions` | list installed extensions (`id`, `name`, `version`, `nav_label`, `route`, `experimental`) |
| `GET /extensions/catalog` | the marketplace catalog, each entry annotated with `installed` |
| `POST /extensions/install` | body `{ "id": "<id>" }`; copy `canonical/<id>` → `installed/<id>`. `400` if `id` missing, `404` if no canonical source |
| `POST /extensions/uninstall` | body `{ "id": "<id>" }`; remove `installed/<id>`. `400` if `id` missing |
| `GET /extensions/<id>/data` | the extension's supplemental-dataset catalog, annotated with local presence |
| `POST /extensions/<id>/data/download` | body `{ "source": "<source>" }`; download a dataset archive |
| `POST /extensions/<id>/data/delete` | body `{ "source": "<source>" }`; delete a downloaded dataset |

All return `{ "ok": true, ... }` on success and `{ "ok": false, "error": ... }`
with the noted status on failure (`veritate_mri/routes/extensions_routes.py:22`).

## install / uninstall flow

1. The marketplace UI reads `GET /extensions/catalog` and shows each entry with its
   `installed` state.
2. Install posts `{id}` to `POST /extensions/install`. The registry copies
   `canonical/<id>` into `installed/<id>` (`extensions/registry.py:134`).
3. Uninstall posts `{id}` to `POST /extensions/uninstall`. The registry removes
   `installed/<id>` (`extensions/registry.py:145`).

## per-extension supplemental data

An extension may ship a `data_catalog.json` declaring large optional datasets (see
[manifest.md](manifest.md)). The marketplace surfaces these per-extension and lets
the operator download or remove each one independently of installing the extension
itself.

- **Catalog.** `GET /extensions/<id>/data` returns the extension's datasets, each
  annotated with local presence (`present`, `files`, `size_gb`, `downloadable`).
- **Download.** `POST /extensions/<id>/data/download` with `{source}` fetches the
  dataset's archive into `extensions/installed/<id>/data/extension_data/<source>`.
- **Delete.** `POST /extensions/<id>/data/delete` with `{source}` removes the local
  copy to reclaim disk.

Storage lives at `extensions/installed/<id>/data/extension_data/<source>`: a
disposable cache, gitignored (`extensions/installed/` in `.gitignore`). It sits
under `installed/` so it is never copied on install and survives uninstall
(`extensions/registry.py:145`). A dataset dir may itself be a **symlink** to an
external drive; in that case delete only **unlinks** it, leaving the archive intact
at its real path (`extensions/data.py:108`).

The mechanism is generic platform code (`extensions/data.py`): it reports presence,
downloads, and deletes for any extension. The **catalog** (which datasets exist and
where they are hosted) is owned by each extension via its `data_catalog.json`.
Request/response detail is in [../api/rest_api.md](../api/rest_api.md).

## restart to activate

Installs and uninstalls change disk **immediately**, but routes and pages are
mounted only at server start, in `register_all` (`veritate_mri/app.py:158`). So:

- A freshly installed extension's page and routes appear on the **next server
  start**, not on the install request.
- An uninstalled extension's routes stay live until the next restart (its files are
  gone, but the already-registered route objects remain in the running process).

Surface this to the operator: an install/uninstall is staged, and a restart
activates it.

## the extensions settings flag

The settings flag (`extensions`, formerly `experimental`) enables the extensions
nav entry and the marketplace UI in the dashboard. It is a boolean read/written via
`GET`/`POST /settings` ([../api/rest_api.md](../api/rest_api.md)). It is a
visibility switch for the UI surface, not an isolation boundary: the
`/extensions/*` routes register regardless of the flag, and an installed
extension's own routes register at startup regardless of the flag.

## see also

- [entry_point.md](entry_point.md) — what registration does at startup.
- [manifest.md](manifest.md) — the manifest fields the catalog mirrors.
- [authoring.md](authoring.md) — building an extension to install.
- [../api/rest_api.md](../api/rest_api.md) — endpoint request/response detail.
