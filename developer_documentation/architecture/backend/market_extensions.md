# market extensions

Downloadable add-on market **datasets** (stocks, forex, the broader crypto archives, ...),
managed from the Extensions section of the settings corpus-library card. Mirrors the
corpus-library pattern but over raw CSV dataset dirs instead of trainer `.bin` corpora.

Distinct from the external-author **extension** concept in
[documentation/extensions/authoring.md](../../../documentation/extensions/authoring.md) (downloadable code /
plugin add-ons): same "downloadable add-on" framing, different artifact. This doc is data only.

## what it is

- Backend: [extensions.py](../../../veritate_mri/market/extensions.py) — `catalog()`,
  `download(source)`, `delete(source)`.
- Registry: [extensions_catalog.json](../../../veritate_mri/market/extensions_catalog.json) —
  one entry per dataset: `source`, `label`, `description`, `url` (hosted archive; `null` =
  placeholder, not hosted yet), `approx_gb`, `schema`.
- Routes (in [market_routes.py](market_routes.md)): `GET /market/extensions/catalog`,
  `POST /market/extensions/download`, `POST /market/extensions/delete`.
- UI: an Extensions block in the corpus-library card + an `#extensionsModal` popup
  (download / delete per dataset), wired in `web/index.js` (`_ext*` functions).

## how it works

- Datasets live at `external_data/extension_data/<source>/` — a gitignored, disposable
  cache. Active training/serving sources (`crypto_of`, `funding`, `sentiment`, `live`)
  stay at `external_data/<source>/` and are NOT extensions.
- `data.source_dir(source)` ([data.py](../../../veritate_mri/market/data.py)) resolves a
  source to whichever of `external_data/<source>` or `external_data/extension_data/<source>`
  exists, so moving a dataset into `extension_data/` is transparent to the builder and the
  serving layer.
- `catalog()` enriches each registry entry with live local status: `present`, `files`,
  `size_gb`, `downloadable` (has a url).
- `delete(source)` reclaims disk. A real local dir is `rmtree`'d (returns `reclaimed_gb`);
  a symlinked dir (a dataset parked on an external drive) has only its link removed
  (`unlinked: true`), leaving the underlying archive intact — a dashboard click never
  wipes an external store.
- `download(source)` pulls the entry's `url` into `extension_data/<source>/`. Entries with
  a `null` url are placeholders and return a "not hosted yet" message until a catalog url +
  fetch implementation land (the S3 host).

## dependencies

- `market.data.EXTENSION_DIR` / `source_dir`.
- `external_data/extension_data/` (gitignored, local cache).

## pitfalls

- A placeholder dataset (null url) is NOT re-downloadable, so deleting it is permanent; the
  UI confirms with a stronger warning in that case.
- `_ext_dir` validates the normalized (not realpath) path so `..` traversal in `source` is
  rejected while a symlinked dataset dir still resolves.
- New routes require a server restart to register; do not restart while a dashboard training
  run is active.
