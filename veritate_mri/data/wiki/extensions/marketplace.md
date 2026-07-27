---
title: extension marketplace
date: 2026-07-27
tags: [extensions, marketplace]
summary: The catalog, the install and uninstall flow, and exactly what takes effect without a restart.
---

# extension marketplace

The marketplace lists installable extensions, installs them from the bundled source, and removes them. `extensions/registry.py` does the disk work and `veritate_mri/routes/extensions_routes.py` exposes it over HTTP. The nav entry and marketplace UI appear when the `extensions` setting is on.

## the catalog

`extensions/catalog.json` is the listing:

```json
{
  "extensions": [
    {
      "id": "hello",
      "name": "Hello",
      "version": "0.1.0",
      "author": "veritate",
      "description": "A static page that calls the platform API.",
      "experimental": false,
      "builtin": true
    }
  ]
}
```

`load_catalog()` reads the file and stamps each entry with an `installed` boolean computed from whether that id is currently discovered. The catalog is a listing only: it holds no extension code, and an entry can exist with no source behind it, in which case install returns `404`.

The shipped catalog is currently empty.

## canonical and user-authored

- **Canonical** extensions ship bundled under `extensions/canonical/<id>/`. That directory is the install source: install copies it into `extensions/installed/<id>/`. A canonical extension is discovered and active without an explicit install, because discovery scans both roots. Installing one places an editable copy in `installed/` that then takes precedence over the bundled original.
- **User-authored** extensions are placed directly under `extensions/installed/<id>/`. They are discovered and registered exactly like an installed canonical extension and need no catalog entry to run.

There is no remote download of extension code. Install resolves an id to `canonical/<id>` and copies it; with no canonical source and no existing `installed/<id>` it raises and the route returns `404`. Extension datasets are a separate mechanism and do download from a hosted URL.

## routes

| method and path | purpose |
|---|---|
| `GET /extensions` | installed extensions: `id`, `name`, `version`, `nav_label`, `route`, `experimental` |
| `GET /extensions/catalog` | the catalog, each entry annotated with `installed` |
| `POST /extensions/install` | body `{"id": "<id>"}`. Clears the disabled flag and copies `canonical/<id>` into `installed/<id>`. `400` missing id, `404` no source |
| `POST /extensions/uninstall` | body `{"id": "<id>"}`. Disables the id and removes `installed/<id>` code, keeping `data/`. `400` missing id |
| `GET /extensions/<id>/data` | the extension's dataset catalog, annotated with local presence |
| `POST /extensions/<id>/data/download` | body `{"source": "<source>"}`. Fetch and extract a dataset archive |
| `POST /extensions/<id>/data/delete` | body `{"source": "<source>"}`. Remove a downloaded dataset |

All return `{"ok": true, ...}` on success and `{"ok": false, "error": ...}` with the noted status on failure. Request and response detail is in the platform API reference.

## install and uninstall

1. The marketplace reads `GET /extensions/catalog` and shows each entry with its `installed` state.
2. Install posts the id. The registry clears any disabled flag and copies `canonical/<id>` into `installed/<id>`.
3. Uninstall posts the id. The registry adds it to `extensions/disabled.json`, so even a bundled canonical extension that cannot be physically deleted is deactivated, then removes the `installed/<id>` code while preserving the `data/` directory. A reinstall clears the flag and restores the code beside the surviving data.

## what is live and what needs a restart

Flask routes cannot be removed from a running app, so uninstall works through a gate rather than through unregistration. `register_all` installs a `before_request` hook that returns `404` for any request whose matched route belongs to a disabled id, using the route-to-extension map it built while registering.

- **Uninstall is immediate.** The gate starts returning `404` on the next request, and the nav and catalog flag update at the same time.
- **Reinstalling an extension that was registered at the last startup is immediate.** The flag clears and the gate stops blocking. No re-registration is needed because the route objects were never removed.
- **An extension that was absent or disabled at the last startup has no registered routes for the gate to unblock.** Its routes and page appear at the next server start. The marketplace says so after installing one.

## supplemental datasets

An extension may ship a `data_catalog.json` declaring large optional datasets. The marketplace surfaces them per extension and downloads or removes each one independently of installing the extension itself.

- **Catalog.** `GET /extensions/<id>/data` returns the datasets annotated with `present`, `files`, `size_gb`, and `downloadable`.
- **Download.** `POST /extensions/<id>/data/download` streams the single hosted archive named by the entry's `url`, a `.tar.gz` or `.zip` with its files at the top level, extracts it into `extensions/installed/<id>/data/extension_data/<source>`, then removes the temporary archive. A `null` url is a placeholder and reports `downloadable: false`.
- **Delete.** `POST /extensions/<id>/data/delete` removes the local copy. A dataset directory that is a symlink to an external drive is only unlinked, leaving the archive intact at its real path.

Storage sits under `installed/` so it is never copied on install and survives an uninstall. The mechanism in `extensions/data.py` is generic platform code; the catalog belongs to each extension.

## the extensions setting

The `extensions` setting enables the nav entry and the marketplace UI. It is a boolean read and written through `GET` and `POST /settings`. It is a visibility switch, not an isolation boundary: the `/extensions/*` routes register regardless of it, and an installed extension's own routes register at startup regardless of it.
