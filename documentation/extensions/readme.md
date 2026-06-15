# extensions

The extensions system lets a self-contained page run beside the Veritate
dashboard. An extension is its own directory: a manifest, an entry point, a page,
and optional server modules. It talks to the platform **only** through the
documented HTTP API ([../api/rest_api.md](../api/rest_api.md)) and the server-side
model-loading surface. It never imports platform internals.

This is distinct from the internal trainer **plugin** (a trainer; see
[../../developer_documentation/plugins/contract.md](../../developer_documentation/plugins/contract.md)).
The two terms are not interchangeable: plugin = trainer contract, extension =
self-contained add-on page.

## documents

- [authoring.md](authoring.md) — build an extension: directory layout, the page,
  isolation rules, what exists vs what is future.
- [manifest.md](manifest.md) — `manifest.json` schema: every field, type, and
  required/optional status, with examples.
- [entry_point.md](entry_point.md) — how `register(app)` is discovered and called,
  how the page route mounts, the `api_prefix` convention, the startup lifecycle.
- [marketplace.md](marketplace.md) — the catalog, install/uninstall endpoints,
  canonical vs user-created extensions, and the restart-to-activate lifecycle.

## the contract

The platform HTTP API is the only stable surface an extension codes against:
[../api/rest_api.md](../api/rest_api.md). Anything not in that reference is not part
of the contract.
