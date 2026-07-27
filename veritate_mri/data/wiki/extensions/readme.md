---
title: extensions overview
date: 2026-07-27
tags: [extensions, overview]
summary: What an extension is, what it may touch, and where each part of the contract is written down.
---

# extensions overview

An extension is a self-contained page that runs beside the Veritate dashboard. It is one directory holding a manifest, an optional entry point, a page, and optional server modules. It reaches the platform only through the documented HTTP API and, on the server side, the model-loading surface. It never imports platform internals.

An extension is not the same thing as a trainer plugin. A plugin is a trainer and follows the plugin contract in the internal developer documentation. The two terms are not interchangeable.

## the four entries

| entry | covers |
|---|---|
| authoring | directory layout, the page, isolation rules, what exists today |
| manifest | every key the registry reads from `manifest.json`, with defaults |
| entry point | how `register(app)` is discovered and called, and the startup sequence |
| marketplace | the catalog, install and uninstall, and what is live versus what needs a restart |

## the contract

The platform HTTP API is the only stable surface an extension codes against. It is documented in the `api` category of this wiki. Anything not in that reference is not part of the contract and can move without notice.

## current state

`extensions/canonical/` and `extensions/installed/` are both empty and `extensions/catalog.json` lists no entries. The registry, the marketplace routes, the install and uninstall flow, and the live disable gate are all present and tested. Nothing ships in the catalog yet.
