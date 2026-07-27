---
title: the wiki
date: 2026-07-27
tags: [index]
summary: What each wiki category holds and where to start.
---

# the wiki

The documentation that ships inside Veritate. The dashboard wiki tab serves it from `veritate_mri/data/wiki/<category>/<slug>.md`, and any entry also renders as a standalone page at `/wiki/<category>/<slug>/page`.

| category | holds |
|---|---|
| `api` | the REST contract: every platform route, and the outward-facing surface with its key gate |
| `extensions` | how to build, install, and publish an extension |
| `settings` | one entry per training setting, linked from the training form |
| `concepts` | plain-language background for anyone new to the platform |
| `build_notes` | one entry per build: what changed and what to do about it |

## where to start

New to the platform: read `concepts`, then open the training tab. Every setting on that form links straight to its own entry in `settings`.

Integrating this box from another service: `api/external_api`.

Building something that runs beside the dashboard: `extensions/authoring`, then `api/internal_api` for the routes it calls.

Internal platform and component contracts are not here. They live under `developer_documentation/` in the repository.
