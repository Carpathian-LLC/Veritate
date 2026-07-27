---
title: Build 1: Veritate 1.0.0
date: 2026-07-27
tags: [build, release]
summary: The 1.0.0 launch. Every component starts at v1.0.0 and nothing on disk needs rebuilding.
---

# Build 1: Veritate 1.0.0

The first released build. Train a byte-level model on this box, watch every layer of it while it learns, chat with it, and serve it to any OpenAI-compatible client.

Nothing to do. There is no earlier version of Veritate, so no models, corpora, settings, or exported files need rebuilding or migrating.

The wiki tab is now the home of the public documentation: the REST API every client and extension calls, how to build and publish an extension, one page per training setting, and these build notes.

| component | version | covers |
|---|---|---|
| build | 1 | the number to quote in a bug report |
| engine | v1.0.0 | the C inference engine and its per-architecture kernels |
| mri | v1.0.0 | the dashboard, routes, training, and inference |
| format | v1.0.0 | on-disk layouts: model files, settings, run configs, hook artifacts |
| trainers | v1.0.0 | the contract trainers are built against |
