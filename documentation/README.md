# Veritate documentation

Central reference for the codebase. Read [claude_preflight.md](../claude_preflight.md)
at the repo root before working in this codebase.

Docs serve two audiences. Pick the lane that matches what you are doing.

## A. building Veritate (internal / platform developer)

You are changing platform code: the Flask app, training pipeline, engine, kernels,
dashboard internals, or a trainer. Start here.

- **[architecture/](architecture/)** — how the system is wired. One file per
  component, split into [frontend/](architecture/frontend/) (dashboard tabs, panels,
  standalone modules) and [backend/](architecture/backend/) (Flask app, routes,
  training, readers, runtime, engine).
- **[agents/](agents/)** — agent-facing rules (coding_roe, claude_merge, agent_roe).
  Required reading.
- **[plugins/](plugins/)** — the internal **plugin = trainer** contract
  (`veritate_core.plugin` surface). Note: "plugin" is the trainer concept here, NOT
  a user add-on (see the external lane below).
- **[trainers/](trainers/)** — per-trainer specs and the authoritative trainer API.
- **[training/](training/)** — model storage layout and training conventions.
- **[corpus/](corpus/)** — corpus formats, framing, and the price-series corpora.
- **[engine/](engine/)** — C engine internals.
- **[kernels/](kernels/)** — quantized matmul kernels (INT8, INT4, ternary).
- **[hooks/](hooks/)** — checkpoint dump artifacts (probe.json, lens.npz, etc.).
- **[platform/](platform/)** — hardware tiers and platform-specific notes.
- **[addons/](addons/)** — inference addons.
- **[market/](market/)** — the Market LLM subsystem reference.

## B. building on Veritate (external extension author)

You are building a self-contained project that runs beside the dashboard, calls the
platform API, or loads a model the user trained — without modifying platform code.
Start here.

- **[extensions/authoring.md](extensions/authoring.md)** — how to structure a
  self-contained extension, where files go, how it surfaces UI, how it stays
  isolated, how it calls the API and loads a trained model, and what is **not yet
  implemented** versus available today. Read this first.

Supporting reads (consumed, not modified, by an extension):

- [architecture/backend/routes.md](architecture/backend/routes.md) — the full HTTP API surface.
- [architecture/frontend/standalone_modules.md](architecture/frontend/standalone_modules.md) — the in-dashboard panel/tab pattern.
- [architecture/frontend/tab_system.md](architecture/frontend/tab_system.md) — how a tab is registered today.
- [architecture/backend/settings.md](architecture/backend/settings.md) — the experimental gate.

**Naming:** "plugin" (lane A) is the internal trainer contract. The external
add-on concept (lane B) is called an **extension**. Do not conflate the two.

## Conventions

- One file per component. Each file covers what it is, how it works, dependencies,
  and any pitfalls.
- File names are lowercase, snake_case, `.md`.
- File:line references use markdown links: `[file.py:42](../path/to/file.py#L42)`.
  Relative paths from the doc.
- Voice: developer-to-developer. No "the user", no "I/we/my", no narrative of how it
  came to be. State what it is and how to use it.
- Keep each file short and digestible. If a component is large, split it into
  multiple files, not one giant file.

## When you change a component

Update its doc in the same change. If the component is new, add a new file. If the
component is removed, delete the file. The doc reflects current state, not history;
commit messages and git log carry the history. When platform code that the external
API or extension contract depends on changes, update
[extensions/authoring.md](extensions/authoring.md) in the same change.

## When you can't find a doc

If the component you're working on doesn't have a doc yet, write one before finishing
the change. Use an existing file as a template.
