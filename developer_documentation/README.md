# Veritate platform internals

Internal reference for developing the Veritate platform: the Flask app, training
pipeline, engine, kernels, dashboard internals, and trainers. Public extension docs
(REST API + extension authoring) live in [documentation/](../documentation/).

One file per component: what it is, how it works (with file:line refs), dependencies,
pitfalls. File names are lowercase, snake_case, `.md`. Voice is developer-to-developer:
state what a component is and how it works, not how it came to be.

## Layout

- **[architecture/frontend/](architecture/frontend/)** — one file per dashboard tab,
  panel, or standalone module (rendering, data flow, state, HUD, tutorial, market page).
- **[architecture/backend/](architecture/backend/)** — one file per Flask app, route
  module, runtime, training, engine, or inference component.
- **[hooks/](hooks/)** — checkpoint dump artifacts (probe.json, lens.npz, ...) and the
  brain-hooks contract.
- **[kernels/](kernels/)** — quantized matmul kernels (INT8, INT4, ternary), MoE, QAT.
- **[engine/](engine/)** — C engine internals and on-disk format versions.
- **[platform/](platform/)** — hardware tiers, memory planner/executor, paged
  optimizer, Metal GPU, and bench notes.
- **[plugins/](plugins/)** — the internal plugin (= trainer) contract over the
  `veritate_core.plugin` surface.
- **[trainers/](trainers/)** — the authoritative per-trainer API and contract.
- **[training/](training/)** — model storage layout and run-launch conventions.
- **[corpus/](corpus/)** — corpus formats, framing, and the price-series corpora.
- **[addons/](addons/)** — inference decode addons and the C engine port contract.
- **[market/](market/)** — the Market LLM subsystem reference.
- **[agents/](agents/)** — agent rule files (coding_roe, claude_merge, agent_roe).

See also [about.md](about.md).
