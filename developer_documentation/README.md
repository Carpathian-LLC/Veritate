# Veritate platform internals

Internal reference for developing the Veritate platform: the Flask app, training
pipeline, engine, kernels, dashboard internals, and trainers. Public docs (REST API +
extension authoring) live in the in-app wiki at `veritate_mri/data/wiki/`, served by the
dashboard wiki tab.

One file per component: what it is, how it works (with file:line refs), dependencies,
pitfalls. File names are lowercase, snake_case, `.md`. Voice is developer-to-developer:
state what a component is and how it works, not how it came to be.

## Layout

- **[architecture/frontend/](architecture/frontend/)**: one file per dashboard tab,
  panel, or standalone module (rendering, data flow, state, HUD, tutorial).
- **[architecture/backend/](architecture/backend/)**: one file per Flask app, route
  module, runtime, training, engine, or inference component.
- **[addons/](addons/)**: inference decode addons and the C engine port contract.
- **[corpus/](corpus/)**: corpus formats, framing, authoring, and the library ladder.
- **[engine/](engine/)**: C engine internals and on-disk format versions.
- **[hooks/](hooks/)**: checkpoint dump artifacts (probe.json, lens.npz, ...) and the
  brain-hooks contract.
- **[kernels/](kernels/)**: quantized matmul kernels (INT8, INT4, ternary), MoE, QAT,
  the Metal compute path, and engine binary versioning.
- **[platform/](platform/)**: hardware tiers, memory planner/executor, paged optimizer,
  model variants, and bench notes.
- **[plugins/](plugins/)**: the internal plugin (= trainer) contract over the
  `veritate_core.plugin` surface.
- **[research/](research/)**: research write-ups, measured results, and negative results.
- **[trainers/](trainers/)**: the authoritative per-trainer API and contract.
- **[training/](training/)**: model storage layout, run-launch conventions, the settings
  index, and corpus sources.
- **[agents/](agents/)**: agent rule files (coding_roe, claude_merge, agent_roe).

See also [about.md](about.md).
