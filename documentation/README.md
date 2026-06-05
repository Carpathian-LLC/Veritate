# Veritate documentation

Central reference for the codebase. Every shipped component is documented here. Read [claude_preflight.md](../claude_preflight.md) at the repo root before working in this codebase.

## Layout

- **[architecture/](architecture/)** — how the system is wired. One file per component, split into `frontend/` and `backend/`.
- **[agents/](agents/)** — agent-facing rules (coding_roe, claude_merge, agent_roe). Required reading.
- **[addons/](addons/)** — inference addons.
- **[corpus/](corpus/)** — corpus formats and loading.
- **[engine/](engine/)** — C engine internals.
- **[hooks/](hooks/)** — checkpoint dump artifacts (probe.json, lens.npz, classroom.json, etc.).
- **[kernels/](kernels/)** — quantized matmul kernels (INT8, INT4, ternary).
- **[multimind/](multimind/)** — multimind variants and experiments.
- **[platform/](platform/)** — hardware tiers and platform-specific notes.
- **[plugins/](plugins/)** — trainer plugin contract.
- **[trainers/](trainers/)** — per-trainer specs.
- **[training/](training/)** — model storage layout and training conventions.

## Conventions

- One file per component. Each file covers what it is, how it works, dependencies, and any pitfalls.
- File names are lowercase, snake_case, `.md`.
- File:line references use markdown links: `[file.py:42](../path/to/file.py#L42)`. Relative paths from the doc.
- Voice: developer-to-developer. No "the user", no "I/we/my", no narrative of how it came to be. State what it is and how to use it.
- Keep each file short and digestible. If a component is large, split it into multiple files, not one giant file.

## When you change a component

Update its doc in the same change. If the component is new, add a new file. If the component is removed, delete the file. The doc reflects current state, not history; commit messages and git log carry the history.

## When you can't find a doc

If the component you're working on doesn't have a doc yet, write one before finishing the change. Use an existing file as a template.
