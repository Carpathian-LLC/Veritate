# Architecture

Per-component documentation, split into frontend and backend.

- **[frontend/](frontend/)** — the dashboard (vanilla JS, served by Flask at `veritate_mri/web/`). One file per tab, panel, or standalone module.
- **[backend/](backend/)** — the Flask app, training pipeline, readers, runtime, and engine. One file per module or subsystem.

Each file follows the same shape:

```
# component name

## What it is
One or two sentences.

## How it works
Implementation summary with file:line references.

## Dependencies
What it imports. What imports it.

## Pitfalls
Optional: surprising behavior or constraints.

## See also
Links to related component docs.
```

Pick the file that matches the thing you're working on; if a doc doesn't exist for a component you touch, write one before finishing the change.
