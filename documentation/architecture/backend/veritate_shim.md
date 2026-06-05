# veritate (backwards-compat shim)

## What it is

A thin compatibility package at [veritate/](../../../veritate/) that re-exports the renamed `veritate_core` and `veritate_mri` modules under the legacy `veritate.X` paths. Without it, every trainer plugin in [trainers/](../../../trainers/) fails at import.

## How it works

- [veritate/__init__.py](../../../veritate/__init__.py) — adds repo root and `veritate_mri/` to `sys.path`, then re-exports:
  - `veritate.model` → `veritate_core.model`
  - `veritate.qat` → `veritate_core.qat`
  
- [veritate/plugin/__init__.py](../../../veritate/plugin/__init__.py) — re-exports:
  - `veritate.plugin.save` → `training.save` (which is `veritate_mri/training/save.py`)
  - `veritate.plugin.paths` → `readers.paths` (which is `veritate_mri/readers/paths.py`)
  - `veritate.plugin.model` → `veritate_core.model`
  - `veritate.plugin.qat` → `veritate_core.qat`

Re-exports use both attribute binding (`from training import save`) and `sys.modules` aliasing so `from veritate.plugin.save import X` works in addition to `from veritate.plugin import save`.

## Trainers that depend on it

- `trainers/distill_teacher/plugin.py`
- `trainers/multimind_m1/plugin.py` + `m1_model.py`
- `trainers/multimind_m3/plugin.py` + `m3_model.py`
- `trainers/multimind_mega/plugin.py` + `mega_model.py`
- `trainers/example_plugin/plugin.py`

All of these import `from veritate.X import ...`. The shim resolves them without per-trainer edits.

## Coexistence with `veritate.py`

The launcher script `veritate.py` lives at the repo root next to `veritate/`. Python's finder prefers the regular package (directory) over a same-name module (file) for imports, so `import veritate` resolves to the shim. The launcher is invoked as `python veritate.py` (script execution, not import) and is unaffected.

## Dependencies

- [veritate_core/](../../../veritate_core/) — model + qat.
- [veritate_mri/training/save.py](../../../veritate_mri/training/save.py) and [veritate_mri/readers/paths.py](../../../veritate_mri/readers/paths.py) — runtime modules.

## Pitfalls

- The shim is for legacy compatibility, not new code. Write new code against the canonical paths (`from veritate_core.model import Veritate`, `from training import save`).
- Removing the shim breaks every existing trainer. Migrating each trainer to canonical imports is a parallel cleanup task that can happen later.
