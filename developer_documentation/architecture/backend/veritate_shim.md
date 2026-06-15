# veritate (backwards-compat shim)

## What it is

A thin compatibility package at [veritate/](../../../veritate/) that re-exports the renamed `veritate_core` and `veritate_mri` modules under the legacy `veritate.X` paths. It exists so any code still written against the old `from veritate.X import ...` paths keeps resolving.

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

## Consumers

The canonical trainers in [trainers/](../../../trainers/) import the new paths directly (`from veritate_core.plugin import save, paths, model, qat, multicorpus`), not the shim. The shim has no current in-repo consumers; it remains only as a safety net for legacy `from veritate.X import ...` code.

## Coexistence with `veritate.py`

The launcher script `veritate.py` lives at the repo root next to `veritate/`. Python's finder prefers the regular package (directory) over a same-name module (file) for imports, so `import veritate` resolves to the shim. The launcher is invoked as `python veritate.py` (script execution, not import) and is unaffected.

## Dependencies

- [veritate_core/](../../../veritate_core/) — model + qat.
- [veritate_mri/training/save.py](../../../veritate_mri/training/save.py) and [veritate_mri/readers/paths.py](../../../veritate_mri/readers/paths.py) — runtime modules.

## Pitfalls

- The shim is for legacy compatibility, not new code. Write new code against the canonical paths (`from veritate_core.model import Veritate`, `from veritate_core.plugin import save`).
- It has no current consumers; treat it as removable once verified no out-of-repo code imports `veritate.X`.
