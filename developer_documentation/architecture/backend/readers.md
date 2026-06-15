# Readers (data layer)

## What it is

The data-access layer at [veritate_mri/readers/](../../../veritate_mri/readers/). Every disk read goes through a reader module. Routes call readers; readers call `os` and `open`. Routes never call `os.listdir`, `os.path.join`, or `open` directly.

## Module inventory

| Module                                                                            | Owns                                                       |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [paths.py](../../../veritate_mri/readers/paths.py)                                | Path resolution (REPO_ROOT, models/, data/, trainers/)     |
| [models.py](../../../veritate_mri/readers/models.py)                              | Model listing, name validation, existence checks           |
| [config.py](../../../veritate_mri/readers/config.py)                              | `models/<name>/config.json` load                           |
| [checkpoints.py](../../../veritate_mri/readers/checkpoints.py)                    | List of `step_<N>.pt` files                                |
| [train_csv.py](../../../veritate_mri/readers/train_csv.py)                        | Parse `models/<name>/train.csv` per the canonical schema   |
| [capabilities.py](../../../veritate_mri/readers/capabilities.py)                  | Capability tiers (reading, math, reasoning) per model      |
| [corpus.py](../../../veritate_mri/readers/corpus.py)                              | List corpus stems; per-stem usage across models            |
| [engine.py](../../../veritate_mri/readers/engine.py)                              | Query C engine state                                       |
| [hooks.py](../../../veritate_mri/readers/hooks.py)                                | Load `hooks/step_<N>/` artifacts (probe.json, lens.npz...) |
| [bin.py](../../../veritate_mri/readers/bin.py)                                    | `.bin` export metadata                                     |
| [trainers.py](../../../veritate_mri/readers/trainers.py)                          | Plugin manifests; available plugin listing                 |

## The contract

- Functions return parsed Python data (dicts, lists), not raw bytes or file handles.
- Missing files return `None` or empty containers; readers never raise for "file not found" on optional artifacts.
- Cache is per-call by default. Persistent caching belongs in a higher layer (e.g., a route can cache for an interval).

## Dependencies

- Path layout owned by [paths.py](../../../veritate_mri/readers/paths.py). Add new path conventions there, not at call sites.
- Schema validation in [models.py](../../../veritate_mri/readers/models.py) for model names; trainers must produce names that pass `is_valid_name`.

## Pitfalls

- Don't add `os.path.join` calls inside routes. Refactor into a reader if a new path is needed.
- `train_csv.py` enforces the canonical CSV header. Trainers that add or rename columns break every reader that consumes it; coordinate before changing the schema.
- File mtimes are surfaced for ordering (newest checkpoint, last-modified run). Time-skewed clocks across NFS or virtualized environments can produce non-monotonic results — readers do not paper over this.
