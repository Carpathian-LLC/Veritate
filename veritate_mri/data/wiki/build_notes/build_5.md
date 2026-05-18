---
title: "Build 5: experimental → dev merge, v11 unified format, tests/ folder"
date: 2026-05-08
tags: [build, format, merge, tests]
summary: experimental folded into dev. v10 retired (assigned twice on different branches) and superseded by v11 unified format with quant_mode + n_experts + router_topk header fields. New tests/ folder with 25 platform regression tests. Dashboard now warns about stale .bin files.

---

## versions

- build: 5
- engine: v2.2.0
- mri: v0.2.0
- format: v0.3.0
- plugins: v0.1.1

## what changed

- engine `.bin` format **v10 retired**, **v11 (`VERITATE_MODEL_VERSION_QAT`) introduced**.
- `tests/` folder added at the repo root (`pytest tests/`).
- dashboard topbar shows a warning when any `.bin` in `models/` is stale.

## what the user has to do

Any model trained on the old `experimental` branch with a v10 ternary `.bin` will fail to load. **Re-export from the most recent `.pt` checkpoint:**

```python
from veritate_mri import export
export.export_checkpoint_ternary("<model_name>", <step>)
```

Or for an INT8 / non-MoE model:

```python
export.export_checkpoint("<model_name>", <step>)
```

The dashboard's topbar banner lists which models need re-export.

## test infrastructure

```
pip install -r requirements.txt   # pytest now bundled
pytest tests/                     # ~30s, runs full suite
```
