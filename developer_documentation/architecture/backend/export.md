# export

## What it is

Converts PyTorch `.pt` checkpoints into the engine's `.bin` format. Lives at [veritate_mri/training/export.py](../../../veritate_mri/training/export.py).

## How it works

`export_checkpoint(name, step)` and variants (`export_checkpoint_ternary`, `export_checkpoint_int4`) load a checkpoint from `models/<name>/checkpoints/step_<N>.pt`, walk the state_dict in canonical order, and emit a binary that the C engine at [veritate_engine/v1/](../../../veritate_engine/v1/) can `mmap` and run.

The `.bin` format is described under [documentation/engine/](../../engine/) and supports versions v3 through v12 with progressively more features (MoE in v11, MTP in v12). See [veritate_engine.md](veritate_engine.md).

## Dependencies

- [veritate_core/model.py](../../../veritate_core/model.py) — defines the source state_dict shape.
- [readers/checkpoints.py](../../../veritate_mri/readers/checkpoints.py) — finds the `.pt` file.
- [readers/bin.py](../../../veritate_mri/readers/bin.py) — reads metadata of exported `.bin` files.
- The C engine kernels — the export format must match what they load.

## Pitfalls

- Exporting INT4 or ternary from an fp32-trained model is lossy. QAT-trained models export cleanly because the quantization basis is already learned.
- Output is one `.bin` per model; multi-checkpoint exports require running the function per step. The convention is to export only the final checkpoint unless multi-step evaluation is needed.
