# export

## What it is

Converts PyTorch `.pt` checkpoints into the engine's `.bin` format. Lives at [veritate_mri/training/export.py](../../../veritate_mri/training/export.py).

## How it works

Two public entry points, `export_checkpoint(name, step, dtype=None)` and `export_checkpoint_ternary(name, step, out_path=None)`, load a checkpoint from `models/<name>/checkpoints/step_<N>.pt`, walk the state_dict in canonical order, and emit a binary that the C engine at [veritate_engine/v1/](../../../veritate_engine/v1/) can `mmap` and run.

`export_checkpoint` reads `training_args.trunk` and dispatches: `dense` exports through the canonical v9/v11/v12 int8 layouts; an MTP-head checkpoint routes to the private `_export_checkpoint_mtp`; `hybrid` (with `state_rule=gla`) exports through the v13 fp32/fp16 writer (`_export_checkpoint_hybrid`, `dtype` kwarg, default fp16); every other trunk (patched/looped/recurrent) is refused with a clear `ValueError`. The int8 byte layout assumes the canonical block order, so a research-trunk state_dict walked in canonical order produces a structurally garbage bin that loads without error and generates noise. `export_checkpoint_ternary` carries the same dense-only guard. Refused trunks serve through the PyTorch brain instead ([inference_brain.md](inference_brain.md)).

The `.bin` format is described under [developer_documentation/engine/](../../engine/) and supports versions v3 through v13 with progressively more features (MoE in v11, MTP in v12, hybrid trunk in v13, spec at [engine_v13_hybrid.md](../../engine/engine_v13_hybrid.md)). See [veritate_engine.md](veritate_engine.md).

## Dependencies

- [veritate_core/model.py](../../../veritate_core/model.py): defines the source state_dict shape.
- [readers/checkpoints.py](../../../veritate_mri/readers/checkpoints.py): finds the `.pt` file.
- [readers/bin.py](../../../veritate_mri/readers/bin.py): reads metadata of exported `.bin` files.
- The C engine kernels: the export format must match what they load.

## Pitfalls

- Exporting INT4 or ternary from an fp32-trained model is lossy. QAT-trained models export cleanly because the quantization basis is already learned. v13 hybrid exports are fp32/fp16 and need no QAT (fp16 measured byte-parity-clean and bpb-identical to 4 decimals on chat80m_80m).
- The trunk guard exists because the failure mode is silent: a non-canonical checkpoint walked through the int8 writer yields a bin that loads cleanly and generates noise, with no error to grep for.
- Output is one `.bin` per model; multi-checkpoint exports require running the function per step. The convention is to export only the final checkpoint unless multi-step evaluation is needed.
