# native_trainer

## What it is

Low-level training loop helpers at [veritate_mri/training/native_trainer.py](../../../veritate_mri/training/native_trainer.py). Used by plugins that want a standard loop with QAT, gradient accumulation, and corpus mixing without re-implementing them.

## How it works

Imports `from veritate_core import qat as veritate_qat` ([line 42](../../../veritate_mri/training/native_trainer.py#L42)) for the QAT switchovers.

Exposes loop building blocks: corpus sampler integration, LR scheduling, gradient accumulation, optimizer step, checkpoint trigger. Plugins compose these — they don't have to use the whole package.

Most existing plugins (`distill_teacher`, `multimind_m3`) hand-roll their own loops in `plugin.py` and don't call this module. Newer trainers can opt in for the common parts.

## Dependencies

- [veritate_core/qat.py](../../../veritate_core/qat.py) — QAT mode switching.
- [training/save.py](save.md) — CSV append + checkpoint save.
- [veritate_core/plugin/multicorpus.py](../../../veritate_core/plugin/multicorpus.py) — mixed corpus loading.

## Pitfalls

- Not a framework. Pieces are composable; don't try to pass a config blob and expect a full training run.
- QAT mode transitions (INT8 → ternary) are caller-driven via `veritate_qat.set_quant_mode(...)`; this module doesn't manage staging.
