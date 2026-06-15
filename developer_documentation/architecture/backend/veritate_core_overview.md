# veritate_core overview

## What it is

The training-side library at [veritate_core/](../../../veritate_core/). Defines the model class, quantization-aware training, checkpoint loading, and the plugin contract that trainers import from.

## Modules

| File                                                                             | Purpose                                                          |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [model.py](../../../veritate_core/model.py)                                      | Canonical byte-level `Veritate` class (vocab=256, pre-norm, RMSNorm, combined qkv, tied lm_head). Forward returns `(logits, loss)`. |
| [model_rope.py](../../../veritate_core/model_rope.py)                            | RoPE variant of the same architecture                            |
| [qat.py](../../../veritate_core/qat.py)                                          | Quantization-aware training: INT8, INT4, ternary fake-quant      |
| [qat_triton.py](../../../veritate_core/qat_triton.py)                            | Triton kernels for QAT (CUDA only)                               |
| [load.py](../../../veritate_core/load.py)                                        | Checkpoint loading utilities                                     |
| [core_plugins.py](../../../veritate_core/core_plugins.py)                        | Built-in plugins exported via `veritate_core/plugin/`            |
| [plugin/hardware.py](../../../veritate_core/plugin/hardware.py)                  | Device detection (CPU/MPS/CUDA), physical-core count             |
| [plugin/multicorpus.py](../../../veritate_core/plugin/multicorpus.py)            | Mixed-corpus loader (`"a+b+c"` or `"a:0.5,b:0.3,c:0.2"`)         |
| [plugin/oom_recovery.py](../../../veritate_core/plugin/oom_recovery.py)          | Catch and recover from CUDA OOM during a step                    |

## Public API surface

Imports trainers use:

```python
from veritate_core.model import Veritate           # canonical class
from veritate_core import qat as vqat              # qat ops
from veritate_core.plugin import hardware          # device detection
from veritate_core.plugin import multicorpus       # data loaders
```

The legacy `from veritate.X import ...` aliases resolve to these via the [veritate_shim.md](veritate_shim.md).

## Forward contract

`Veritate.forward(tokens, targets=None)` returns `(logits, loss)`. When `targets` is None, `loss` is None. Callers should not bypass this signature — the QAT switchovers and the MTP byte-0 transform depend on it.

## QAT

`qat.set_qat(model, True)` flips fake-quant on for every QAT-aware module. `qat.set_quant_mode(model, mode)` switches between `QUANT_MODE_INT8`, `QUANT_MODE_INT4`, `QUANT_MODE_TERNARY`. The state_dict shape is unchanged — quantization is training-time only; export to `.bin` is a separate conversion step (see [export.md](export.md)).

## Pitfalls

- Vocab is hard-coded to 256 (byte-level). Any model class that overrides this breaks the engine and every reader.
- Adding a new model variant must follow the `hook_spec()` contract so [checkpoint_probe.py](checkpoint_probe.md) can walk it.
- QAT is opt-in. Trainers must call `set_qat(model, True)` before the loop; otherwise fake-quant nodes are inert.
