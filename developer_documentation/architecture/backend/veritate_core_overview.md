# veritate_core overview

## What it is

The training-side library at [veritate_core/](../../../veritate_core/). Defines the model class, quantization-aware training, checkpoint loading, and the plugin contract that trainers import from.

## Modules

| File                                                                             | Purpose                                                          |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [model.py](../../../veritate_core/model.py)                                      | Canonical byte-level `Veritate` class (vocab=256, pre-norm, RMSNorm, combined qkv, tied lm_head). Forward returns `(logits, loss)`. |
| [model_rope.py](../../../veritate_core/model_rope.py)                            | RoPE variant of the same architecture                            |
| [model_patched.py](../../../veritate_core/model_patched.py)                      | Boundary-patched trunk (local attention + global mixer over patch slots). Doc: [model_patched.md](../../platform/model_patched.md) |
| [model_recurrent.py](../../../veritate_core/model_recurrent.py)                  | Constant-state recurrent trunk and `RecurrentMixer` state rules. Doc: [model_recurrent.md](../../platform/model_recurrent.md) |
| [model_moe.py](../../../veritate_core/model_moe.py)                              | `MoEFFN` mixture-of-experts feed-forward, used by the `hybrid_moe` trunk. Doc: [model_moe.md](../../platform/model_moe.md) |
| [model_memory.py](../../../veritate_core/model_memory.py)                        | Fast-weight memory trunk variant. Doc: [model_memory.md](../../platform/model_memory.md) |
| [qat.py](../../../veritate_core/qat.py)                                          | Quantization-aware training: INT8, INT4, ternary fake-quant      |
| [qat_triton.py](../../../veritate_core/qat_triton.py)                            | Triton kernels for QAT (CUDA only)                               |
| [load.py](../../../veritate_core/load.py)                                        | Checkpoint loading utilities                                     |
| [core_plugins.py](../../../veritate_core/core_plugins.py)                        | Catalog of dashboard-selectable Core Plugins: each entry declares the trainer args it injects or omits, grouped so same-group entries are mutually exclusive |
| [plugin/__init__.py](../../../veritate_core/plugin/__init__.py)                  | The plugin contract surface: the only module a trainer may import from outside its own bundle |
| [plugin/hardware.py](../../../veritate_core/plugin/hardware.py)                  | Device detection (CPU/MPS/CUDA), physical-core count, unified-memory budget, inference-thread autotune |
| [plugin/multicorpus.py](../../../veritate_core/plugin/multicorpus.py)            | Mixed-corpus loader (`"a+b+c"` or `"a:0.5,b:0.3,c:0.2"`)         |
| [plugin/oom_recovery.py](../../../veritate_core/plugin/oom_recovery.py)          | Catch and recover from CUDA OOM during a step                    |
| [plugin/optim.py](../../../veritate_core/plugin/optim.py)                        | Optimizer builders (Muon, AdamW, paged). Doc: [optim.md](../../platform/optim.md) |
| [plugin/paged_optimizer.py](../../../veritate_core/plugin/paged_optimizer.py)    | NVMe-paged optimizer state for models past unified memory. Doc: [paged_optimizer.md](../../platform/paged_optimizer.md) |
| [plugin/mem_planner.py](../../../veritate_core/plugin/mem_planner.py)            | Pre-launch memory feasibility plan and tier selection. Doc: [mem_planner.md](../../platform/mem_planner.md) |
| [plugin/mem_executor.py](../../../veritate_core/plugin/mem_executor.py)          | Applies the planner's tier at run time. Doc: [mem_executor.md](../../platform/mem_executor.md) |
| [plugin/bench.py](../../../veritate_core/plugin/bench.py)                        | Throughput/latency bench used by auto-tune. Doc: [bench.md](../../platform/bench.md) |
| [plugin/sysprobe.py](../../../veritate_core/plugin/sysprobe.py)                  | Host probe feeding the tuning defaults. Doc: [sysprobe.md](sysprobe.md) |
| [plugin/deps.py](../../../veritate_core/plugin/deps.py)                          | Missing-package auto-installer and torch wheel repair. Doc: [deps.md](deps.md) |
| [plugin/slm.py](../../../veritate_core/plugin/slm.py)                            | Selective language modeling: a frozen reference model masks the loss down to the surprising tokens. Doc: [slm.md](../../platform/slm.md) |

## Public API surface

Imports trainers use:

```python
from veritate_core.model import Veritate           # canonical class
from veritate_core import qat as vqat              # qat ops
from veritate_core.plugin import hardware          # device detection
from veritate_core.plugin import multicorpus       # data loaders
```

## Forward contract

`Veritate.forward(tokens, targets=None)` returns `(logits, loss)`. When `targets` is None, `loss` is None. Callers must not bypass this signature: the QAT switchovers and the MTP byte-0 transform depend on it.

## QAT

`qat.set_qat(model, True)` flips fake-quant on for every QAT-aware module. `qat.set_quant_mode(model, mode)` switches between `QUANT_MODE_INT8`, `QUANT_MODE_INT4`, `QUANT_MODE_TERNARY`. The state_dict shape is unchanged because quantization is training-time only; export to `.bin` is a separate conversion step (see [export.md](export.md)).

## Pitfalls

- Vocab is hard-coded to 256 (byte-level). Any model class that overrides this breaks the engine and every reader.
- Adding a new model variant must follow the `hook_spec()` contract so [checkpoint_probe.py](checkpoint_probe.md) can walk it.
- QAT is opt-in. Trainers must call `set_qat(model, True)` before the loop; otherwise fake-quant nodes are inert.
