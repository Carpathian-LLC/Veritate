# mem_planner

Size-adaptive training-memory planner. Decides, at startup, whether a run fits in unified memory and which offload tier to engage. Lives at [veritate_core/plugin/mem_planner.py](../../veritate_core/plugin/mem_planner.py); every trainer reaches it through the plugin surface.

## what it is

A pure-arithmetic estimator. Given a model's parameter count and shape plus the run's batch, sequence length, and dtype, it sums the four training-memory buckets and compares the total against the host's usable unified-memory budget, returning the cheapest escalation tier that fits. No torch, no device, no allocation: it runs before the model is on device, so a trainer can size its run up front instead of discovering an OOM mid-step.

## the four buckets

- **params** — `param_count × dtype_bytes`.
- **grads** — same size as params.
- **optimizer** — Adam's two fp32 moment buffers, plus one fp32 master copy when training in a low-precision dtype.
- **activations** — coarse `batch × seq × layers × (hidden + ffn)` store with an overhead factor for attention and norm scratch.

## escalation ladder

Apple Silicon has no separate VRAM; the GPU shares the unified RAM pool, so moving a tensor "to host" frees nothing. The ladder only uses rungs that cut a real bucket:

1. `none` — everything resident.
2. `checkpoint_activations` — recompute activations in backward instead of storing them; the activation bucket drops to `CHECKPOINT_ACT_RETAIN`.
3. `checkpoint+bf16_optimizer` — moment buffers in bf16, halving the optimizer bucket.
4. `checkpoint+page_optimizer_to_nvme` — optimizer state paged to SSD, removed from the RAM budget. This is the "train bigger than RAM" rung.
5. `infeasible_reduce_batch_or_seq` — params + grads alone exceed the budget; no offload helps, the run must shrink batch or seq.

`plan_training_memory()` walks the rungs in order and returns the first that fits, so the result is the minimal intervention. Tier is monotonic in budget: a smaller budget never yields a cheaper tier.

## how it works

- `plan_training_memory(param_count, hidden, layers, ffn, batch, seq, dtype="bf16", budget_bytes=None)` → `MemoryPlan`. `param_count` is `sum(p.numel() for p in model.parameters())`. `budget_bytes` overrides auto-detection (tests pass explicit budgets).
- Budget defaults to `hardware.unified_memory_bytes() × USABLE_FRACTION` ([mem_planner.py:USABLE_FRACTION](../../veritate_core/plugin/mem_planner.py)). The 0.85 headroom covers OS, framework, and MPS-allocator fragmentation; MPS cannot hand out the full pool.
- `MemoryPlan` is a frozen dataclass: `tier`, `fits`, `budget_bytes`, `required_bytes`, and the four bucket sizes.
- `format_plan(plan)` renders a one-line GB summary for the trainer to print to the dashboard log.

## dependencies

- [hardware.unified_memory_bytes()](../../veritate_core/plugin/hardware.py) for the budget. On a discrete-GPU host that value is system RAM, not VRAM, so the unified-budget interpretation only holds when the device is `mps`.

## calibration

`ACT_OVERHEAD` and `CHECKPOINT_ACT_RETAIN` are fit against measured MPS forward retention (memory held after forward, before backward, minus the param/buffer baseline via `torch.mps.current_allocated_memory()`). Measured points:

| shape | batch×seq | measured | estimate | est/meas |
|---|---|---|---|---|
| h512 L8 ffn2048 | 8×256 | 883 MB | 1080 MB | 1.22 |
| h1024 L12 ffn4096 | 4×512 | 3241 MB | 3240 MB | 1.00 |

Estimate is biased to meet-or-exceed measured: under-prediction OOMs and costs training hours, over-prediction only offloads slightly early. Checkpointing measured at ~5% retention across shapes; `CHECKPOINT_ACT_RETAIN = 0.06` sits just above it. Re-fit if the model's block structure changes materially.

## pitfalls

- The activation estimate is linear in `(hidden + ffn) × batch × seq × layers`. It does not carry a separate `seq²` attention term, so very long sequences drift toward under-prediction; the `est/meas = 1.00` point above is already at the conservative edge for seq 512. Add an attention term before trusting it past seq ~2k.
- The planner decides the tier; it does not execute offload. Wiring checkpointing, bf16 moments, and NVMe paging into the training step is separate work that consumes this plan.
- Optimizer bucket assumes Adam. A different optimizer (e.g. Muon) changes the moment-slot count; the constant must follow the optimizer the trainer actually uses.

## tests

[tests/plugin_contract/test_mem_planner.py](../../tests/plugin_contract/test_mem_planner.py) pins the ladder boundaries, tier monotonicity, and size-adaptivity (same budget, small model clean vs large model offloading).
