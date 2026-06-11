# mem_executor

Applies a [MemoryPlan](mem_planner.md) to a model. Owns the activation-checkpointing wiring that trainers previously monkeypatched inline, so the intervention is plan-driven and shared across every trainer instead of a per-plugin flag. Lives at [veritate_core/plugin/mem_executor.py](../../veritate_core/plugin/mem_executor.py).

## what it is

The execution half of the memory system: `mem_planner` decides the tier, `mem_executor` carries it out on the model. Today it wires the activation-checkpointing rung. The optimizer-level rungs (bf16 moments, NVMe paging) are not yet wired; when a plan needs one, the executor reports it in `unmet` so the trainer surfaces the gap loudly rather than silently under-delivering and OOMing.

## how it works

- `apply_plan(model, plan)` → `AppliedPlan(tier, grad_checkpoint, unmet)`. Engages checkpointing when `plan.tier` is in `CHECKPOINT_TIERS` (every tier whose name carries `checkpoint`), and returns `unmet` listing interventions the tier needs that this executor does not perform.
- `enable_grad_checkpoint(model)` wraps each block in `model.blocks`, replacing `blk.forward` with a `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` shim. Idempotent: an already-wrapped block carries a `_grad_checkpointed` marker and is skipped, so repeated calls never nest checkpoints.
- The block list is the `.blocks` contract, shared by the canonical `Veritate`, `VeritateRoPE`, and trainer model classes (`Mega`, `Veritate800M`). The executor never branches on model variant (preflight rule 11a).

## correctness

Activation checkpointing is numerically transparent: it recomputes the forward during backward instead of storing it, so logits, loss, and gradients are identical to a non-checkpointed run within fp tolerance. [tests/plugin_contract/test_mem_executor.py](../../tests/plugin_contract/test_mem_executor.py) asserts this (`test_checkpointing_is_numerically_transparent`) and that checkpointing measurably lowers retained activation memory on MPS (`test_checkpointing_lowers_retained_activation_memory`, slow). Measured reduction: ~5% of the non-checkpointed activation store retained.

## dependencies

- [mem_planner](mem_planner.md) for the tier constants and the `MemoryPlan` it consumes.
- `torch.utils.checkpoint`.

## pitfalls

- The checkpoint shim assumes each block's forward takes a single residual tensor `x` (the contract every Veritate block follows). A block whose forward needs extra positional args would need a wider shim.
- Wrapping replaces the instance `forward` attribute; it relies on `nn.Module.__call__` dispatching through `self.forward`. Saving/loading a checkpointed model does not persist the wrapper (it is not a parameter), which is correct: re-apply via `apply_plan` after load.
- `unmet` is data the trainer must act on. A non-empty `unmet` means the plan's memory target is not fully met by what was applied; the trainer should log it and expect the planner's headroom assumption to be optimistic for that run.

## adoption

A trainer replaces its inline checkpointing with two calls: build a plan from the model's param count and shape, then `apply_plan(model, plan)`. See [trainers/contract.md](../trainers/contract.md) for the canonical call site.
