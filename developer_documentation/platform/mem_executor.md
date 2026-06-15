# mem_executor

Applies a [MemoryPlan](mem_planner.md) to a training run. Splits into a model-side half (activation checkpointing) and an optimizer-side half (NVMe optimizer paging), so the intervention is plan-driven and shared across every trainer instead of a per-plugin flag. Lives at [veritate_core/plugin/mem_executor.py](../../veritate_core/plugin/mem_executor.py).

## what it is

The execution half of the memory system: `mem_planner` decides the tier, `mem_executor` carries it out. Two entry points:

- `apply_plan(model, plan)` engages the model-side rung (activation checkpointing).
- `make_optimizer(params, plan, ...)` builds the optimizer the tier requires: a NVMe-[paged AdamW](paged_optimizer.md) for the optimizer-offload tiers, a plain in-RAM `torch.optim.AdamW` otherwise.

Both optimizer-offload tiers (`checkpoint+bf16_optimizer`, `checkpoint+page_optimizer_to_nvme`) are delivered by paging the Adam moments to NVMe. Paging drives resident optimizer memory toward zero, which is a superset of the bf16-moment saving the planner models for the lighter rung, so one mechanism serves both.

## how it works

- `apply_plan(model, plan)` → `AppliedPlan(tier, grad_checkpoint, optimizer_offload, unmet=())`. Engages checkpointing when `plan.tier` is in `CHECKPOINT_TIERS` (every tier whose name carries `checkpoint`). `optimizer_offload` is `True` for the tiers in `OFFLOAD_TIERS`; it tells the caller it MUST route the optimizer through `make_optimizer` rather than building a plain in-RAM AdamW, or the run will under-deliver and OOM. `unmet` is retained for backward compatibility (older callers read it); every tier is now wired, so it is always `()`.
- `make_optimizer(params, plan, *, lr, betas, eps, weight_decay, state_dir=None)` returns `PagedAdamW` for offload tiers (moments live in mmap files under `state_dir`) and `torch.optim.AdamW` otherwise.
- `enable_grad_checkpoint(model)` wraps each block in `model.blocks` with a `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` shim. Idempotent: an already-wrapped block carries a `_grad_checkpointed` marker and is skipped, so repeated calls never nest checkpoints.
- The block list is the `.blocks` contract, shared by every Veritate model class. The executor never branches on model variant (preflight rule 11a).

## correctness

Activation checkpointing is numerically transparent: it recomputes the forward during backward instead of storing it, so logits, loss, and gradients are identical to a non-checkpointed run within fp tolerance. The paged optimizer is bitwise the standard decoupled-AdamW update (its state just lives on disk).

## dependencies

- [mem_planner](mem_planner.md) for the tier constants and the `MemoryPlan` it consumes.
- [paged_optimizer](paged_optimizer.md) for the offload-tier optimizer.
- `torch.utils.checkpoint`.

## pitfalls

- The checkpoint shim assumes each block's forward takes a single residual tensor `x` (the contract every Veritate block follows). A block whose forward needs extra positional args would need a wider shim.
- Wrapping replaces the instance `forward` attribute; it relies on `nn.Module.__call__` dispatching through `self.forward`. Saving/loading a checkpointed model does not persist the wrapper; re-apply via `apply_plan` after load.
- `optimizer_offload=True` is data the trainer must act on. Calling `apply_plan` but then building a plain AdamW for an offload tier defeats the plan and OOMs. Route the optimizer through `make_optimizer`.
- Paging only frees the optimizer buckets. Weights and grads stay resident; a size whose weights+grads alone exceed the budget is infeasible regardless of paging (the planner returns `infeasible` and the trainer refuses before building).

## adoption

A trainer: build a plan from the size preset (params + shape) BEFORE constructing the model, refuse if `not plan.fits`, then `apply_plan(model, plan)` and `make_optimizer(model.parameters(), plan, ..., state_dir=<model_dir>/optim_state)`. See [trainers/contract.md](../trainers/contract.md) and the bench/real paths in `vanilla_trainer.py` / `native_trainer.py`.
