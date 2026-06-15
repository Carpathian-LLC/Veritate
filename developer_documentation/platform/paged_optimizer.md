# paged_optimizer

Decoupled-AdamW whose optimizer state (`exp_avg`, `exp_avg_sq`) lives in mmap-backed files on NVMe instead of RAM. It is the mechanism behind the [mem_executor](mem_executor.md) optimizer-offload tiers: paging the Adam moments out of the resident pool lets a model whose full fp32 Adam state exceeds unified memory still train. Lives at [veritate_core/plugin/paged_optimizer.py](../../veritate_core/plugin/paged_optimizer.py).

## what it is

`PagedAdamW(params, lr, betas, eps, weight_decay, state_dir=None)` is a drop-in `torch.optim.Optimizer`. Its update is bitwise the standard decoupled-AdamW step; the only difference from `torch.optim.AdamW` is where the two moment buffers live. On a unified-memory host the moments are the largest single training bucket (8 B/param at fp32 — larger than weights or grads), so moving them to disk is the highest-leverage single reduction.

## how it works

- For each parameter, `exp_avg` and `exp_avg_sq` are fp32 tensors created with `torch.from_file(..., shared=True)` under `state_dir`. `MAP_SHARED` means writes go to the page cache and the OS spills cold pages to the file under memory pressure; resident optimizer memory is bounded by the page cache, not the full state size.
- `step()` does the AdamW math in fp32: moves each param's grad to host, updates the file-backed moments in place, applies the decoupled weight-decay shrink then the `addcdiv` update to the param on its own device. One parameter at a time, so transient extra memory is O(largest param tensor).
- `state_dir=None` allocates a throwaway temp dir (used by [bench](bench.md)); `close()`/`__del__` removes it. An explicit `state_dir` (a real run passes `<model_dir>/optim_state`) is kept across resume.
- `_file_backed` zeroes only newly-created files; an existing correctly-sized file is mapped as-is, so resume preserves the moments.
- `state_dict()` carries only the per-param step counts plus `state_dir` — never the moment buffers — so checkpoints stay tiny. `load_state_dict()` restores the steps and rebinds the on-disk files in place.

## dependencies

- `torch` (`torch.from_file` for the mmap, standard tensor ops for the update).
- Consumed by [mem_executor.make_optimizer](mem_executor.md) and [bench](bench.md).

## pitfalls

- **It is I/O-bound.** Each step touches the full moment state, so step time at scale is bound by NVMe bandwidth. Paging shines when the optimizer state only modestly exceeds RAM; for state many times larger than RAM the per-step disk traffic dominates and tok/s collapses. The bench measures the real paged tok/s so the trade is visible, never hidden.
- **It only frees the optimizer buckets.** Weights and grads stay resident. A model whose weights+grads alone exceed the budget does not fit no matter how the optimizer is paged; the planner returns `infeasible` upstream.
- The moment math runs on host (CPU) tensors; for huge models that host-side elementwise pass is itself non-trivial. This is inherent to optimizer offload.
- On process kill, dirty `MAP_SHARED` pages may not have flushed; rely on checkpoint cadence for durability, not on the live state files.
