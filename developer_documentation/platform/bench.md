# bench

Empirical training-memory + throughput benchmark. Ramps the batch size on a real model until the device runs out of memory, recording measured memory and tok/s at each rung. Powers the dashboard's Auto tune. Lives at [veritate_core/plugin/bench.py](../../veritate_core/plugin/bench.py).

## what it is

The measured counterpart to [mem_planner](mem_planner.md)'s analytic estimate. The estimate sizes the offload-tier decision at run start; bench replaces guesswork where guesswork undershoots: it runs short throwaway training steps on the actual model (MoE routing, variant footprint and all) and reports what the machine really does. Distinct from the canonical-prompt neuron probe (`probe.json` in the dump suite); this is a hardware benchmark.

## how it works

- `bench.run(model, device, seq, vocab, plan=None, batch_ramp=DEFAULT_BATCH_RAMP, on_progress=None)` → result dict.
- For each batch size in the ramp: synthetic random byte batches, `WARMUP_STEPS` untimed then `TIMED_STEPS` timed forward/backward/step. Records device high-water memory and tok/s, then frees the cache.
- The probe optimizer follows the plan: when `plan.tier` is an [mem_executor](mem_executor.md) offload tier the optimizer is the NVMe-[paged AdamW](paged_optimizer.md), so the measured tok/s reflects the real disk-bound regime rather than a RAM-only fantasy. Otherwise a throwaway in-RAM AdamW.
- Stops on a **measured memory budget**, not by waiting for OOM: `BUDGET_FRACTION` (0.85, matching `mem_planner.USABLE_FRACTION`) × detected RAM/VRAM. Once a rung's high-water reaches the budget, the next rung is not attempted. This is required on unified memory, where an over-budget allocation is SIGKILLed by the OS rather than raised as a catchable error — waiting for OOM loses the whole result (and the process). A catchable OOM (`oom_recovery.is_oom_error`, mainly cuda) is still handled as a secondary stop. A backend tensor-size error is a third stop: MPS caps any single graph tensor at INT_MAX elements (the attention scores `batch x heads x seq x seq` overflow this before the memory budget on large-RAM machines), so `SIZE_LIMIT_MARKERS` (`int_max`, `tensor dims larger`, `invalid buffer size`) is treated as a ceiling rather than crashing the run. A non-OOM, non-size-limit failure at a rung after at least one fit is also taken as the ceiling; only a failure on the very first rung re-raises (a real model bug, not a capacity limit).
- `on_progress(str)` receives human-readable lines as it runs; the trainer prints them with a `bench:` prefix and the dashboard modal renders them live.
- Feasible result: `{device, seq, fits: true, tier, max_batch, mem_ceiling_gb, tok_per_s, ramp: [...], required_gb, budget_gb, params_gb, grads_gb, optimizer_gb}`.
- `bench.plan_result(plan, device, seq)` is the infeasible result a trainer emits instead of building a model whose weights+grads exceed the budget: `{fits: false, tier, max_batch: 0, ramp: [], + the bucket gb breakdown}`. Paging the optimizer cannot rescue this case, so the bench is never run.
- Forward return contract: loss is index 1 of the model's output tuple across every variant (MoE adds an aux term at index 2).
- Nothing is saved: no checkpoint, no CSV, no real weights touched (the paged optimizer's throwaway state dir is removed on close).

## trainer integration

A trainer exposes bench as a `--bench` flag. It plans the memory ladder from the size preset BEFORE building the model (a model whose weights+grads exceed the budget cannot be built — the allocation OOMs/SIGKILLs), refuses with `plan_result` when infeasible, and otherwise builds, applies the plan's checkpointing, and benches with the plan's optimizer:

```python
plan = mem_planner.plan_training_memory(shape["params"], shape["hidden"],
                                        shape["layers"], shape["ffn"], 1, args.seq, "fp32")
if not plan.fits:
    print("BENCH_RESULT " + json.dumps(bench.plan_result(plan, device, args.seq)), flush=True)
    return
# build model, then:
mem_executor.enable_grad_checkpoint(model)           # when plan.tier checkpoints
result = bench.run(model, device, args.seq, VOCAB, plan=plan,
                   on_progress=lambda s: print("bench: " + s, flush=True))
print("BENCH_RESULT " + json.dumps(result), flush=True)
return
```

Bench plans at batch 1 (the ramp explores batch); a real run plans at its configured batch. Bench mode skips the description requirement and corpus resolution (no data is read). The trainer's manifest declares `"bench": true` so the dashboard knows the flag is implemented — without the declaration, `parse_known_args` would silently drop `--bench` and start a real run.

## dashboard flow (Auto tune)

1. Auto tune (Training tab next to the memory estimate, or Settings next to Detect system) opens a modal listing bench-capable trainers.
2. Start launches the trainer through the normal runner (`POST /trainers/run` with `bench: true`); single-instance enforcement means it cannot touch a live training run.
3. The modal tails `bench:` lines from the log-ring SSE stream and renders them live; `BENCH_RESULT` ends the run.
4. The recommendation is the throughput sweet spot (fastest measured batch, capped by the MPS INT_MAX attention-tensor limit), with lr sqrt-scaled from the manifest pair and cadence scaled to total_steps. The modal shows the memory strategy (the plan tier); when the size only fits by paging the optimizer to NVMe it says so and notes the tok/s is disk-bound.
5. An infeasible result (`fits: false`) renders the floor breakdown — weights + grads + optimizer vs the machine budget — and the honest remedies (smaller size, shorter seq, bigger machine) instead of a recommendation. No subprocess ever allocates the oversized model.
6. Apply writes the values into the trainer's `manifest.json` defaults (`POST /trainers/tune_defaults`) and the measured summary into `data/system_specs.json` (`measured` key, shown in Settings). When the plan checkpoints, `use_act_ckpt` is included in the applied args so the form reflects the regime.

## pitfalls

- Measured values are machine-specific. A tuned manifest must never be pushed upstream; the trainers sync marks it `modified` and protects it locally.
- Checkpointing in the bench is plan-driven: the trainer calls `enable_grad_checkpoint` when the plan's tier checkpoints, so the ramp measures the regime the size actually needs.
- Throughput is not monotonic in batch size on MPS; the largest batch that fits is often not the fastest. With a paged optimizer, larger batches amortize the fixed per-step disk traffic, so the sweet spot trends larger. Use `ramp` for the full curve.

## tests

[tests/plugin_contract/test_bench.py](../../tests/plugin_contract/test_bench.py): CPU full-ramp contract, positive throughput, saves-nothing, the memory-budget stop, the backend size-limit stop (MPS INT_MAX), and the MPS OOM-ceiling path (slow).
