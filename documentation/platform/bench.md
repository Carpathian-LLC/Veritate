# bench

Empirical training-memory + throughput benchmark. Ramps the batch size on a real model until the device runs out of memory, recording measured memory and tok/s at each rung. Powers the dashboard's Auto tune. Lives at [veritate_core/plugin/bench.py](../../veritate_core/plugin/bench.py).

## what it is

The measured counterpart to [mem_planner](mem_planner.md)'s analytic estimate. The estimate sizes the offload-tier decision at run start; bench replaces guesswork where guesswork undershoots: it runs short throwaway training steps on the actual model (MoE routing, variant footprint and all) and reports what the machine really does. Distinct from the canonical-prompt neuron probe (`probe.json` in the dump suite); this is a hardware benchmark.

## how it works

- `bench.run(model, device, seq, vocab, batch_ramp=DEFAULT_BATCH_RAMP, on_progress=None)` → result dict.
- For each batch size in the ramp: synthetic random byte batches, a throwaway AdamW, `WARMUP_STEPS` untimed then `TIMED_STEPS` timed forward/backward/step. Records device high-water memory and tok/s, then frees the cache.
- Stops at the first OOM (detected via [oom_recovery.is_oom_error](../../veritate_core/plugin/oom_recovery.py)); the largest rung that fit is the ceiling.
- `on_progress(str)` receives human-readable lines as it runs; the trainer prints them with a `bench:` prefix and the dashboard modal renders them live.
- Result: `{device, seq, max_batch, mem_ceiling_gb, tok_per_s, ramp: [{batch, mem_gb, tok_per_s}, ...]}`.
- Forward return contract: loss is index 1 of the model's output tuple across every variant (MoE adds an aux term at index 2).
- Nothing is saved: no checkpoint, no CSV, no real weights touched.

## trainer integration

A trainer exposes bench as a `--bench` flag: build the model exactly as for training (size/seq/precision flags all apply), then instead of training call `bench.run` and print the result line:

```python
if args.bench:
    result = bench.run(model, device, args.seq, VOCAB,
                       on_progress=lambda s: print("bench: " + s, flush=True))
    print("BENCH_RESULT " + json.dumps(result), flush=True)
    return
```

Bench mode skips the description requirement and corpus resolution (no data is read). The trainer's manifest declares `"bench": true` so the dashboard knows the flag is implemented — without the declaration, `parse_known_args` would silently drop `--bench` and start a real run.

## dashboard flow (Auto tune)

1. Auto tune (Training tab next to the memory estimate, or Settings next to Detect system) opens a modal listing bench-capable trainers.
2. Start launches the trainer through the normal runner (`POST /trainers/run` with `bench: true`); single-instance enforcement means it cannot touch a live training run.
3. The modal tails `bench:` lines from the log-ring SSE stream and renders them live; `BENCH_RESULT` ends the run.
4. The recommendation is the throughput sweet spot (fastest measured batch, capped by the MPS INT_MAX attention-tensor limit), with lr sqrt-scaled from the manifest pair and cadence scaled to total_steps.
5. Apply writes the values into the trainer's `manifest.json` defaults (`POST /trainers/tune_defaults`) and the measured summary into `data/system_specs.json` (`measured` key, shown in Settings).

## pitfalls

- Measured values are machine-specific. A tuned manifest must never be pushed upstream; the trainers sync marks it `modified` and protects it locally.
- The ramp benches the configuration as submitted — `--use_act_ckpt` wraps blocks before the bench branch, so checkpointed and non-checkpointed configs measure differently (as they should).
- Throughput is not monotonic in batch size on MPS; the largest batch that fits is often not the fastest. Use `ramp` for the full curve.

## tests

[tests/plugin_contract/test_bench.py](../../tests/plugin_contract/test_bench.py): CPU full-ramp contract, positive throughput, saves-nothing, and the MPS OOM-ceiling path (slow).
