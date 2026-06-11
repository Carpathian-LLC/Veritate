# Trainer plugin contract

The platform surface that trainers are allowed to import. Trainers may only reach `veritate_core.plugin` (or the legacy `veritate.plugin` alias via the [shim](../architecture/backend/veritate_shim.md)). Direct imports from `veritate_mri/` or `veritate_engine/` are forbidden by preflight rule 39.

The full, authoritative API reference (every `save`, `paths`, `model`, `qat`, `hardware` call, the manifest schema, and reserved flags) lives in [trainers/contract.md](../trainers/contract.md). This file is the short orientation; trainers/contract.md is the spec.

## Plugin bundle layout

```
trainers/<plugin_id>/
├── manifest.json
└── trainer.py
```

Optional per-trainer files (corpus builders, helpers used by only this trainer) live in the same directory. Shared helpers live in `trainers/common/` (preflight rule 36).

## manifest.json schema

```json
{
  "name": "Human-readable name",
  "description": "One-sentence purpose.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": {
    "size": "30m",
    "batch": 8,
    "seq": 256,
    "total_steps": 6000
  }
}
```

- `name` — display name in the dashboard.
- `kind` — `"trainer"` (currently the only kind).
- `flow` — list of valid entry modes. `scratch` = new model; `continue` = resume from a checkpoint.
- `defaults` — every argparse arg the plugin accepts, with its default value. The dashboard generates form fields from this; missing keys mean missing form fields.
- `bench` (optional) — `true` when the trainer implements the `--bench` flag ([bench.md](../platform/bench.md)). Gates the dashboard's Auto tune; without it the flag would be silently dropped by `parse_known_args` and a real run would start.

## trainer.py contract

The trainer is a standalone Python script. It must:

1. **Import only from the allowed surface.** `veritate_core.plugin` performs its own `sys.path` setup, so trainers do not inject paths manually:

   ```python
   from veritate_core.plugin import save, paths, model, qat, multicorpus
   ```

   Direct imports from `veritate_mri.routes`, `veritate_mri.runtime`, or `veritate_engine` are forbidden.

2. **Build argparse from manifest defaults** (the convention; not strictly enforced).

3. **Write the canonical CSV** via [`save.append_train_row(name, step, split, loss, lr=..., grad_norm=..., tok_per_s=..., wall_s=..., seed=...)`](../architecture/backend/save.md) at every log interval.

4. **Save checkpoints** via `save.save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None)` so the dump suite runs ([checkpoint_probe](../architecture/backend/checkpoint_probe.md)). Every save writes the checkpoint, `config.json`, and the full hooks dump suite. No trainer writes `.pt` files, CSV rows, or dump artifacts directly.

5. **Stream progress to stdout** with `print(..., flush=True)`. The dashboard tails stdout into the in-memory log ring.

## Plugin surface (`veritate_core.plugin`)

| Module                                                                                | Provides                                                       |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [save](../../veritate_mri/training/save.py)                                           | `save()`, `append_train_row()`, `compose_name()`, `hash_corpus()`, `resolve_corpus()`, `truncate_train_csv_at()` |
| [paths](../../veritate_mri/readers/paths.py)                                          | read-only path composition over `models/<name>/` and `trainers/corpus/` |
| [model](../../veritate_core/model.py)                                                 | `Veritate(...)` canonical byte-level decoder, `VOCAB_BYTE_LEVEL` |
| [qat](../../veritate_core/qat.py)                                                     | fake-quant helpers, `set_qat`, `set_quant_mode`, quant constants |
| [hardware](../../veritate_core/plugin/hardware.py)                                    | `pick_device()`, `physical_cores()`, device detection          |
| [multicorpus](../../veritate_core/plugin/multicorpus.py)                              | `make_mixed_loader(stems_or_spec, ...)` for `"a+b"` or `"a:0.5,b:0.5"` |
| [oom_recovery](../../veritate_core/plugin/oom_recovery.py)                            | Wrap a training step to catch and recover from CUDA OOM        |
| [bench](../../veritate_core/plugin/bench.py)                                          | `run(model, device, seq, vocab)` measured memory/throughput ramp for Auto tune ([doc](../platform/bench.md)) |
| [mem_planner](../../veritate_core/plugin/mem_planner.py)                              | `plan_training_memory(...)` size-adaptive unified-memory plan ([doc](../platform/mem_planner.md)) |
| [mem_executor](../../veritate_core/plugin/mem_executor.py)                            | `apply_plan(model, plan)` applies the plan (activation checkpointing) ([doc](../platform/mem_executor.md)) |

Full signatures and semantics for each namespace are in [trainers/contract.md](../trainers/contract.md).

## Launch paths

- **Dashboard Training tab** → calls `/trainers/<id>/start` → `trainer_runner.start(plugin_id, args)`.
- **Direct CLI** → `python trainers/<id>/trainer.py --arg val ...`. Bypasses the runner.
- **Programmatic** → `from training import trainer_runner; trainer_runner.start(id, args)`.

The dashboard's Training tab is the canonical path. Direct CLI is valid for experiments but bypasses the heartbeat primary-detection path; the [heartbeat fallback](../architecture/backend/heartbeat.md) (train.csv mtime scan) catches direct-CLI runs.

## Forbidden

- No `import` from `veritate_mri.routes`, `veritate_mri.inference`, `veritate_engine`, or any other internal path.
- No writing to `models/<name>/` outside the `save.py` API.
- No background threads outside the main training loop.
- No new files outside the plugin bundle and `models/<name>/`.

## Pitfalls

- The `from veritate.X import` aliases work today via the [shim](../architecture/backend/veritate_shim.md). New code should use `from veritate_core.plugin import save, paths, model, qat, ...` instead.
- Subprocess stdout is line-buffered. Use `flush=True` or unbuffered output (`python -u`) so logs appear in the dashboard in real time.
- Single-instance enforced by `trainer_runner.start()`. Direct CLI launches can be concurrent — be deliberate.
