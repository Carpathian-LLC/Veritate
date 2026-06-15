# checkpoint_probe

## What it is

Runs at checkpoint time and produces the per-step artifacts in `models/<name>/hooks/step_<N>/`. Lives at [veritate_mri/training/checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py).

## Artifacts produced

| File                       | Content                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| `probe.json`               | Top-k FFN neurons per layer + logit lens + residual norms on a fixed prompt ([line 199–251](../../../veritate_mri/training/checkpoint_probe.py#L199)) |
| `lens.npz`                 | Per-layer logits over vocab (int32) and residual norms (float32) ([line 250](../../../veritate_mri/training/checkpoint_probe.py#L250)) |
| `classroom.json`           | Per-grade reading perplexity                                                                             |
| `grades.json`              | Pass/fail at grade bands                                                                                 |
| `math.json`, `grammar.json`, `reasoning.json` | Capability evals; pass threshold 0.80, emerging 0.50 ([line 70–71](../../../veritate_mri/training/checkpoint_probe.py#L70)) |
| `concepts.json`            | 50-concept surprise probe (bits/byte)                                                                    |
| `surprise.json`            | Per-token surprise on a held-out prompt set                                                              |
| `quant_kl.json`            | KL divergence between fp32 and quantized predictions                                                     |
| `generation.json`          | Sample greedy generation outputs at the current step                                                     |
| `writing_health.json`, `reading_comprehension.json` | Higher-tier evals                                                                          |

## How it works

Called by [save.py](save.md) at every checkpoint. Functions named `dump_<artifact>` produce a file prefixed with `_step_<N>.json` or `.npz`, then `save.py` renames them via `RENAME_MAP_TEMPLATE` to the canonical names.

The probe runs in `torch.no_grad()` on the model in eval mode. No gradient state is mutated.

## Dependencies

- [save.py](save.md) — orchestrates the calls and the rename.
- The model class from [veritate_core/model.py](../../../veritate_core/model.py) — uses `hook_spec()` for the probe view.
- [readers/hooks.py](../../../veritate_mri/readers/hooks.py) — reads these artifacts back out for the Learning tab.

## Pitfalls

- Probe runtime scales with vocab × hidden × seq. On large models the dump can take seconds; checkpoint cadence should be sparse enough that probe time is much less than training time between checkpoints.
- Probes use a fixed prompt set so step-to-step comparisons are meaningful. Don't randomize the probe inputs.
- Adding a new probe artifact requires updating both `dump_<name>` here and the rename map in [save.py:51](../../../veritate_mri/training/save.py#L51).
