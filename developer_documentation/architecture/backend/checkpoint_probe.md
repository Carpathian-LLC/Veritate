# checkpoint_probe

## What it is

Runs at checkpoint time and produces the per-step artifacts in `models/<name>/hooks/step_<N>/`. Lives at [veritate_mri/training/checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py).

## Artifacts produced

| File                       | Content                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| `probe.json`               | Top-k FFN neurons per layer + logit lens + residual norms, aggregated (mean) over a deterministically sampled seed collection (`PROBE_SEEDS`, `sample_probe_prompts`) |
| `lens.npz`                 | Per-layer logits over vocab (int32) and residual norms (float32) ([line 250](../../../veritate_mri/training/checkpoint_probe.py#L250)) |
| `classroom.json`           | Per-grade reading perplexity                                                                             |
| `grades.json`              | Pass/fail at grade bands                                                                                 |
| `math.json`, `grammar.json`, `reasoning.json` | Capability evals; pass threshold 0.80, emerging 0.50 ([line 70-71](../../../veritate_mri/training/checkpoint_probe.py#L70)) |
| `concepts.json`            | 50-concept surprise probe (bits/byte)                                                                    |
| `surprise.json`            | Per-token surprise on a held-out prompt set                                                              |
| `quant_kl.json`            | KL divergence between fp32 and quantized predictions                                                     |
| `generation.json`          | Sample greedy generation outputs at the current step                                                     |
| `writing_health.json`, `reading_comprehension.json` | Higher-tier evals                                                                          |

## How it works

Called by [save.py](save.md) at every checkpoint. Functions named `dump_<artifact>` produce a file prefixed with `_step_<N>.json` or `.npz`, then `save.py` renames them via `RENAME_MAP_TEMPLATE` to the canonical names.

The probe runs in `torch.no_grad()` on the model in eval mode. No gradient state is mutated.

**Per-block snapshot column (rule 23).** `dump_probe` and `dump_generation` snapshot each block's activations at the column returned by `model.probe_columns(tokens)` (defaults to the last byte when the model does not define it). Dense/recurrent trunks read the last byte for every block. The patched/hybrid trunk runs its global stack on a slot stream whose trailing slots are masked padding (exactly zero); its `probe_columns` returns the last **live** slot for those global blocks, so their residual norms, logit lens, and top neurons are non-zero. Global GLA blocks carry no per-position attention, so `dump_generation` emits `attn[L] = []` for them, mirroring the C engine and the inference Brain.

**Probe seeds.** `PROBE_SEEDS` is a fixed pool of short, diverse seeds (chat / knowledge / code). `sample_probe_prompts()` draws `PROBE_SAMPLE_N` of them under a fixed RNG seed (`PROBE_SAMPLE_SEED`), so the collection is diverse yet identical across checkpoints (step-to-step comparisons stay valid). `save.py` passes this collection to `dump_probe`; single-prompt dumps (`surprise`, `quant_kl`, `generation`) use `PROBE_PROMPT`.

**Writing-health PMI.** `dump_writing_health` scores adjacent word pairs in each generation against the training corpus's bigram index. `_wh_load_pmi_index` resolves the sidecar through `build_bigram_index.sidecar_path` and builds it on first use, capped at `WH_PMI_MAX_SCAN_BYTES` so a large corpus cannot stall a save; the loaded index is cached per process (`WH_PMI_CACHE_MAX`). PMI is null only when no corpus path reaches the dump. See [bigram index](../../corpus/bigram_index.md).

**Grade eval data.** `EVAL_ROOT` resolves through `readers.paths.GRADE_EVAL_ROOT` (`veritate_mri/data/eval/grade/`), the same root `dump_grades` uses. The `comprehension_*.json` files sit directly under it; `math/`, `grammar/`, `reasoning/` are subdirs of tier `.jsonl` files.

## Dependencies

- [save.py](save.md): orchestrates the calls and the rename.
- The model class from [veritate_core/model.py](../../../veritate_core/model.py): uses `hook_spec()` for the probe view.
- [readers/hooks.py](../../../veritate_mri/readers/hooks.py): reads these artifacts back out for the Learning tab.

## Pitfalls

- Probe runtime scales with vocab × hidden × seq × number of sampled seeds. `dump_probe` runs one forward per seed (`PROBE_SAMPLE_N`); keep the sample count modest and checkpoint cadence sparse so probe time stays well under the training time between checkpoints.
- The seed collection is deterministic (fixed RNG seed), not fixed to one sentence: step-to-step comparisons stay meaningful because the same seeds are drawn every checkpoint. Do not seed the sampler from wall-clock or step.
- A variant whose blocks run on a non-byte stream (e.g. the patched-trunk slot stack) MUST expose `probe_columns(tokens)`, or its non-byte blocks get snapshotted at a masked/padding position and read zero.
- Adding a new probe artifact requires updating both `dump_<name>` here and the rename map in [save.py](../../../veritate_mri/training/save.py).
- The graded question sets this probe scores against are regenerated by `POST /eval_sets`; a new probe also needs its builder registered there. See [eval sets](../../training/eval_sets.md).
