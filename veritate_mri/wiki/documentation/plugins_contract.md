---
title: plugins contract
date: 2026-05-05
tags: [plugins, contract, save, paths, model, qat, manifest]
summary: What a plugin is, what the platform gives it, and what its manifest looks like.
---

> Friendly summary. The canonical, signature-level contract is `documentation/plugins/contract.md`.

## what a plugin is

A script + a manifest under `plugins/<name>/`. The platform exposes a small set of helpers that every plugin needs to use the same way: name a model, hash its corpus, write a checkpoint, log a training row, find paths on disk. Everything specific to the plugin's job — the model, the optimizer, the training loop — is the plugin's own.

## the four namespaces

```python
from veritate.plugin import save, paths, model, qat
```

Anything else in the parent repo is internal and may move without notice.

| namespace | what it does |
|---|---|
| `save` | write checkpoints, run the dump suite, log training rows, hash corpora, validate names |
| `model` | the canonical `Veritate` byte-level decoder. One class, one shape, one place. |
| `qat` | quantization-aware training ops that match the C engine's INT8 scheme exactly |
| `paths` | pure read-only string composition over the on-disk `models/<name>/` layout |

## save (the headline calls)

| call | does |
|---|---|
| `save.save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None)` | write `step_<N>.pt` and the full dump suite (probe, lens, classroom, grades, concepts, surprise, quant_kl, generation). Returns the `.pt` path. |
| `save.append_train_row(name, step, split, loss, *, lr=None, grad_norm=None, tok_per_s=None, wall_s=None, seed=None)` | append one row to `train.csv`. Cheap, called every step. The dashboard renders the loss chart from this file. |
| `save.compose_name(corpus, size, precision, version)` | build the canonical `<corpus>_<size>_<precision>_<version>` model name |
| `save.hash_corpus(stem)` | sha256 the train (and val) `.bin` files for a corpus stem so two models can be compared honestly |
| `save.require_description(desc)` | enforce ROE rule 6 ("every model has a description") |
| `save.resolve_corpus(stem)` | return `(train_path, val_path)` from `plugins/corpus/` or the calling plugin's bundled corpus folder |

`save.save` writes `models/<name>/checkpoints/step_<N>.pt` plus `models/<name>/hooks/step_<N>/<artifact>` and bootstraps `models/<name>/config.json` if missing. Failures inside individual dump functions are logged and swallowed — the checkpoint still gets written. A failed name or description validation raises.

## model

`model.Veritate(vocab, hidden, layers, ffn, heads, seq)` is the canonical byte-level decoder. Pre-norm RMSNorm. Causal SDPA attention with combined `qkv` projection. GELU FFN. Learned positional embedding. Tied LM head (`lm_head.weight = tok_emb.weight`). All linears `bias=False`. `vocab` must be `model.VOCAB_BYTE_LEVEL` (256) — anything else raises.

Methods: `forward(tokens, targets=None)` returns `(logits, loss_or_None)` with standard CE when `targets` is given (`ignore_index=-1`); `embed(tokens)` does token + positional embedding lookup for adapters that thread state through the residual stream.

If a plugin wraps `Veritate` (adapter, head, side network), the wrapper must expose `model.base` of type `Veritate` so `save.save(model.base, ...)` can dump against the standard layout.

## qat

QAT helpers whose ops match the C engine's quantization scheme exactly. A model wrapped with these exports to a v9 binary with `act_boost=1` and runs without quantization-noise gibberish.

| call | does |
|---|---|
| `qat.fake_quant_weight(w)` | per-tensor symmetric maxabs INT8 round-and-dequant on a weight tensor, straight-through gradient |
| `qat.fake_quant_act(x, scale=32.0)` | INT8 quant on activations at the engine's residual-stream scale |
| `qat.fake_quant_ln_weight(w, scale=64.0)` | round-and-dequant for an RMSNorm weight at the engine's fixed LN scale |
| `qat.set_qat(module, value)` | recursively flip `.qat = bool(value)` on every submodule that has the attribute |

Constants: `qat.INT8_MAX = 127`, `qat.ACT_INT8_SCALE = 32.0`, `qat.LN_FIXED_SCALE = 64.0`.

The canonical `model.Veritate` already wires these in. Plugins only call them directly when they have their own matmuls outside the base (e.g. an adapter sidecar) and want those quantized too.

## paths

Pure string composition. No side effects, no filesystem creation. Plugins should not assemble these by hand — go through `paths.*` so a future layout change doesn't break every plugin.

| call | returns |
|---|---|
| `paths.model_dir(name)` | `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | one of the eight dump files |
| `paths.corpus_dir()` | `plugins/corpus/` |
| `paths.corpus_train_path(stem)` | `plugins/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `plugins/corpus/<stem>_val.bin` |

## manifest format

Every plugin ships a `manifest.json` next to its `plugin.py`. The dashboard reads it to render the trainer form; the plugin's `parse_args` reads it to bootstrap argparse defaults.

Top-level shape:

```json
{
  "name": "Human-Friendly Plugin Name",
  "description": "One-line summary shown in the trainer picker.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": { "<key>": <value>, ... }
}
```

Labels, help text, types, and `advanced` / `featured` flags live in `TRAINER_SCHEMA` (in `veritate_mri/static/index.html`), not the manifest. A plugin opts a field into the form by adding the field name to `defaults`. To add a brand-new field that isn't in the schema yet, update `TRAINER_SCHEMA` in the same commit.

### required fields

| key | type | sample |
|---|---|---|
| `size` | string | `"120m"` — shape preset name from the plugin's own `SIZE_PRESETS` |
| `precision` | string | `"bf16"` or `"fp32"` |
| `version` | string | `"v1"`, `"v1a"`, `"v2"` — for `compose_name` |

### common training-loop knobs

| key | sample | meaning |
|---|---|---|
| `total_steps` | `5000` | gradient updates |
| `batch_size`, `seq`, `n_chunks` | `8`, `256`, `48` | sequences per step, chunk length, chunks per step |
| `bptt_window` | `4` | chunks of past activations carrying gradient (M3 / M1 only; MEGA omits) |
| `base_lr`, `min_lr`, `warmup_steps`, `lr_schedule` | `0.0001`, `1e-5`, `200`, `"cosine"` | LR schedule |
| `weight_decay`, `beta1`, `beta2` | `0.01`, `0.9`, `0.95` | AdamW knobs |
| `label_smoothing`, `grad_clip` | `0.0`, `1.0` | training stability |
| `ckpt_every`, `log_every`, `eval_every`, `eval_iters` | `200`, `20`, `200`, `8` | cadence |
| `seed`, `use_act_ckpt` | `0`, `true` | RNG, activation checkpointing |

### reserved feature flags

These keys are reserved across all plugins. One trainer's `qat_enabled` checkbox and another's are the same checkbox on the dashboard, set the same field in `config.json`, and feed the same downstream signals.

| key | sample | what it does |
|---|---|---|
| `qat_enabled` | `false` | wrap matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops. Match the C engine's quant scheme so the binary exports with `act_boost=1`. |
| `quant_mode` | `"ternary"` | weight quant scheme. `"int8"` (default), `"int4"` (2x density), `"ternary"` (BitNet b1.58, 5x density). Activation and RMSNorm quant stay INT8 regardless. |
| `use_8bit_adam` | `true` | `bitsandbytes.optim.AdamW8bit` instead of `torch.optim.AdamW`. INT8 storage for the moments. Required to fit ~1B-class MoE on 12 GB GPUs. Trainer must `import bitsandbytes` only when set, so plugins that don't need it never pull the dep. |

### M1 / M3 adapter clusters

Only for plugins that use them.

| key | sample | applies to | meaning |
|---|---|---|---|
| `rank` | `32` | M3 | low-rank adapter rank |
| `n_slots` | `256` | M1 | named slot vectors in the working-memory table |
| `alpha` | `0.2` | M1, M3 | per-token write strength to the adapter state |
| `inject_layer` | `-1` | M1, M3 | which layer the adapter attaches to; `-1` = mid-stack |
| `init_from` | `""` | M1 | name of an existing model whose latest checkpoint seeds the base. New model is named `<init_from>_m1`. |
| `freeze_base` | `false` | M1 | freeze the base; only the adapter trains |

### MEGA cluster

Ternary + MoE moonshot.

| key | sample | meaning |
|---|---|---|
| `quant_mode` | `"ternary"` | weight quant scheme |
| `n_experts` | `8` | FFN experts per block. Total params scale linearly. |
| `router_topk` | `1` | experts active per token. `1` = sticky single-expert routing (cheapest, L3-fittest). |
| `router_aux_loss` | `0.01` | Switch-Transformer load-balance coefficient; prevents router collapse |
| `use_8bit_adam` | `true` | see reserved flags |

## minimum trainer manifest

```json
{
  "name": "Example Plugin (minimal)",
  "description": "Tiny scratch trainer used as a manifest template.",
  "kind": "trainer",
  "flow": ["scratch"],
  "defaults": {
    "size": "30m",
    "precision": "bf16",
    "version": "v1",
    "total_steps": 1000,
    "batch_size": 8,
    "seq": 256,
    "n_chunks": 8,
    "base_lr": 0.0003,
    "min_lr": 1e-05,
    "warmup_steps": 100,
    "lr_schedule": "cosine",
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "label_smoothing": 0.0,
    "grad_clip": 1.0,
    "ckpt_every": 100,
    "log_every": 10,
    "eval_every": 100,
    "eval_iters": 4,
    "seed": 0,
    "use_act_ckpt": false,
    "qat_enabled": false
  }
}
```

The canonical contract has full M1 and MEGA examples plus the line-by-line update obligation.

## what plugins must not do

- Import from `veritate_mri.*` directly. Use `veritate.plugin.save` and `veritate.plugin.paths`.
- Write outside `models/<name>/`. The dashboard reads from a fixed layout; writes elsewhere are invisible.
- Edit `config.json` after `save()` has bootstrapped it, except via fields the contract defines.
- Invent your own dump artifacts. The dashboard renders the eight in `HOOK_ARTIFACTS`. Add real ones via the hooks contract update process.

## stability

Function and field names here are stable across patch releases. A signature change or rename is a minor-version bump and requires the canonical contract's update obligation to be honored in the same commit.
