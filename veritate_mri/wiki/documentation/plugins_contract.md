---
title: plugins contract
date: 2026-05-05
tags: [plugins, contract, save, paths, model, qat, manifest]
summary: The platform surface plugins are allowed to call. save / model / qat / paths namespaces and the manifest format.
---

> Source: `documentation/plugins/contract.md` (mirrored copy; the file at that path remains the canonical contract).

# plugins contract

The platform surface that plugins are allowed to call. Anything not listed here is internal. Plugins must not import from it, and the platform may rename or restructure it without notice.

This is a versioned contract. Adding, removing, or changing the signature of any function below requires updating this file in the same commit. Same rule as [hooks/contract.md](../hooks/contract.md).

## scope

A plugin is a script + manifest under `plugins/<name>/`. The platform exposes a small set of helpers for the things every plugin needs to do consistently: name a model, hash its corpus, write a checkpoint, log a training row, find paths on disk. Everything specific to the plugin's job (the model, the optimizer, the training loop) is the plugin's own.

## import path

Plugins import the surface from `veritate.plugin`:

```python
from veritate.plugin import save, paths, model, qat
```

`save`, `paths`, `model`, and `qat` are the namespaces in this contract. Nothing else in the parent repo is callable from a plugin.

## save

### `save.save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None) -> str`

Writes one checkpoint and runs the full dump suite (probe, lens, classroom, grades, concepts, surprise, quant_kl, generation). Returns the absolute path of the `.pt` file written.

| arg | type | meaning |
|---|---|---|
| `model` | `torch.nn.Module` | the model being trained, on its device. Must expose a vanilla `state_dict()`. |
| `name` | `str` | model dir name; must validate per [training/model_naming.md](../training/model_naming.md). |
| `step` | `int` | training step number. |
| `optimizer` | optional | optimizer whose `state_dict()` gets embedded in the `.pt`. |
| `args` | optional `dict` | training args. Must contain a non-empty `description` if `config.json` doesn't already exist. |
| `prompt` | optional `str` | canonical probe prompt for `dump_probe` / `dump_generation` / `dump_surprise` / `dump_quant_kl`. Defaults to `PROBE_PROMPT`. |
| `dump_set` | optional iterable | set of dump names to skip (`"probe"`, `"lens"`, `"classroom"`, `"grades"`, `"concepts"`, `"surprise"`, `"quant_kl"`, `"generation"`). |

Side effects: writes `models/<name>/checkpoints/step_<N>.pt`, `models/<name>/hooks/step_<N>/<artifact>`, and bootstraps `models/<name>/config.json` if missing.

Failures inside individual dump functions are logged and swallowed; the checkpoint is still written. A failure validating the model name or the description raises.

### `save.append_train_row(name, step, split, loss, *, lr=None, grad_norm=None, tok_per_s=None, wall_s=None, seed=None)`

Appends one row to `models/<name>/train.csv`. Writes the header if the file is new.

| column | type | required |
|---|---|---|
| `step` | int | yes |
| `split` | `"train"` or `"val"` | yes |
| `loss` | float | yes |
| `lr` | float | no |
| `grad_norm` | float | no |
| `tok_per_s` | float | no |
| `wall_s` | float | no |
| `seed` | int | no |

Cheap, called every step. The dashboard renders the loss chart from this file.

### `save.compose_name(corpus, size, precision, version) -> str`

Builds the canonical model name: `<corpus_leaf>_<size>_<precision>_<version>`. If `corpus` contains a `:` (bundled-corpus form like `multimind_m3:pg19`), only the part after the last `:` is used.

### `save.hash_corpus(stem) -> dict`

Hashes the train (and val, if present) `.bin` files for a corpus stem. Returns:

```python
{
    "stem":         "<stem>",
    "train_sha256": "<hex>",
    "train_bytes":  <int>,
    "val_sha256":   "<hex>",   # present only if a val file exists
    "val_bytes":    <int>,     # present only if a val file exists
}
```

Used to record the exact training data fingerprint in the model's `config.json` so two models can be compared honestly.

### `save.require_description(desc) -> str`

Returns the trimmed description, or raises `ValueError` if it's empty or non-string. Enforces ROE rule 6 ("every model has a description").

### `save.resolve_corpus(stem) -> (str, str | None)`

Returns `(train_path, val_path)` for a corpus stem. Searches the shared corpus folder (`plugins/corpus/`) and the calling plugin's bundled corpus folder (`plugins/<id>/corpus/`). Raises `FileNotFoundError` if no train file exists. `val_path` is `None` if there's no val file.

## model

### `model.Veritate(vocab, hidden, layers, ffn, heads, seq) -> torch.nn.Module`

The canonical byte-level decoder. One class, one shape, one place. Plugins train this; the inference Brain loads this; tools diff this.

| arg | meaning |
|---|---|
| `vocab` | must be `model.VOCAB_BYTE_LEVEL` (256). Construction raises if anything else is passed. |
| `hidden` | model dimension. Must be divisible by `heads`. |
| `layers` | number of transformer blocks. |
| `ffn` | FFN inner dimension. |
| `heads` | number of attention heads. |
| `seq` | maximum sequence length the position embedding supports. |

Architecture: pre-norm RMSNorm, sdpa causal self-attention with combined `qkv` projection, GELU FFN with separate `up` / `down` linears, learned positional embedding, tied LM head (`lm_head.weight = tok_emb.weight`). All linears `bias=False`.

State dict layout (keys you will see in checkpoints):

```
tok_emb.weight, pos_emb.weight,
blocks.<L>.n1.weight, blocks.<L>.attn.qkv.weight, blocks.<L>.attn.proj.weight,
blocks.<L>.n2.weight, blocks.<L>.ff.up.weight,    blocks.<L>.ff.down.weight,
n_out.weight,
lm_head.weight       (same tensor as tok_emb.weight, tied)
```

Methods:

- `forward(tokens, targets=None) -> (logits, loss_or_None)`. Standard CE loss when `targets` is given; `ignore_index=-1`.
- `embed(tokens) -> tensor`. Token + positional embedding lookup. Used by adapters that thread state across the residual stream.

`model.VOCAB_BYTE_LEVEL` is the constant 256. Use it instead of hardcoding.

If a plugin wraps `Veritate` (adapter, head, side network), the wrapper must expose `model.base` of type `Veritate` so `save.save(model.base, ...)` can dump against the standard layout.

## qat

Quantization-aware training helpers. Use these to train a model that exports cleanly to the C engine's INT8 format. The ops match the C engine's quantization scheme exactly, so a model trained with these wrapped into its forward pass will export to a v9 binary with `act_boost = 1` and run without quantization-noise gibberish.

The canonical `model.Veritate` already wires these in; flipping `qat=True` on the base activates them. Plugins only need to call the helpers directly when they have their own matmuls outside the base (e.g. an adapter sidecar) and want those quantized too.

### `qat.fake_quant_weight(w) -> tensor`

Per-tensor symmetric maxabs INT8 round-and-dequant of a weight tensor, with a straight-through gradient. Use to wrap matmul weights.

### `qat.fake_quant_act(x, scale=32.0) -> tensor`

Round `x * scale` to INT8, clip, divide back. Straight-through gradient. The default scale is the C engine's residual-stream scale; do not change it without coordinating with the engine and the exporter.

### `qat.fake_quant_ln_weight(w, scale=64.0) -> tensor`

Round-and-dequant for an RMSNorm weight at the C engine's fixed LN scale. Straight-through gradient.

### `qat.set_qat(module, value) -> module`

Recursively set `.qat = bool(value)` on every submodule that has the attribute. The canonical `model.Veritate`, its `Block`, `RMSNorm`, and `QuantLinear` all carry the flag. Returns the module for chaining.

### Constants

`qat.INT8_MAX` (127), `qat.ACT_INT8_SCALE` (32.0), `qat.LN_FIXED_SCALE` (64.0). Match `veritate_mri/export.py`.

## paths

`paths` is a pure read-only namespace. No side effects, no filesystem creation. Just string composition over the on-disk layout the platform owns.

| call | returns |
|---|---|
| `paths.model_dir(name)` | `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | path to one of the dump files (`"probe"`, `"lens"`, `"classroom"`, `"grades"`, `"concepts"`, `"surprise"`, `"quant_kl"`, `"generation"`) |
| `paths.corpus_dir()` | `plugins/corpus/` |
| `paths.corpus_train_path(stem)` | `plugins/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `plugins/corpus/<stem>_val.bin` |

Plugins should not assemble these paths themselves. If the platform reorganizes the layout, every plugin breaks unless they all go through `paths.*`.

## manifest format

Every plugin ships a `manifest.json` next to its `plugin.py`. The dashboard reads it to render the trainer form; the plugin's own `parse_args` reads it to bootstrap `argparse` defaults.

### top-level shape

```json
{
  "name":        "Human-Friendly Plugin Name",
  "description": "One-line summary shown in the trainer picker.",
  "kind":        "trainer",
  "flow":        ["scratch", "continue"],
  "defaults":    { "<key>": <value>, ... }
}
```

| field | type | meaning |
|---|---|---|
| `name` | string | display name in the dashboard's trainer list. |
| `description` | string | one-line summary; appears in the trainer picker tooltip and the form header. |
| `kind` | string | `"trainer"` is the only kind today. |
| `flow` | array of strings | which start flows this plugin supports. `"scratch"` (new model from random init or `init_from`) and `"continue"` (resume an existing model). |
| `defaults` | object | preset values for the form fields. Keys are the field names; values are the default (number, string, bool, or array as appropriate). The dashboard only renders fields whose name appears here (or that are required). |

### what the manifest does NOT carry

Labels, help text, types, choice lists, and `advanced` / `featured` flags live in the dashboard's `TRAINER_SCHEMA` (in `veritate_mri/static/index.html`), not the manifest. A plugin opts a field into its form by adding the field name to `defaults`; the dashboard pulls the rest from the schema. To introduce a brand-new field that isn't in the schema yet, update `TRAINER_SCHEMA` in the same commit you update the manifest.

### `defaults` — known field names

The schema groups fields into adapter-specific clusters and shared training-loop knobs. A plugin only needs to declare the fields it cares about — the dashboard hides the rest.

#### Required (every trainer)

| key | type | sample | meaning |
|---|---|---|---|
| `size` | string | `"120m"` | shape preset name; valid choices come from the plugin's own `SIZE_PRESETS`. |
| `precision` | string | `"bf16"` | training precision. `"bf16"` or `"fp32"`. |
| `version` | string | `"v1"` | version tag for `compose_name`. `v1`, `v1a`, `v2`, ... |

#### Common training-loop knobs

| key | type | sample | meaning |
|---|---|---|---|
| `total_steps` | int | `5000` | how many gradient updates to run. |
| `batch_size` | int | `8` | sequences per step. |
| `seq` | int | `256` | per-chunk sequence length. |
| `n_chunks` | int | `48` | number of `seq`-length chunks per step (per-step bytes = `seq * n_chunks`). |
| `bptt_window` | int | `4` | how many chunks of past activations carry gradient. Only meaningful for adapters with cross-chunk recurrent state (M3, M1). MEGA omits. |
| `base_lr` | float | `0.0001` | peak learning rate after warmup. |
| `min_lr` | float | `1e-05` | floor LR at the end of the schedule. |
| `warmup_steps` | int | `200` | linear warmup from 0 to `base_lr` over this many steps. |
| `lr_schedule` | string | `"cosine"` | post-warmup curve. `"cosine"`, `"linear"`, or `"constant"`. |
| `weight_decay` | float | `0.01` | AdamW weight decay. |
| `beta1` | float | `0.9` | AdamW first-moment decay. |
| `beta2` | float | `0.95` | AdamW second-moment decay. `0.95` for LM training, `0.999` for general use. |
| `label_smoothing` | float | `0.0` | cross-entropy label smoothing. |
| `grad_clip` | float | `1.0` | per-step gradient-norm cap. |
| `ckpt_every` | int | `200` | save a checkpoint every N steps. |
| `log_every` | int | `20` | append a `train.csv` row every N steps. |
| `eval_every` | int | `200` | run a validation pass every N steps. |
| `eval_iters` | int | `8` | batches per validation pass. |
| `seed` | int | `0` | RNG seed for the corpus loader and weight init. |
| `use_act_ckpt` | bool | `true` | wrap each block with `torch.utils.checkpoint` to trade compute for activation VRAM. |

#### Reserved feature flags

(See *reserved manifest flags* below for the full contract.)

| key | type | sample | meaning |
|---|---|---|---|
| `qat_enabled` | bool | `false` | INT8 QAT for the canonical Veritate path. MEGA additionally honors `quant_mode`. |

#### M1 / M3 adapter clusters

(Only for plugins that use them.)

| key | type | sample | applies to | meaning |
|---|---|---|---|---|
| `rank` | int | `32` | M3 | low-rank adapter rank. |
| `n_slots` | int | `256` | M1 | named slot vectors in the working-memory table. |
| `alpha` | float | `0.2` | M1, M3 | per-token write strength to the adapter state. |
| `inject_layer` | int | `-1` | M1, M3 | which layer the adapter attaches to; `-1` = auto (mid-stack). |
| `init_from` | string | `""` | M1 | name of an existing model whose latest checkpoint seeds the base. New model is named `<init_from>_m1`. |
| `freeze_base` | bool | `false` | M1 | freeze the base; only the adapter trains. |

#### MEGA cluster

(Ternary + MoE moonshot.)

| key | type | sample | meaning |
|---|---|---|---|
| `quant_mode` | string | `"ternary"` | weight quant scheme. `"int8"` (1 byte/param), `"int4"` (2x density), or `"ternary"` (BitNet b1.58, 5x density). |
| `n_experts` | int | `8` | FFN experts per block. Total params scale linearly. |
| `router_topk` | int | `1` | experts active per token. `1` = sticky single-expert routing (cheapest, L3-fittest). |
| `router_aux_loss` | float | `0.01` | Switch-Transformer load-balance coefficient; prevents router collapse. |
| `use_8bit_adam` | bool | `true` | use `bitsandbytes.optim.AdamW8bit` to fit larger models on small GPUs. |

### example: minimum trainer manifest

A standalone scratch-only trainer with the bare minimum surface:

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

### example: adapter trainer (M1-style)

Adds the M1 adapter cluster on top of the minimum surface:

```json
{
  "name": "Example M1 Adapter Trainer",
  "description": "Adapter on top of an existing base model.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": {
    "size": "120m",
    "precision": "bf16",
    "version": "v1",
    "n_slots": 256,
    "alpha": 0.2,
    "inject_layer": -1,
    "init_from": "",
    "freeze_base": true,
    "total_steps": 5000,
    "batch_size": 8,
    "seq": 256,
    "n_chunks": 48,
    "bptt_window": 4,
    "base_lr": 0.0003,
    "min_lr": 1e-05,
    "warmup_steps": 100,
    "lr_schedule": "cosine",
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "label_smoothing": 0.0,
    "grad_clip": 1.0,
    "ckpt_every": 200,
    "log_every": 20,
    "eval_every": 200,
    "eval_iters": 8,
    "seed": 0,
    "use_act_ckpt": true,
    "qat_enabled": false
  }
}
```

### example: ternary + MoE moonshot trainer (MEGA-style)

```json
{
  "name": "Example MEGA Trainer",
  "description": "Ternary + MoE base for L3-fit at 1B-class total params.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": {
    "size": "1b",
    "precision": "bf16",
    "version": "v1",
    "quant_mode": "ternary",
    "n_experts": 8,
    "router_topk": 1,
    "router_aux_loss": 0.01,
    "use_8bit_adam": true,
    "total_steps": 5000,
    "batch_size": 2,
    "seq": 256,
    "n_chunks": 12,
    "base_lr": 0.0001,
    "min_lr": 1e-05,
    "warmup_steps": 200,
    "lr_schedule": "cosine",
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "label_smoothing": 0.0,
    "grad_clip": 1.0,
    "ckpt_every": 250,
    "log_every": 20,
    "eval_every": 250,
    "eval_iters": 4,
    "seed": 0,
    "use_act_ckpt": true,
    "qat_enabled": true
  }
}
```

## reserved manifest flags

A subset of `defaults` keys are reserved: their meaning, dashboard treatment, and downstream side-effects are fixed across plugins. Trainers must not invent near-synonyms (e.g. `int8_qat`, `quantize`) — use the reserved key. Reserved keys are stable across patch releases.

The point of the reservation is consistency. One trainer's `qat_enabled` checkbox and another's are the same checkbox on the dashboard, set the same field in `config.json`, and feed the same downstream signals (the Generation tab's Veritate warning, the bin reader's `act_boost`, the export pipeline).

| key | type | required behavior when set true | dashboard treatment | downstream contract |
|---|---|---|---|---|
| `qat_enabled` | bool | Wrap matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops using a straight-through estimator on backprop. Default scheme is per-tensor maxabs INT8 with scale-32 activation quant and scale-64 RMSNorm-weight quant. Plugins that declare `quant_mode` may substitute `"int4"` or `"ternary"` for the *weight* quant; activation and RMSNorm scales are always INT8. Match the C engine's quant scheme exactly so the trained checkpoint exports to a v9 binary with `act_boost=1`. Applies to both `scratch` and `continue` flows. | Checkbox labeled **QAT enabled**, in the featured row. | Trainer must set `args["training"] = "qat"` so `save.save` records it in `config.json`. The Veritate warning on the Generation tab keys off this field plus the v9 binary's `act_boost`. |
| `quant_mode` | string | Selects the weight quant scheme used by `qat.fake_quant_weight_mode` and the `QuantLinear.quant_mode` flag. `"int8"` is the canonical Veritate scheme; `"int4"` packs 2 weights per byte; `"ternary"` is BitNet b1.58 (3 levels, ~1.58 bits/param, 5x density vs INT8). Activation and RMSNorm quant remain INT8 regardless of mode. | Dropdown labeled **weight quant mode**. Visible only when the plugin's manifest declares the field. | Trainer must record the chosen mode in `config.json` so the exporter and the C engine pick the matching kernel. |
| `use_8bit_adam` | bool | Construct the optimizer as `bitsandbytes.optim.AdamW8bit` instead of `torch.optim.AdamW`. INT8 storage for the AdamW first/second moment buffers; fp32 master weights are still kept by bnb. Required to fit ~1B-class MoE training on 12 GB-class consumer GPUs. | Checkbox labeled **8-bit AdamW (bitsandbytes)**, in the featured row. | The trainer must `import bitsandbytes` only when the flag is set so plugins that don't need it never pull the dependency. |

Adding a new reserved flag follows the [update obligation](#update-obligation): implementation, table row, dashboard render, and any shipped plugin that should expose it land in the same commit.

## what plugins must not do

- Do not import from `veritate_mri.*` directly. Use `veritate.plugin.save` and `veritate.plugin.paths`.
- Do not write outside `models/<name>/`. The dashboard reads from a fixed layout; writing elsewhere is invisible to it.
- Do not edit `config.json` after `save()` has bootstrapped it, except via fields the contract defines.
- Do not invent your own dump artifacts. The dashboard only renders the eight in `HOOK_ARTIFACTS`. Add a real one through the [hooks contract](../hooks/contract.md) update process if you need a new field.

## stability

Functions and field names in this file are stable across patch releases of the platform. A signature change or rename is a minor-version bump and requires this file's update obligation to be honored in the same commit. Plugins targeting one version are expected to keep working until the next minor bump.

## update obligation

Adding, removing, or renaming any function or field above requires:

1. The implementation in `veritate_mri/save.py` or `veritate_mri/readers/paths.py` is updated.
2. The re-export in `veritate/plugin/__init__.py` is updated.
3. This file's table is updated in the same commit.
4. Any plugin shipped with the platform (`plugins/multimind_m3/`, etc.) is updated and verified in the same commit.
