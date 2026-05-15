# trainers contract

The platform surface trainer plugins are allowed to call. Anything not listed here is internal: trainers must not reach into it, and the platform may rename or restructure it without notice.

Adding, removing, or changing the signature of any function or field below requires updating this file in the same commit. Same rule as [hooks/contract.md](../hooks/contract.md).

## scope

A trainer plugin is a folder under `trainers/<name>/` containing a `trainer.py` script and a `manifest.json`. The platform exposes a small set of helpers for the things every trainer needs to do consistently: name a model, hash its corpus, write a checkpoint, log a training row, find paths on disk. The model, the optimizer, and the training loop are the trainer's own.

## import path

Trainers import the surface from `veritate_core.plugin`:

```python
from veritate_core.plugin import save, paths, model, qat, get_teacher_client
```

`save`, `paths`, `model`, and `qat` are the namespaces in this contract. `get_teacher_client(provider=None, model=None)` returns a configured teacher-model `Client` for distillation, or `None` when no `teacher_provider` is set in settings. Nothing else in the parent repo is callable from a trainer.

## save

### `save.save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None) -> str`

Writes one checkpoint and runs the full dump suite. Returns the absolute path of the `.pt` file written.

| arg | type | meaning |
|---|---|---|
| `model` | `torch.nn.Module` | the model being trained, on its device. Must expose a vanilla `state_dict()`. |
| `name` | `str` | model dir name. Must match `^[a-z0-9](?:[a-z0-9_]*[a-z0-9])?$`. |
| `step` | `int` | training step number. |
| `optimizer` | optional | optimizer whose `state_dict()` gets embedded in the `.pt`. |
| `args` | optional `dict` | training args. `vars(args)` from the trainer's argparse is the canonical input. |
| `prompt` | optional `str` | canonical probe prompt for the probe / generation / surprise / quant_kl dumps. Defaults to the platform's `PROBE_PROMPT`. |
| `dump_set` | optional iterable | set of dump names to skip. Names below. |

Side effects:

- Writes `models/<name>/checkpoints/step_<N>.pt` (atomic: written as `.pt.tmp`, then renamed).
- Writes `models/<name>/hooks/step_<N>/<artifact>.<ext>` for each dump.
- Bootstraps `models/<name>/config.json` from `args` if missing.
- Mirrors `args["qat_enabled"]` to a top-level `qat_enabled` in `config.json` on every save.

Description handling: every model carries a one-line description in `config.json`. The trainer may pass `args["description"]`; otherwise `save` auto-builds one from the args dict (corpus, size, precision, version, shape, training mode, seed). If nothing usable is in args and no description is already in `config.json`, `save` raises.

Failures inside individual dump functions are logged and swallowed; the checkpoint still lands. A failure validating the model name or building a description raises.

### dump artifacts

`save.save` writes the artifacts below into `models/<name>/hooks/step_<N>/`. Pass any subset of these names in `dump_set` to skip them.

| name | filename | what it is |
|---|---|---|
| `probe` | `probe.json` | top-K firing neurons per layer on the canonical prompt |
| `lens` | `lens.npz` | logit-lens projections per layer |
| `classroom` | `classroom.json` | param count, INT8/INT4 byte budget, weight-delta L2, alive neurons |
| `grades` | `grades.json` | reading-grade rubric scores |
| `math` | `math.json` | arithmetic-eval rubric scores |
| `grammar` | `grammar.json` | grammar-eval rubric scores |
| `reasoning` | `reasoning.json` | reasoning-eval rubric scores |
| `concepts` | `concepts.json` | top concept neurons per layer |
| `surprise` | `surprise.json` | per-byte surprise on the canonical prompt |
| `quant_kl` | `quant_kl.json` | KL between fp32 logits and a quantised projection |
| `writing_health` | `writing_health.json` | repetition + vocab-spread telemetry |
| `reading_comprehension` | `reading_comprehension.json` | multi-prompt comprehension rubric scores |
| `generation` | `generation.json` | full per-byte frame stream for the canonical prompt |

Field schemas for every artifact live in [hooks/contract.md](../hooks/contract.md).

### `save.append_train_row(name, step, split, loss, *, lr=None, grad_norm=None, tok_per_s=None, wall_s=None, seed=None)`

Appends one row to `models/<name>/train.csv`. Writes the header if the file is new. Cheap, called every log step. The dashboard's loss, throughput, learning-rate, and gradient-norm charts all read from this one file.

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

### `save.truncate_train_csv_at(name, resume_step) -> int`

When resuming from `step_<N>.pt`, removes rows where `step > N` from `train.csv` so duplicate step numbers don't poison the loss curve. Returns the number of rows removed. Call once, immediately after loading the checkpoint, before the training loop starts.

### `save.compose_name(user_name, size) -> str`

Builds the canonical model dir name from a human label and a size token. The label is slugified (lowercased, non-alnum collapsed to `_`); size is lowercased and stripped. Example: `compose_name("Chatty Otter", "85m")` returns `"chatty_otter_85m"`.

Legacy 4-arg form `compose_name(corpus, size, precision, version)` returning `"<corpus_leaf>_<size>_<precision>_<version>"` is kept for older trainers. New trainers use the 2-arg form.

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

The hash is cached next to each `.bin` as `<bin>.sha256` and invalidated on mtime+size change; running this every step is cheap. Record the result in `config.json` to fingerprint the training data.

### `save.require_description(desc) -> str`

Returns the trimmed description, or raises `ValueError` if it's empty or non-string.

### `save.resolve_corpus(stem) -> (train_path, val_path)`

Returns `(train_path, val_path)` for a corpus stem. Searches the shared corpus folder (`trainers/corpus/`) first, then the calling trainer's bundled corpus folder (`trainers/<id>/corpus/`). Raises `FileNotFoundError` if no train file exists. `val_path` is `None` when no val file exists.

## model

### `model.Veritate(vocab, hidden, layers, ffn, heads, seq, *, activation="gelu", capture_l1=False) -> torch.nn.Module`

The canonical byte-level decoder. One class, one shape. Trainers train this; the inference Brain loads this; tools diff this.

| arg | meaning |
|---|---|
| `vocab` | must be `model.VOCAB_BYTE_LEVEL` (256). Construction raises otherwise. |
| `hidden` | model dimension. Must be divisible by `heads`. |
| `layers` | number of transformer blocks. |
| `ffn` | FFN inner dimension. May be an int (uniform) or a per-layer list whose length equals `layers`. |
| `heads` | number of attention heads. |
| `seq` | maximum sequence length the position embedding supports. |
| `activation` | one of `"gelu"`, `"relu"`, `"silu"`. ReLU and SiLU expose post-activation sparsity for the L1 capture path; GELU does not. |
| `capture_l1` | when true, each FFN stores the post-activation L1 mean for the optional sparsity penalty. |

Architecture: pre-norm RMSNorm, sdpa causal self-attention with combined `qkv` projection, GELU/ReLU/SiLU FFN with separate `up` / `down` linears, learned positional embedding, tied LM head (`lm_head.weight = tok_emb.weight`). All linears `bias=False`.

State dict layout:

```
tok_emb.weight, pos_emb.weight,
blocks.<L>.n1.weight, blocks.<L>.attn.qkv.weight, blocks.<L>.attn.proj.weight,
blocks.<L>.n2.weight, blocks.<L>.ff.up.weight,    blocks.<L>.ff.down.weight,
n_out.weight,
lm_head.weight       (same tensor as tok_emb.weight, tied)
```

Contract methods (inference code dispatches on these without branching on the variant class):

| method | meaning |
|---|---|
| `forward(tokens, targets=None) -> (logits, loss_or_None)` | standard CE loss when `targets` is given; `ignore_index=-1`. |
| `embed(tokens, start_pos=0) -> tensor` | token + positional embedding lookup. Used by adapters that thread state across the residual. |
| `run_blocks(x, start_pos=0, exit_after=None) -> tensor` | run the transformer stack. `exit_after` truncates for early-exit experiments. |
| `run_block(x, L, start_pos=0) -> tensor` | run a single block at index `L`. |
| `project_byte0(residual) -> logits` | apply final norm + LM head. Inference calls this blindly; new variants override to route through their own head. |
| `supports_mtp_decode() -> bool` | reports whether this model can produce multi-token-prediction draft tokens. |
| `hook_spec() -> nn.Module` | returns a canonical-shaped view for the dump suite. Canonical `Veritate` returns `self`; wrappers (MoE, sidecars) return a canonical adapter. |
| `set_qat(value) -> self` | flips the `.qat` flag on every QAT-aware submodule. Equivalent to `qat.set_qat(self, value)`. |

`model.VOCAB_BYTE_LEVEL` is the constant 256. Use it instead of hardcoding.

If a trainer wraps `Veritate` (adapter, head, side network), the wrapper must expose `model.base` of type `Veritate` so `save.save(model.base, ...)` can dump against the standard layout. The wrapper's own state goes to a sidecar `.pt` next to the standard checkpoint.

## qat

Quantization-aware training helpers. Every op matches the C engine's quant scheme exactly, so a model trained with `qat=True` exports to an INT8 binary that the engine runs without quantization-noise gibberish.

The canonical `model.Veritate` already wires QAT into its forward pass; calling `model.set_qat(True)` (or `qat.set_qat(model, True)`) activates it. Trainers only call the helpers directly when they have their own matmuls outside the base (e.g. an adapter sidecar) and want those quantized too.

### `qat.fake_quant_weight(w) -> tensor`

Per-tensor symmetric maxabs INT8 round-and-dequant of a weight tensor, with a straight-through gradient. The canonical Veritate weight scheme.

### `qat.fake_quant_weight_int4(w) -> tensor`

Per-tensor symmetric maxabs INT4 (16 levels, `{-7..+7}`) with a straight-through gradient. 2x density vs INT8; engine packs 2 weights per byte.

### `qat.fake_quant_weight_ternary(w) -> tensor`

BitNet b1.58 ternary: per-tensor mean-abs scale, levels `{-1, 0, +1}`, straight-through gradient. ~1.58 bits/param, ~5x density vs INT8; engine packs 5 trits per byte.

### `qat.fake_quant_weight_mode(w, mode) -> tensor`

Dispatch by `mode` string. Accepts `qat.QUANT_MODE_INT8`, `qat.QUANT_MODE_INT4`, or `qat.QUANT_MODE_TERNARY`. Raises `ValueError` on anything else.

### `qat.fake_quant_act(x, scale=qat.ACT_INT8_SCALE) -> tensor`

Round `x * scale` to INT8, clip, divide back. Straight-through gradient. The default scale is the engine's residual-stream scale; do not change it without coordinating with the engine and the exporter.

### `qat.fake_quant_ln_weight(w, scale=qat.LN_FIXED_SCALE) -> tensor`

Round-and-dequant for an RMSNorm weight at the engine's fixed LN scale. Straight-through gradient.

### `qat.set_qat(module, value) -> module`

Recursively set `.qat = bool(value)` on every submodule that has the attribute. Returns the module for chaining.

### `qat.set_quant_mode(module, mode) -> module`

Recursively set `.quant_mode` on every submodule that has the attribute. Activation and RMSNorm quant stay INT8 regardless; only weight quant changes. Returns the module.

### `qat.set_engine_faithful(module, value) -> module`

Recursively set `.engine_faithful` on every submodule that has the attribute. When true together with `qat=True`, attention fake-quants q/k/v/out so the PyTorch forward matches the engine's INT8 attention path.

### constants

| name | value |
|---|---|
| `qat.INT8_MAX` | 127 |
| `qat.INT4_MAX` | 7 |
| `qat.ACT_INT8_SCALE` | 32.0 |
| `qat.LN_FIXED_SCALE` | 64.0 |
| `qat.QUANT_MODE_INT8` | `"int8"` |
| `qat.QUANT_MODE_INT4` | `"int4"` |
| `qat.QUANT_MODE_TERNARY` | `"ternary"` |
| `qat.QUANT_MODES` | `("int8", "int4", "ternary")` |

## paths

`paths` is a pure read-only namespace: no side effects, no filesystem creation. Just string composition over the on-disk layout the platform owns. Trainers must use these helpers rather than assemble paths by hand; if the platform reorganizes the layout, every trainer that goes through `paths.*` keeps working.

| call | returns |
|---|---|
| `paths.model_dir(name)` | absolute path to `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | path to one of the dump files. `artifact` is one of the names in the dump-artifacts table. |
| `paths.corpus_dir()` | `trainers/corpus/` |
| `paths.corpus_train_path(stem)` | `trainers/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `trainers/corpus/<stem>_val.bin` |

## manifest format

Every trainer ships a `manifest.json` next to its `trainer.py`. The dashboard reads it to render the trainer form; the trainer's own `parse_args` reads it to bootstrap `argparse` defaults.

### top-level shape

```json
{
  "name":        "Human-Friendly Trainer Name",
  "description": "One-line summary shown in the trainer picker.",
  "kind":        "trainer",
  "flow":        ["scratch", "continue"],
  "sizes":       { "<size>": { "layers": <int>, "hidden": <int>, "ffn": <int>, "heads": <int>, "params": <int> }, ... },
  "defaults":    { "<key>": <value>, ... }
}
```

| field | type | meaning |
|---|---|---|
| `name` | string | display name in the dashboard's trainer list. |
| `description` | string | one-line summary; appears in the trainer picker tooltip and the form header. |
| `kind` | string | `"trainer"` is the only kind today. |
| `flow` | string or array | which start flows this trainer supports. `"scratch"` (new model) and `"continue"` (resume an existing model). May be a single string or an array. |
| `sizes` | object | shape table for the size dropdown. Keys are size labels (`"80m"`, `"1b"`); values are dicts with `layers`, `hidden`, `ffn`, `heads`, `params`, optionally `active_params` for MoE. The size dropdown choices and the VRAM estimator both read from this. |
| `defaults` | object | preset values for the form fields. Keys are the field names; values are the default (number, string, bool, or array). The dashboard only renders fields whose name appears here (or that are required). |

### what the manifest does NOT carry

Labels, help text, types, choice lists, and `advanced` / `featured` flags live in the dashboard's `TRAINER_SCHEMA` (in `veritate_mri/web/index.js`), not the manifest. A trainer opts a field into its form by adding the field name to `defaults`; the dashboard pulls the rest from the schema. To introduce a brand-new field that isn't in the schema yet, update `TRAINER_SCHEMA` in the same commit that updates the manifest.

### required `defaults` keys

Every trainer's manifest declares these three keys:

| key | type | sample | meaning |
|---|---|---|---|
| `size` | string | `"80m"` | shape preset name; must be a key in the manifest's top-level `sizes` table. |
| `precision` | string | `"bf16"` | training precision. `"bf16"` or `"fp32"`. |
| `version` | string | `"v1"` | version tag for legacy `compose_name`. `v1`, `v1a`, `v2`, ... |

### common training-loop knobs

A trainer only declares the keys it cares about; the dashboard hides the rest.

| key | type | sample | meaning |
|---|---|---|---|
| `corpus` | string | `"fineweb"` | corpus stem fed to `resolve_corpus` / `hash_corpus`. Use the bare stem (no `_train.bin` suffix). |
| `total_steps` | int | `5000` | how many gradient updates to run. |
| `batch_size` | int | `8` | sequences per step. |
| `seq` | int | `256` | per-chunk sequence length. |
| `n_chunks` | int | `48` | number of `seq`-length chunks per step (per-step bytes = `seq * n_chunks`). |
| `bptt_window` | int | `4` | how many chunks of past activations carry gradient. Meaningful only when the trainer threads recurrent state across chunks. |
| `base_lr` | float | `0.0001` | peak learning rate after warmup. |
| `min_lr` | float | `1e-05` | floor LR at the end of the schedule. |
| `warmup_steps` | int | `200` | linear warmup from 0 to `base_lr` over this many steps. |
| `lr_schedule` | string | `"cosine"` | post-warmup curve. `"cosine"`, `"linear"`, `"constant"`, or `"wsd"`. |
| `wsd_decay_frac` | float | `0.1` | fraction of `total_steps` spent in the WSD decay tail. Used only when `lr_schedule == "wsd"`. |
| `wsd_decay_kind` | string | `"sqrt"` | WSD decay shape: `"sqrt"`, `"linear"`, or `"cosine"`. |
| `weight_decay` | float | `0.1` | AdamW weight decay. |
| `beta1` | float | `0.9` | AdamW first-moment decay. |
| `beta2` | float | `0.95` | AdamW second-moment decay. |
| `label_smoothing` | float | `0.0` | cross-entropy label smoothing. |
| `grad_clip` | float | `1.0` | per-step gradient-norm cap. |
| `ckpt_every` | int | `200` | save a checkpoint every N steps. |
| `log_every` | int | `20` | append a `train.csv` row every N steps. |
| `eval_every` | int | `200` | run a validation pass every N steps. |
| `eval_iters` | int | `8` | batches per validation pass. |
| `seed` | int | `0` | RNG seed for the corpus loader and weight init. |
| `use_act_ckpt` | bool | `true` | wrap each block with `torch.utils.checkpoint` to trade compute for activation VRAM. |

### adapter knobs

For trainers that add a low-rank or workspace adapter on top of a base.

| key | type | sample | meaning |
|---|---|---|---|
| `rank` | int | `32` | low-rank adapter rank. |
| `n_slots` | int | `256` | named slot vectors in the working-memory table. |
| `alpha` | float | `0.2` | per-token write strength to the adapter state. |
| `inject_layer` | int | `-1` | which layer the adapter attaches to; `-1` = auto (mid-stack). |
| `init_from` | string | `""` | name of an existing model whose latest checkpoint seeds the base. The new model is named `<init_from>_<suffix>`. |
| `freeze_base` | bool | `false` | freeze the base; only the adapter trains. |

### MoE knobs

For trainers whose FFN is replaced with N independent experts and a router.

| key | type | sample | meaning |
|---|---|---|---|
| `n_experts` | int | `8` | FFN experts per block. Total params scale linearly. |
| `router_topk` | int | `1` | experts active per token. `1` = sticky single-expert routing. |
| `router_aux_loss_coef` | float | `0.01` | Switch-Transformer load-balance coefficient; prevents router collapse. |

## reserved manifest flags

A subset of `defaults` keys are reserved: their meaning, dashboard treatment, and downstream side-effects are fixed across trainers. Don't invent near-synonyms (e.g. `int8_qat`, `quantize`) — use the reserved key. The reservation gives one consistent checkbox on the dashboard, one field name in `config.json`, and one signal for downstream consumers (the engine wiring, the bin-picker warning, the exporter).

`qat_enabled` is the authoritative QAT signal across the platform. `save.save` mirrors `training_args.qat_enabled` to a top-level `qat_enabled` key in `config.json` on every save; consumers read it via `readers.config.qat_enabled(name)` (which accepts either location) rather than re-parsing `training_args`. The bin's `act_boost` field is a magnitude heuristic and is not authoritative: legitimately QAT-trained checkpoints can still export with `act_boost > 1` when embeddings are small, so the engine subprocess spawns with `VERITATE_ALLOW_HIGH_ACT_BOOST=1` and the dashboard suppresses the act_boost warning when `qat_enabled` is true.

| key | type | required behavior when set true | dashboard treatment | downstream contract |
|---|---|---|---|---|
| `qat_enabled` | bool | Wrap matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops using a straight-through estimator on backprop. Default scheme: per-tensor maxabs INT8 weights, scale-32 INT8 activations, scale-64 INT8 RMSNorm weights. When `quant_mode` is also declared, that selects the weight scheme (`int8` / `int4` / `ternary`); activations and RMSNorm stay INT8. Applies to both `scratch` and `continue` flows. | Checkbox labeled **QAT enabled**, in the featured row. | Trainer must set `args["qat_enabled"] = True` so `save.save` records it in `config.json`. The dashboard's Generation tab suppresses the engine warning when this is true. |
| `quant_mode` | string | Selects the weight quant scheme used by `qat.fake_quant_weight_mode` and the `QuantLinear.quant_mode` flag. `"int8"` is the canonical scheme; `"int4"` packs 2 weights per byte; `"ternary"` is BitNet b1.58. Activation and RMSNorm quant remain INT8 regardless. | Dropdown labeled **weight quant mode**, visible when the trainer declares the field. | Trainer must record the chosen mode in `config.json` so the exporter and the engine pick the matching kernel. |
| `use_8bit_adam` | bool | Construct the optimizer as `bitsandbytes.optim.AdamW8bit` instead of `torch.optim.AdamW`. INT8 storage for the AdamW first/second moment buffers; fp32 master weights are still kept by bnb. Lets ~1B-class training fit on 12 GB-class consumer GPUs. | Checkbox labeled **8-bit AdamW (bitsandbytes)**, in the featured row. | Trainer must `import bitsandbytes` only when the flag is set so trainers that don't need it never pull the dependency. |

Adding a new reserved flag follows the [update obligation](#update-obligation): implementation, table row, dashboard render, and any shipped trainer that should expose it all land in the same commit.

## example manifests

### minimum trainer

A scratch-only trainer with the bare minimum surface:

```json
{
  "name": "Example Trainer (minimal)",
  "description": "Tiny scratch trainer used as a manifest template.",
  "kind": "trainer",
  "flow": ["scratch"],
  "sizes": {
    "30m": { "layers": 10, "hidden":  512, "ffn": 2048, "heads":  8, "params":  31000000 },
    "80m": { "layers": 12, "hidden":  768, "ffn": 3072, "heads": 12, "params":  85000000 }
  },
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

### adapter trainer

Adds the adapter cluster on top of the minimum surface:

```json
{
  "name": "Example Adapter Trainer",
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

### ternary + MoE trainer

```json
{
  "name": "Example Ternary MoE Trainer",
  "description": "Ternary weights + 8-expert top-1 MoE.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": {
    "size": "1b",
    "precision": "bf16",
    "version": "v1",
    "quant_mode": "ternary",
    "n_experts": 8,
    "router_topk": 1,
    "router_aux_loss_coef": 0.01,
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

## what trainers must not do

- Do not import from `veritate_mri.*` directly. Use the namespaces in `veritate_core.plugin`.
- Do not write outside `models/<name>/`. The dashboard reads from a fixed layout; writing elsewhere is invisible to it.
- Do not edit `config.json` after `save.save` has bootstrapped it, except via fields this contract defines.
- Do not invent new dump artifacts. The dashboard only renders the artifacts in the dump-artifacts table. Add new ones through the [hooks contract](../hooks/contract.md) update process.

## stability

Functions and field names in this file are stable across patch releases. A signature change or rename is a minor-version bump and requires this file's update obligation to be honored in the same commit. Trainers targeting one version keep working until the next minor bump.

## update obligation

Adding, removing, or renaming any function or field above requires:

1. Update the implementation in `veritate_mri/training/save.py`, `veritate_mri/readers/paths.py`, `veritate_core/model.py`, or `veritate_core/qat.py`.
2. Update the re-export in `veritate_core/plugin/__init__.py`.
3. Update the table in this file in the same commit.
4. Update every shipped trainer under `trainers/` that consumes the changed surface, in the same commit.
