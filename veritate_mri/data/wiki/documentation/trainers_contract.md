---
title: trainers contract
date: 2026-05-15
tags: [trainers, contract, save, paths, model, qat, manifest]
summary: What a trainer plugin is, what the platform gives it, and what its manifest looks like.
---

> Friendly summary. The canonical, signature-level contract is `documentation/trainers/contract.md`.

## what a trainer is

A script + a manifest under `trainers/<name>/`. The platform exposes a small set of helpers every trainer uses the same way: name a model, hash its corpus, write a checkpoint, log a training row, find paths on disk. The model, the optimizer, and the training loop are the trainer's own.

## the four namespaces

```python
from veritate_core.plugin import save, paths, model, qat, get_teacher_client
```

Anything else in the parent repo is internal and may move without notice.

| namespace | what it does |
|---|---|
| `save` | write checkpoints, run the dump suite, log training rows, hash corpora, validate names |
| `model` | the canonical `Veritate` byte-level decoder. One class, one shape. |
| `qat` | quantization-aware training ops that match the engine's INT8 scheme exactly |
| `paths` | pure read-only string composition over the on-disk `models/<name>/` layout |

`get_teacher_client(provider=None, model=None)` returns a configured teacher-model `Client` for distillation, or `None` when no `teacher_provider` is set in settings.

## save (the headline calls)

| call | does |
|---|---|
| `save.save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None)` | write `step_<N>.pt` and the full dump suite. Returns the `.pt` path. |
| `save.append_train_row(name, step, split, loss, *, lr=None, grad_norm=None, tok_per_s=None, wall_s=None, seed=None)` | append one row to `train.csv`. Cheap, called every log step. The loss chart, throughput chart, LR chart, and grad-norm chart all read from this file. |
| `save.truncate_train_csv_at(name, resume_step)` | drop rows where `step > resume_step` after loading a checkpoint, so a resumed run doesn't double-log step numbers |
| `save.compose_name(user_name, size)` | build `<slug>_<size>` from a human label and a size token (e.g. `chatty_otter_85m`). Legacy 4-arg `compose_name(corpus, size, precision, version)` is kept for older trainers. |
| `save.hash_corpus(stem)` | sha256 the train (and val) `.bin` files for a corpus stem so two models can be compared honestly. Cached next to each `.bin` and invalidated on mtime+size change. |
| `save.require_description(desc)` | trim a description string; raise if empty |
| `save.resolve_corpus(stem)` | return `(train_path, val_path)` from `trainers/corpus/` or the calling trainer's bundled `trainers/<id>/corpus/` |

`save.save` writes `models/<name>/checkpoints/step_<N>.pt` plus `models/<name>/hooks/step_<N>/<artifact>` and bootstraps `models/<name>/config.json` if missing. Failures inside individual dump functions are logged and swallowed; the checkpoint still lands. A failed name or description validation raises.

### dump artifacts

`save.save` writes 13 artifacts per step. Pass any subset in `dump_set` to skip them.

`probe`, `lens`, `classroom`, `grades`, `math`, `grammar`, `reasoning`, `concepts`, `surprise`, `quant_kl`, `writing_health`, `reading_comprehension`, `generation`.

Field schemas for every artifact live in [hooks/contract.md](../hooks/contract.md).

## model

`model.Veritate(vocab, hidden, layers, ffn, heads, seq, *, activation="gelu", capture_l1=False)` is the canonical byte-level decoder. Pre-norm RMSNorm. Causal SDPA attention with combined `qkv` projection. GELU / ReLU / SiLU FFN. Learned positional embedding. Tied LM head (`lm_head.weight = tok_emb.weight`). All linears `bias=False`. `vocab` must be `model.VOCAB_BYTE_LEVEL` (256); anything else raises.

Contract methods inference code dispatches on without branching on variant:

| method | meaning |
|---|---|
| `forward(tokens, targets=None)` | returns `(logits, loss_or_None)`. CE loss with `ignore_index=-1` when `targets` is given. |
| `embed(tokens, start_pos=0)` | token + positional embedding lookup |
| `run_blocks(x, start_pos=0, exit_after=None)` | run the transformer stack |
| `run_block(x, L, start_pos=0)` | run a single block |
| `project_byte0(residual)` | apply final norm + LM head |
| `supports_mtp_decode()` | reports MTP draft-token capability |
| `hook_spec()` | canonical-shaped view for the dump suite |
| `set_qat(value)` | flip the `.qat` flag recursively |

If a trainer wraps `Veritate` (adapter, head, side network), the wrapper must expose `model.base` of type `Veritate` so `save.save(model.base, ...)` dumps against the standard layout.

## qat

QAT ops whose schemes match the engine's quantization exactly. The canonical `model.Veritate` already wires these in; `model.set_qat(True)` activates them. Trainers call helpers directly only when they have their own matmuls outside the base.

| call | does |
|---|---|
| `qat.fake_quant_weight(w)` | per-tensor symmetric maxabs INT8 round-and-dequant on a weight tensor, straight-through gradient |
| `qat.fake_quant_weight_int4(w)` | per-tensor symmetric maxabs INT4 (`{-7..+7}`); engine packs 2 weights per byte |
| `qat.fake_quant_weight_ternary(w)` | BitNet b1.58 ternary (`{-1, 0, +1}`), per-tensor mean-abs scale |
| `qat.fake_quant_weight_mode(w, mode)` | dispatch by `qat.QUANT_MODE_INT8` / `QUANT_MODE_INT4` / `QUANT_MODE_TERNARY` |
| `qat.fake_quant_act(x, scale=qat.ACT_INT8_SCALE)` | INT8 quant on activations at the engine's residual-stream scale |
| `qat.fake_quant_ln_weight(w, scale=qat.LN_FIXED_SCALE)` | INT8 round-and-dequant for an RMSNorm weight at the engine's fixed LN scale |
| `qat.set_qat(module, value)` | recursively set `.qat` on every submodule that has the attribute |
| `qat.set_quant_mode(module, mode)` | recursively set `.quant_mode` (weight scheme only; activations and RMSNorm stay INT8) |
| `qat.set_engine_faithful(module, value)` | recursively flip `.engine_faithful`; with `qat=True`, fake-quants attention q/k/v/out |

Constants: `qat.INT8_MAX = 127`, `qat.INT4_MAX = 7`, `qat.ACT_INT8_SCALE = 32.0`, `qat.LN_FIXED_SCALE = 64.0`. Mode strings: `qat.QUANT_MODE_INT8`, `QUANT_MODE_INT4`, `QUANT_MODE_TERNARY`.

## paths

Pure string composition. No side effects, no filesystem creation. Trainers go through `paths.*` so a future layout change doesn't break every trainer.

| call | returns |
|---|---|
| `paths.model_dir(name)` | absolute path to `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | one of the 13 dump files |
| `paths.corpus_dir()` | `trainers/corpus/` |
| `paths.corpus_train_path(stem)` | `trainers/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `trainers/corpus/<stem>_val.bin` |

## manifest format

Every trainer ships a `manifest.json` next to its `trainer.py`. The dashboard reads it to render the trainer form; the trainer's `parse_args` reads it to bootstrap argparse defaults.

```json
{
  "name": "Human-Friendly Trainer Name",
  "description": "One-line summary shown in the trainer picker.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "sizes": { "80m": { "layers": 12, "hidden": 768, "ffn": 3072, "heads": 12, "params": 85000000 } },
  "defaults": { "size": "80m", "precision": "bf16", "version": "v1", ... }
}
```

Labels, help text, types, choice lists, and `advanced` / `featured` flags live in `TRAINER_SCHEMA` in `veritate_mri/web/index.js`, not the manifest. A trainer opts a field into the form by adding the field name to `defaults`. To add a brand-new field that isn't in the schema yet, update `TRAINER_SCHEMA` in the same commit.

### required fields

| key | type | sample |
|---|---|---|
| `size` | string | `"80m"` — shape preset name from the manifest's `sizes` table |
| `precision` | string | `"bf16"` or `"fp32"` |
| `version` | string | `"v1"`, `"v1a"`, `"v2"` — for legacy `compose_name` |

### common training-loop knobs

| key | sample | meaning |
|---|---|---|
| `corpus` | `"fineweb"` | corpus stem (no `_train.bin` suffix) |
| `total_steps` | `5000` | gradient updates |
| `batch_size`, `seq`, `n_chunks` | `8`, `256`, `48` | sequences per step, chunk length, chunks per step |
| `bptt_window` | `4` | chunks of past activations that carry gradient (when the trainer threads recurrent state across chunks) |
| `base_lr`, `min_lr`, `warmup_steps`, `lr_schedule` | `0.0001`, `1e-5`, `200`, `"cosine"` | LR schedule. Schedule options: `cosine`, `linear`, `constant`, `wsd` |
| `wsd_decay_frac`, `wsd_decay_kind` | `0.1`, `"sqrt"` | WSD decay tail length and shape |
| `weight_decay`, `beta1`, `beta2` | `0.1`, `0.9`, `0.95` | AdamW knobs |
| `label_smoothing`, `grad_clip` | `0.0`, `1.0` | training stability |
| `ckpt_every`, `log_every`, `eval_every`, `eval_iters` | `200`, `20`, `200`, `8` | cadence |
| `seed`, `use_act_ckpt` | `0`, `true` | RNG, activation checkpointing |

### reserved feature flags

These keys are reserved across all trainers. One trainer's `qat_enabled` checkbox and another's are the same checkbox on the dashboard, set the same field in `config.json`, and feed the same downstream signals.

`qat_enabled` is the authoritative QAT signal. `save.save` mirrors it from `training_args` to a top-level `qat_enabled` key on every save. Consumers read it via `readers.config.qat_enabled(name)` (which accepts either location). The bin's `act_boost` is magnitude-based and not authoritative: QAT-trained models can still export `act_boost > 1`, so the engine spawns with `VERITATE_ALLOW_HIGH_ACT_BOOST=1` and the dashboard hides the act_boost warning when `qat_enabled` is true.

| key | sample | what it does |
|---|---|---|
| `qat_enabled` | `false` | wrap matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops. Match the engine's quant scheme so the binary exports with `act_boost=1`. |
| `quant_mode` | `"ternary"` | weight quant scheme. `"int8"` (default), `"int4"` (2x density), `"ternary"` (BitNet b1.58, ~5x density). Activation and RMSNorm quant stay INT8 regardless. |
| `use_8bit_adam` | `true` | `bitsandbytes.optim.AdamW8bit` instead of `torch.optim.AdamW`. INT8 storage for the moments. Required to fit ~1B-class training on 12 GB GPUs. Trainer must `import bitsandbytes` only when set, so trainers that don't need it never pull the dep. |

### adapter knobs

For trainers that add a low-rank or workspace adapter on top of a base.

| key | sample | meaning |
|---|---|---|
| `rank` | `32` | low-rank adapter rank |
| `n_slots` | `256` | named slot vectors in the working-memory table |
| `alpha` | `0.2` | per-token write strength to the adapter state |
| `inject_layer` | `-1` | which layer the adapter attaches to; `-1` = mid-stack |
| `init_from` | `""` | name of an existing model whose latest checkpoint seeds the base |
| `freeze_base` | `false` | freeze the base; only the adapter trains |

### MoE knobs

For trainers whose FFN is replaced with N independent experts and a router.

| key | sample | meaning |
|---|---|---|
| `n_experts` | `8` | FFN experts per block |
| `router_topk` | `1` | experts active per token |
| `router_aux_loss_coef` | `0.01` | Switch-Transformer load-balance coefficient |

## what trainers must not do

- Import from `veritate_mri.*` directly. Use `veritate_core.plugin.save` and `veritate_core.plugin.paths`.
- Write outside `models/<name>/`. The dashboard reads from a fixed layout; writes elsewhere are invisible.
- Edit `config.json` after `save.save` has bootstrapped it, except via fields the contract defines.
- Invent your own dump artifacts. The dashboard renders the 13 in the dump-artifacts table. Add real ones via the hooks contract update process.

## stability

Function and field names here are stable across patch releases. A signature change or rename is a minor-version bump and requires the canonical contract's update obligation to be honored in the same commit.
