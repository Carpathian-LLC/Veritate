---
title: hooks contract
date: 2026-05-05
tags: [hooks, contract, tfrm, dump]
summary: What the dashboard, MRI server, and external tools can pull from a training run or a live forward.
---

> Friendly summary. The canonical, field-level contract is `documentation/hooks/contract.md`.

## the thirteen per-checkpoint artifacts

Every artifact is produced by a single call to `save.save(model, name, step, ...)`. Paths are relative to `models/<name>/hooks/step_<N>/`. The dashboard's hook-reader walks the canonical filenames in `readers/paths.py::HOOK_ARTIFACTS`; `save.RENAME_MAP_TEMPLATE` performs the `*_step_<N>.*` → canonical rename after each dump produces its prefixed file.

| file | producer | what it carries |
|---|---|---|
| `probe.json` | `dump_probe` | top-K active neurons per layer for a fixed prompt |
| `lens.npz` | `dump_probe` | logit-lens projections per layer + residual L2 norms |
| `classroom.json` | `dump_classroom` | parameter count, INT8/INT4 byte sizes, weight-delta L2, alive neurons per layer |
| `grades.json` | `dump_grades` | reading-grade ladder per level + estimated grade |
| `math.json` | `dump_math` | arithmetic rubric scores |
| `grammar.json` | `dump_grammar` | grammar-eval rubric scores |
| `reasoning.json` | `dump_reasoning` | reasoning-eval rubric scores |
| `concepts.json` | `dump_concepts` | concept → top-neurons mapping per layer |
| `surprise.json` | `dump_surprise` | per-token bits/byte for the canonical prompt |
| `quant_kl.json` | `dump_quant_kl` | KL between fp and quantized output distributions |
| `writing_health.json` | `dump_writing_health` | writing-style telemetry (repetition, vocab spread) |
| `reading_comprehension.json` | `dump_reading_comprehension` | multi-prompt comprehension rubric scores |
| `generation.json` | `dump_generation` | full TFRM v7 frames, one per generated token |

## the TFRM v7 frame

One frame per generated token. Same shape from training-time `dump_generation` and live inference, so the dashboard's render path consumes one shape from both sources.

| field | shape | meaning |
|---|---|---|
| `kind` | string | always `"token"` |
| `byte`, `argmax_byte` | u8 | sampled byte; what greedy would have picked |
| `T`, `fwd_ms` | int, float | total positions, wall-clock for this forward |
| `entropy_bits`, `surprise_bits` | float | softmax entropy; -log2 p(byte) at the sampled byte |
| `ffn_full`, `ffn_argmax` | u8 `[L][buckets]` | bucketed FFN activations and which neuron fired hardest per bucket |
| `ffn_top` | `[L][k]` of `{id, v}` | global top-K neurons per layer |
| `ffn_downsample` | int | bucket width |
| `saturation` | float `[L]` | fraction clipping at INT8 budget per layer |
| `attn` | `[L][heads]` of `{ent, top: [{p, w}]}` | per-head attention top positions and entropy |
| `info_flow` | `[{p, w}]` | top-K positions across layers, normalized |
| `res`, `contrib` | float `[L]` | residual L2 per layer; per-layer L2 of `(residual_post − residual_pre)` |
| `lens` | `[L][3]` of `{b, p}` | logit-lens top-3 bytes per layer |
| `cand` | `[{b, p}]` | top-K next-byte candidates from full logits |
| `decisiveness` | float `[L]` | per-layer max_abs / mean_abs of logit-delta projection |
| `dla_picked`, `dla_argmax` | `[{layer, neuron, act, w, contrib}]` | direct logit attribution for the sampled byte / argmax byte |
| `dla_cand` (v8) | `[cands]` of dla list | DLA per top-K candidate. Element `i` corresponds to `cand[i].b` — answers "why did it almost say X". |
| `ablation` (v8) | {layer, neuron} or null | echo of any active ablation request for this token. UI metadata only. |
| `margin`, `entropy`, `lens_consistency`, `residual_stab`, `confidence` | float | calibration components and the calibrated `confidence` ∈ [0,1] |
| `memory` | [{text, score, peak_pos}] | neuron-memory hits |
| `backend` | string | `"c"`, `"pytorch"`, or `"training"` |

## what is pullable

| mode | source | endpoint or path |
|---|---|---|
| live inference | TFRM frame stream | `GET /generate?prompt=...&backend=<c-or-pytorch>` |
| in-process training | TFRM frame per checkpoint | `models/<name>/hooks/step_<N>/generation.json` |
| past-checkpoint learning | all thirteen artifacts per saved step | `models/<name>/hooks/step_<N>/{probe,lens,classroom,grades,math,grammar,reasoning,concepts,surprise,quant_kl,writing_health,reading_comprehension,generation}.{json,npz}` |
| distillation | the thirteen artifacts on teacher and student at the same step | `models/<teacher>/...` and `models/<student>/...` |
| live training stream | per-step TFRM-lite over SSE | `GET /train_stream` (opt-in via the streaming helper) |
| ablation | inference forward with one FFN neuron zeroed at the kernel boundary | `GET /generate?...&ablate_layer=L&ablate_neuron=N` |
| ablation replay | same but loads a saved checkpoint instead of the latest bin | `GET /generate?...&checkpoint=step_<N>&ablate_layer=L&ablate_neuron=N` |

## interpretability endpoints

Atlas endpoints aggregate frames from the in-memory ring (live) or `step_<N>.json` (past). Backed by one module (`veritate_mri/atlas.py`) that consumes a frame iterator + query and returns aggregated stats. No new TFRM fields, no new dump artifacts: pure derivations.

| endpoint | input | output |
|---|---|---|
| `GET /atlas/concept/<substring>?model=...&step=<N>` | byte-substring filter | top-N `(layer, neuron)` ranked by aggregated `dla_picked` contribution |
| `GET /atlas/neuron/<layer>/<index>?model=...&step=<N>` | neuron coordinate | inverse map: which output substrings, attention positions, and concepts coincide with this neuron's activation |
| `GET /atlas/lifetime/<layer>/<index>?model=...` | neuron coordinate | per-step trajectory across `probe_step_<N>.json` (rank, magnitude, prompt-position) |
| `GET /atlas/circuit?model=...` | model name | static `W_down[L] @ W_up[L+1]` neuron-to-neuron transfer matrix; computed once at model load |
| `GET /atlas/concepts_inverted?model=...&step=<N>` | model + step | `concepts_step_<N>.json::top_neurons` inverted into a neuron-keyed map |

## update obligation

Adding, removing, or renaming any TFRM field above requires:

1. The producer in `veritate_mri/checkpoint_probe.py` is updated.
2. The canonical contract table is updated in the same commit.
3. If a TFRM frame field is added, both the C engine emit and `dump_generation` are updated in the same commit.
4. The MRI dashboard render function gates on field presence so old runs do not break.
