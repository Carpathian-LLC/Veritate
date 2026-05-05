---
title: hooks contract
date: 2026-05-05
tags: [hooks, contract, tfrm, dump]
summary: API reference for the dump artifacts and TFRM v7 frame field index.
---

> Source: `documentation/hooks/contract.md` (mirrored copy; the file at that path remains the canonical contract).

# hooks contract

API reference for the data the dashboard, MRI server, and external tools can pull. Every artifact and field listed here is part of the contract. Adding, removing, or renaming a field requires updating this file in the same commit.

## artifacts (training-time, per checkpoint)

All paths relative to `models/<name>/hooks/step_<N>/`. Every artifact below is produced by a single call to `veritate_mri/save.py::save(model, name, step, ...)`, exposed to plugins as `veritate.plugin.save.save`.

| file | producer | fields |
|---|---|---|
| `probe_step_<N>.json` | `dump_probe` | `step`, `precision`, `prompt`, `top_k`, `layers[]: {layer, neurons[]: {id, v}}` |
| `lens_step_<N>.npz` | `dump_probe` | `lens_logits` int32 [layers, vocab], `residual_norms` fp32 [layers] |
| `classroom_step_<N>.json` | `dump_classroom` | `step`, `precision`, `params`, `int8_bytes`, `int4_bytes`, `weight_delta_l2`, `alive_neurons_per_layer`, `time_s` |
| `grades_step_<N>.json` | `dump_grades` | `step`, `precision`, `grades: {level: {ppl, n_bytes}}`, `estimated_reading_grade`, `time_s` |
| `concepts_step_<N>.json` | `dump_concepts` | `step`, `precision`, `concepts`, `top_k_per_layer`, `time_s` |
| `surprise_step_<N>.json` | `dump_surprise` | `step`, `precision`, `prompt`, `tokens`, `surprise` (bits/byte per token), `time_s` |
| `quant_kl_step_<N>.json` | `dump_quant_kl` | `step`, `precision`, `quant_kl_bits`, `n_levels`, `time_s` |
| `step_<N>.json` | `dump_generation` | `meta`, `frames[]` (each frame carries the full TFRM v7 field set below) |

## tfrm v7 frame fields (per generated token)

Emitted by both training-time `dump_generation` and inference-time chat. The dashboard's render path consumes one shape from both sources.

| field | shape | meaning |
|---|---|---|
| `kind` | string | always `"token"` |
| `byte` | u8 | sampled byte |
| `argmax_byte` | u8 | byte the model would have picked greedy |
| `T` | int | total positions including this one |
| `fwd_ms` | float | wall-clock for this forward |
| `entropy_bits` | float | softmax entropy of next-byte distribution |
| `surprise_bits` | float | -log2 p(byte) at the sampled byte |
| `ffn_full` | `u8 [layers][buckets]` | bucketed FFN activations per layer |
| `ffn_argmax` | `u8 [layers][buckets]` | which neuron in each bucket fired hardest |
| `ffn_top` | `[layers][k]: {id, v}` | global top-K neurons per layer |
| `ffn_downsample` | int | bucket width |
| `saturation` | `float [layers]` | fraction of post-GELU activations clipping at INT8 budget |
| `attn` | `[layers][heads]: {ent, top: [{p, w}]}` | per-head attention top positions and entropy |
| `info_flow` | `[{p, w}]` | top-K positions across layers, normalized |
| `res` | `float [layers]` | residual L2 norm per layer |
| `contrib` | `float [layers]` | per-layer L2 of (residual_post - residual_pre) |
| `lens` | `[layers][3]: {b, p}` | logit lens top-3 bytes per layer |
| `cand` | `[{b, p}]` | top-K next-byte candidates from full logits |
| `decisiveness` | `float [layers]` | per-layer max_abs / mean_abs of logit-delta projection |
| `dla_picked` | `[{layer, neuron, act, w, contrib}]` | direct logit attribution for sampled byte |
| `dla_argmax` | `[{layer, neuron, act, w, contrib}]` | direct logit attribution for argmax byte |
| `dla_cand` | `[cands][{layer, neuron, act, w, contrib}]` | direct logit attribution per top-K candidate byte (v8). Length matches `cand`. Element i is the DLA for `cand[i].b`. Answers the question "why did it almost say X". |
| `ablation` | `{layer, neuron}` or null | echo of any active ablation request for this token (v8). Null when no ablation. UI metadata only, not a derivation. |
| `margin` | float | logit gap between top-1 and top-2 |
| `entropy` | float | entropy of next-byte distribution (nats) |
| `lens_consistency` | float | per-layer lens agreement with final |
| `residual_stab` | float | residual stability score |
| `confidence` | float [0..1] | calibrated confidence (margin + entropy + lens consistency + residual stab via logistic regression) |
| `memory` | `[{text, score, peak_pos}]` | neuron-memory hits |
| `backend` | string | `"c"`, `"pytorch"`, or `"training"` |

## modes (what is pullable)

| mode | source | endpoint or path |
|---|---|---|
| active inference | live forward, TFRM frame stream | `GET /generate?prompt=...&backend=c\|pytorch` |
| in-process training | TFRM frame dumped each checkpoint via `dump_generation` | `models/<name>/hooks/step_<N>/generation.json` |
| past-checkpoint learning | all eight artifacts above per saved step | `models/<name>/hooks/step_<N>/{probe,lens,classroom,grades,concepts,surprise,quant_kl,generation}.{json,npz}` |
| distillation | the same eight-artifact suite runs on teacher and student per checkpoint; quality comparison is field-level | `models/<teacher>/hooks/...` and `models/<student>/hooks/...` at the same step |
| live training stream | per-step TFRM-lite (residual norms, top-K neurons, lens top-3) over SSE from `veritate_mri/app.py` | `GET /train_stream` (opt-in via trainer call to the streaming helper) |
| ablation | inference forward with one FFN neuron zeroed at the kernel boundary | `GET /generate?...&ablate_layer=L&ablate_neuron=N` (engine + pytorch) |
| ablation replay | same as ablation but loads a saved checkpoint instead of the latest bin | `GET /generate?...&checkpoint=step_<N>&ablate_layer=L&ablate_neuron=N` |

## interpretability endpoints

Atlas endpoints aggregate frames from the in-memory ring (live) or `step_<N>.json` (past). Backed by one module (`veritate_mri/atlas.py`) that consumes a frame iterator + query and returns aggregated stats. No new TFRM fields, no new dump artifacts: pure derivations over Rule-4 data.

| endpoint | input | output |
|---|---|---|
| `GET /atlas/concept/<substring>?model=...&step=<N>` | byte-substring filter | top-N `(layer, neuron)` pairs ranked by aggregated `dla_picked` contribution across frames whose sampled bytes match `substring` |
| `GET /atlas/neuron/<layer>/<index>?model=...&step=<N>` | neuron coordinate | inverse map: which output substrings, attention positions, and concepts most often coincide with this neuron's activation |
| `GET /atlas/lifetime/<layer>/<index>?model=...` | neuron coordinate | per-step trajectory across `probe_step_<N>.json` (rank, magnitude, prompt-position) |
| `GET /atlas/circuit?model=...` | model name | static `W_down[L] @ W_up[L+1]` neuron-to-neuron transfer matrix; computed once at model load, cached per model |
| `GET /atlas/concepts_inverted?model=...&step=<N>` | model + step | `concepts_step_<N>.json::top_neurons` inverted into a neuron-keyed map |

## update obligation

Adding, removing, or renaming any field above requires:

1. The producer in `veritate_mri/checkpoint_probe.py` is updated.
2. This file's table is updated in the same commit.
3. If a TFRM frame field is added, both the C engine emit and `dump_generation` are updated in the same commit.
4. The MRI dashboard render function gates on field presence so old runs do not break.
