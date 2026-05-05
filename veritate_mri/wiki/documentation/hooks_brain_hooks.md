---
title: brain hooks
date: 2026-05-05
tags: [hooks, contract, transformer, mamba2]
summary: Which hook fields are available on which architecture and engine path.
---

> Source: `documentation/hooks/brain_hooks.md` (mirrored copy; the file at that path remains the canonical contract).

# brain hooks

Which hook fields are available on which architecture and engine path. Field shapes are identical across paths so the MRI dashboard renders one shape regardless of source. Field index lives in `docs/hooks/contract.md`.

## paths

| path | what it is | typical use |
|---|---|---|
| pytorch | `veritate_mri/backends/pytorch.py::Brain.stream` with forward hooks on each block | full activation capture, slow (~20-30 ms/token) |
| engine | `engine/src/model.c::forward_decode` writing `trace_record_t` slices | INT8 inference, fast (~3 ms/token with full trace) |

## architecture coverage

| architecture | pytorch | engine |
|---|---|---|
| transformer | full TFRM v7 | full TFRM v7 |
| mamba-2 | partial (no FFN, no attention fields; SSD-specific fields not yet emitted) | not integrated; SSD scalar + AVX-512 kernels exist, no `model_t` yet |

### transformer field sources

| group | pytorch | engine |
|---|---|---|
| residual_pre, residual_post | `block` pre-hook + post-hook | `trace->residual_pre[L][pos]`, `trace->residual_post[L][pos]` (int16) |
| ffn_full, ffn_top, ffn_argmax | `ffn_up` post-hook + `F.gelu` | `trace->ffn_neurons[L][pos]` (int8, post-GELU) |
| attn | `qkv` hook + softmax recompute | `trace->attention_scores` dequantized from int16 |
| lens | `embed @ residual_post[L]` (fp32) | `lens_project` inline scalar dot in model.c (int32) |
| dla_picked, dla_argmax, decisiveness | brain.py fp32 from cap_ffn + byte_direction | `byte_direction_build` at model_load + `dla_top` + `decisiveness_compute` (int16/int32) |
| dla_cand (v8) | brain.py fp32 over each top-K candidate byte | engine: `dla_top` invoked once per `cand[i].b` against the same neuron projection used for `dla_picked` |
| ablation (v8) | pytorch hook in `Brain.stream` zeros `ffn_neurons[L][index]` pre-`ffn_down` | engine: `ablation_mask[V_FFN]` int8 multiplier passed into `ffn_down` (scalar + AVX-512). Null pointer = no-op |
| saturation | per-layer fraction over INT8 budget | not emitted (depends on QAT scale the engine does not track) |
| memory | brain.memory probe lookup | not emitted |
| confidence components (margin, entropy, lens_consistency, residual_stab, confidence) | computed in brain.py | computed by the server from engine fields |

### mamba-2 field availability

| field | pytorch | engine |
|---|---|---|
| residual_pre, residual_post | block_in / block_out hooks | not integrated |
| ffn_* | n/a (no FFN block in mamba-2) | n/a |
| attn | n/a (no attention; SSD scan is the op) | n/a |
| lens | `embed @ residual_post[L]` | not integrated |
| ssm_state (per-layer SSM hidden state) | not yet emitted; field reserved | not yet emitted; requires TFRM bump |

## kernels

Kernel selection affects performance only. Trace field shape is kernel-agnostic.

| kernel | path | architectures it serves |
|---|---|---|
| scalar | `engine/kernels/scalar/` | reference oracle for every architecture; parity bar for x86_64 / arm64 ports |
| x86_64 avx-512 vnni | `engine/kernels/x86_64/` | transformer matmul (qkv, out_proj, ffn_up, ffn_down, lm_head); mamba-2 ssd kernel prototype |
| arm64 neon | `engine/kernels/arm64/` | planned, transformer matmul first |

## adding a hook

1. Add the field to `documentation/hooks/contract.md` (this commit is the gate).
2. Emit from the producer on every supported path: `veritate_mri/checkpoint_probe.py::dump_generation`, engine forward, pytorch `Brain.stream`. All in the same commit.
3. Add the dashboard render path. The render must gate on field presence so older runs do not break.
4. If the field changes the TFRM frame size, bump the trace version in `engine/src/veritate.h` and update the parser.

## interpretability layer (v8)

Five capabilities sit above the per-token frame. They are not new hooks: they are derivations and one runtime knob (ablation). All consume the existing Rule 4 + Rule 5 fields.

| capability | tier | path | input fields | new TFRM field |
|---|---|---|---|---|
| concept→neuron atlas | 1 | server-side aggregation in `veritate_mri/atlas.py` | `dla_picked` across many frames keyed by output substring | none |
| neuron lifetime across training | 1 | server walks `probe_step_<N>.json` per model | `probe_step_<N>::layers[].neurons[]` ID + magnitude per step | none |
| neuron→concept inversion | 1 | server inverts `concepts_step_<N>.json::top_neurons` | `concepts_step_<N>::concepts[].top_neurons` | none |
| static circuit graph | 1 | computed once at model load: `W_down[L] @ W_up[L+1]` | model weights only | none |
| top-K candidate DLA | 2 | engine emit + pytorch `Brain.stream` + `dump_generation` | `cand`, `byte_direction`, neuron projection | `dla_cand` |
| causal ablation | 3 | optional `ablation_mask[V_FFN]` into `ffn_down`; pytorch hook mirrors | request param `ablate_layer`, `ablate_neuron` | `ablation` echo only |
| live training stream | 4 | trainer flag `--mri-stream` routes per-step TFRM-lite through `veritate_mri/save.py` | residual norms, top-K neurons, lens top-3 | none (TFRM-lite is a subset, not a new field) |

Tiers 1, 3, and 4 add **no** TFRM fields. Tier 2 adds two end-of-frame fields under the rule-7 chain bump v7→v8. Tier 3's per-frame `ablation` echo is metadata, not a derivation.
