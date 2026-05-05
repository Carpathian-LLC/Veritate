---
title: Why the C engine spoke gibberish
date: 2026-05-05
tags: [engine, quantization, debugging, quality]
summary: Two compounding export bugs (transposed weights, mismatched embed scales) made the C path output garbage while PyTorch was coherent. Fixed without touching engine math.
---

For weeks the same checkpoint produced clean prose under PyTorch and word-salad under the C engine. The engine math turned out to be fine. The `.bin` exporter was wrong in two independent places, and the two bugs partially cancelled, which is why the output looked plausibly broken instead of obviously broken.

## bug 1: transposed weights

PyTorch stores `nn.Linear.weight` as `[out, in]` (shape `[N, K]`). The C engine's `prep_b()` reads weights as `[K, N]` row-major. Every weight matmul in the engine was therefore reading a transposed matrix.

Fix: `np.ascontiguousarray(W.T)` before serialize, in `training/train.py::export_to_bin`.

## bug 2: mixed-unit embedding sum

Token embeddings and positional embeddings were quantized at independent scales (55.7 and 489.2). The engine then summed them as int8. Adding two int8 vectors with different scales is meaningless arithmetic: the sum lives in no consistent unit, so every downstream layer sees noise of unknown magnitude.

Fix: `quantize_embed_at_act_scale()` quantizes both at the activation scale (32) so the sum has a defined unit.

## how we found it

`mri/server/diff.py`, the differential trace harness, runs the same prompt through both backends step-by-step and reports cosine distance between matching tensors layer by layer. Pre-fix, layer-0 `residual_post` had cos_dist 0.987 vs PyTorch (essentially orthogonal). Post-fix, 0.011. The harness pinpointed the divergence location; the bugs were obvious from there.

## findings

- Engine math was untouched. Bit-match scalar oracle preserved.
- A re-export of every previously-trained checkpoint was required.
- Quality is the gate for any speed claim; this was the precondition for the rest of the perf work.

## takeaway

Differential traces beat staring at outputs. If two paths are supposed to compute the same function, instrument every layer pair and let the numbers tell you which one drifted first.
