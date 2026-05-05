---
title: What Veritate is
date: 2026-05-05
tags: [intro, overview, theoretical]
summary: One-page primer on what the project is, what it runs on, and what it predicts.
---

> **Status: research project, not a product.** Most of what is described in the wiki under `concepts/` and `experiments/` is theoretical or in-flight. The shipped pieces are the INT8 engine, the training pipeline, and a small PoC model. Everything past that, including long-context memory, multimind modules, and BitNet ternary scaling, is under active design and has not been validated end-to-end.

Veritate is a hand-written INT8 inference engine written in C and assembly, paired with a PyTorch training pipeline. PyTorch trains the model. Veritate runs it.

## the one-line version

Predict the next byte. Do it fast enough on a desktop CPU that the response feels instant. Keep every layer inspectable while it runs.

## the constraints, in plain words

1. **Byte-level only.** The model reads and writes raw bytes (256 possible values). No subword tokenizers, no word lists. "hello" is five tokens, an emoji is four. See [byte vs BPE tokens](byte_vs_bpe_tokens.md).
2. **Decode budget.** Every byte produced must take under 0.1 ms on the target CPU (AMD Ryzen 7 9800X3D, AVX-512 + VNNI, 96 MB L3). Anything that busts the budget gets cut or redesigned.
3. **Glass-box.** Every intermediate tensor at every layer is dumped to disk during training and inference. The dashboard renders them. Nothing is hidden behind an opaque kernel.

## the moving parts

- **`veritate_engine/`** the C inference engine. Hand-tuned kernels per CPU architecture. Loads a model, produces bytes.
- **`veritate_mri/`** the Python training pipeline plus a web dashboard for watching the model think.
- **`veritate/`** the canonical PyTorch model definition. One file, no second class.
- **`plugins/`** training experiments. Each plugin owns its corpus, its trainer, and its hooks. The platform just loads them.
- **`models/`** trained checkpoints, one folder per model.

## the goal

A model that *feels* sub-millisecond. The forward pass cannot literally finish in under a millisecond, but a streamed first token under 100 ms after the user commits a prompt feels instant to a human eye. Getting there is part faster forward, part running the forward speculatively while the user is still typing.
