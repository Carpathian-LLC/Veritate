---
title: optimizer
date: 2026-07-27
tags: [settings, training]
summary: The algorithm that turns each step's gradients into an actual weight update.
---

# optimizer

The rule that decides how far and in which direction every weight moves once the gradients for a step are known.

## what it does

A gradient says which way each weight should move. The optimizer decides how big that move is, using the history of previous moves. Two choices ship.

- `adamw`: `torch.optim.AdamW`. Per-weight adaptive step size from a running mean and variance of the gradient, plus decoupled weight decay. The standard, well understood, works everywhere.
- `muon`: orthogonalized momentum. Every 2-D weight matrix gets its momentum matrix orthogonalized by a short Newton-Schulz iteration before the step, which spreads the update evenly across directions instead of letting a few dominate. Embeddings, norms, and all 1-D parameters stay on AdamW inside the same optimizer object.

Both are built in `veritate_mri/training/veritate_trainer.py::build_optimizer`. The Muon path is assembled by `veritate_core/plugin/optim.py::build_muon`, which splits parameters into the two groups and presents them as one optimizer, so a single learning-rate schedule drives both.

## options and default

| value | meaning |
|---|---|
| `adamw` | AdamW on every parameter |
| `muon` | Muon on 2-D weights, AdamW on the rest |

Trainer manifests from `veritate_10m` through `veritate_3b` set `muon`. Manifests from `veritate_13b` up set nothing, so the field renders empty and the trainer falls back to `adamw` from `RESERVED_STR_FLAGS` in `veritate_trainer.py`.

If `muon` is requested on a box where the optimizer helper cannot load, the trainer logs that it is unavailable and continues on AdamW rather than failing the run.

## when to change it

Use `muon` for a fresh pretrain: it typically reaches a given loss in fewer steps at the same learning rate. Use `adamw` when reproducing a published AdamW recipe, or when comparing against an existing AdamW run where the optimizer must be held fixed.
