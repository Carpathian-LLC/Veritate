---
title: weight decay
date: 2026-07-27
tags: [settings, training]
summary: A constant pull of every weight toward zero, applied each step to keep the model from memorizing.
---

# weight decay

A small shrink applied to every weight on every step, independent of the gradient.

## what it does

Left alone, weights grow to whatever size fits the training data best, including sizes that fit its noise. Decay applies a steady inward pull, so a weight only stays large if the gradient keeps pushing it there. The result is a model that leans on patterns that repeat rather than details it saw once.

Both optimizers apply it in decoupled form, meaning the shrink is a separate operation from the gradient step rather than something folded into the gradient. `torch.optim.AdamW` handles the AdamW path; the Muon group applies its own decoupled shrink inside `veritate_core/plugin/optim.py`.

## range and default

Any non-negative float. Zero disables it.

Every trainer manifest sets `0.1` except `veritate_80m`, which sets `0.18`.

## when to change it

Raise it when validation loss rises while training loss keeps falling: the model is fitting the corpus rather than the language in it.

Lower it when both losses stall high on a small model, where the pull toward zero can cost more capacity than it saves.
