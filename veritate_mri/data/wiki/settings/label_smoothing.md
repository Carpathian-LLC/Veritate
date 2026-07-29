---
title: label smoothing
date: 2026-07-27
tags: [settings, training]
summary: A recorded run parameter that no loss path currently reads.
---

# label smoothing

Label smoothing is the technique of training a model toward slightly-less-than-certain targets: instead of "this byte is correct and every other byte is wrong", the target keeps a small share of probability spread across the alternatives, so the model learns to be confident without being absolute.

## what it does in this platform

Nothing to the loss. The value is parsed, recorded into the run's `config.json`, and stored with the checkpoint arguments, and no loss site reads it. Every cross-entropy call in `veritate_core/model.py`, `model_patched.py`, `model_recurrent.py`, `model_memory.py`, `model_rope.py`, and `chunked_step` in `veritate_mri/training/veritate_trainer.py` omits the smoothing argument.

Two runs that differ only in this value produce the same loss curve.

## range and default

A float from 0 to 1. Every trainer manifest sets `0.0` except `veritate_80m` and `veritate_200m`, which set `0.05`.

## when to change it

There is no behavior to change. To reduce overconfidence in a trained model today, the levers that do act are `weight_decay` and the decode-time settings on the generation surface.
