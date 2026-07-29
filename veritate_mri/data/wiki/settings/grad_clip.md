---
title: gradient clip
date: 2026-07-27
tags: [settings, training]
summary: A ceiling on how large one step's gradient may be before it is scaled down.
---

# gradient clip

A cap on the total size of the gradient for a step. When the gradient is larger than the cap, the whole gradient is scaled down to exactly the cap, keeping its direction and shrinking its length.

## what it does

Most steps produce ordinary gradients. Occasionally a batch contains something unusual and produces one enormous gradient that would move the weights far out of a good region in a single update. Clipping keeps that step in proportion instead of letting it dominate.

Applied unconditionally in `veritate_mri/training/veritate_trainer.py::run` through `torch.nn.utils.clip_grad_norm_`. The measured gradient norm, before clipping, is written to `train.csv` every logged step, so the training charts show how often the cap is actually reached.

## range and default

A positive float. Every trainer manifest sets `1.0`.

Zero is not an off switch here: a cap of zero scales every gradient to nothing and the model stops learning. To effectively disable clipping, set a large value.

## when to change it

Lower it to around 0.5 when the loss curve shows sharp spikes and the logged gradient norm confirms occasional huge steps.

Raise it when the logged norm sits at the cap for most steps. At that point clipping rescales every update instead of catching outliers.
