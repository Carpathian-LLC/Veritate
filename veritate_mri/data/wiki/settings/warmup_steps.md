---
title: warmup steps
date: 2026-07-27
tags: [settings, training]
summary: How many steps the learning rate spends climbing from zero to its peak before the schedule takes over.
---

# warmup steps

The number of steps at the start of a run during which the learning rate rises linearly from zero to `base_lr`.

## what it does

A freshly initialized model has random weights and produces large, badly aimed gradients. A full-size update on step one can push the weights somewhere the run never recovers from. Warmup starts the updates tiny and grows them, giving the optimizer time to build its running statistics before it takes a real step.

Warmup runs under every schedule, including `constant`. `lr_at()` in `veritate_mri/training/veritate_trainer.py` applies it before any schedule shape is considered, and the field is hidden on the form when the schedule is `constant`, where the flat rate makes the ramp the only shaping there is.

## range and default

Any non-negative integer. Zero skips warmup entirely.

Manifest defaults rise with model size: 120 at `veritate_10m`, 500 at `veritate_80m` and `veritate_400m`, 1000 at `veritate_800m`, 1500 at `veritate_1b3`, 2000 to 3000 in the multi-billion range, and 8000 to 10000 for the largest trainers. `veritate_200m` sets 0.

Auto tune sets it to 3 percent of the total step count, with a floor of 50.

## when to change it

Raise it for a large model or an aggressive `base_lr`: the bigger the peak rate, the longer the ramp needs to be.

On a continue run it usually has no effect. The step counter picks up where the previous run stopped, which is already past the warmup window, so the schedule resumes mid-curve.
