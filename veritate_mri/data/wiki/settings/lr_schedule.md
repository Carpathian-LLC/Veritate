---
title: LR schedule
date: 2026-07-27
tags: [settings, training]
summary: The shape of the learning-rate curve from the first step to the last.
---

# LR schedule

The path the learning rate takes across the run: up through warmup, then down to the floor in one of four shapes.

## what it does

Warmup applies to every schedule. For the first `warmup_steps` steps the rate climbs linearly from zero to `base_lr`. After that, with `p` running from 0 to 1 across the remaining steps:

| value | shape | formula |
|---|---|---|
| `cosine` | smooth S-curve down | `min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * p))` |
| `linear` | straight line down | `base_lr + (min_lr - base_lr) * p` |
| `constant` | flat at the peak | `base_lr`, and `min_lr` is unused |
| `wsd` | flat, then a decay tail | `base_lr` until the tail starts, then decays to `min_lr` |

`wsd` is warmup-stable-decay: the rate holds at `base_lr` for most of the run and only decays over the final fraction. Two extra fields control that tail. `wsd_decay_frac` is the fraction of total steps spent decaying, typically 0.1. `wsd_decay_kind` is the tail shape: `sqrt`, `linear`, or `cosine`.

All four are computed by `lr_at()` in `veritate_mri/training/veritate_trainer.py`.

## options and default

Valid values are fixed in `LR_SCHEDULES` in `veritate_mri/training/veritate_trainer.py`; anything else stops the run with a clear error.

Most trainer manifests default to `cosine`. `veritate_80m`, `veritate_200m`, and `veritate_3b` default to `wsd`.

## when to change it

Pick `wsd` when the total step count might change. Because the rate stays flat until the tail, a run can be stopped at any point during the stable phase and continued without distorting the curve, and the decay can be applied whenever the budget actually ends.

Pick `cosine` for a run with a fixed, known length: it is the default for good reason and needs no extra fields.

Pick `constant` for a short probe where the schedule should not be a variable.
