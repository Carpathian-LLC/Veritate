---
title: learning rate peak (base_lr)
date: 2026-07-27
tags: [settings, training]
summary: The highest learning rate the schedule ramps up to, and the single most consequential number in a run.
---

# learning rate peak (base_lr)

The learning rate is how big a step the weights take each update. `base_lr` is the top of the curve: warmup climbs to it, and the schedule decays away from it.

## what it does

Too high and the loss spikes or diverges. Too low and the run crawls and may settle in a worse place. Larger models tolerate smaller peak rates, which is why the manifest defaults fall as size rises.

`lr_at()` in `trainers/common/vanilla_trainer.py` computes the rate for the current step from `base_lr`, `min_lr`, `warmup_steps`, and the schedule, and the result is written into every optimizer parameter group before the step.

## range and default

Any positive float. Nothing in the code checks that `base_lr` is above `min_lr`, so set it that way deliberately.

Manifest defaults by trainer:

| trainer | base_lr |
|---|---|
| `veritate_10m` | 6e-4 |
| `veritate_80m` | 1e-5 |
| `veritate_200m` | 2e-5 |
| `veritate_400m`, `veritate_800m` | 3e-4 |
| `veritate_1b3` | 2e-4 |
| `veritate_3b` | 1.5e-4 |
| `veritate_13b` | 1.2e-4 |
| `veritate_50b` | 1e-4 |
| `veritate_70b` to `veritate_120b` | 6e-5 |
| `veritate_160b` to `veritate_350b` | 5e-5 |
| `veritate_500b` and above | 4e-5 |

Auto tune rescales the manifest value by the square root of the batch-size ratio when it changes batch size, because a larger batch supports a larger step.

## when to change it

Lower it when continuing or fine-tuning an already-trained model: a fresh-pretrain rate will undo what the model learned. A tenth of the pretrain rate is a reasonable starting point.

Lower it when the loss curve spikes or goes flat and high. Raise it when loss falls smoothly but far too slowly and the box has steps to spare.
