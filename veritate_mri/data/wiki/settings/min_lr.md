---
title: learning rate floor (min_lr)
date: 2026-07-27
tags: [settings, training]
summary: The lowest learning rate the schedule decays to by the end of the run.
---

# learning rate floor (min_lr)

The bottom of the learning-rate curve. The schedule decays from `base_lr` toward this value and never goes below it.

## what it does

A run that decays all the way to zero stops learning before it stops running. A floor keeps small, useful updates flowing through the final steps while still letting the model settle.

Consumed by `lr_at()` in `veritate_mri/training/veritate_trainer.py` for the `cosine`, `linear`, and `wsd` schedules. Under `constant` it is unused, and the form hides the field when `constant` is selected.

## range and default

Any positive float at or below `base_lr`. The relationship is not enforced in code.

Every trainer manifest sets `min_lr` to exactly one tenth of its `base_lr`. At `veritate_800m`, `base_lr` is 3e-4 and `min_lr` is 3e-5.

## when to change it

Raise the floor when a run ends with the loss still falling and there is no budget to extend it: more of the run happens at a useful rate.

Lower it when the last part of training is noisy and the model needs to settle harder.
