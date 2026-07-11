---
title: Learning Rate Peak (base_lr)
summary: The highest learning rate the run reaches after warmup; the top speed at which the model updates.
tags: training, settings
---

# Learning Rate Peak (base_lr)

The learning rate is how big a step the model takes each time it updates. **base_lr** is the peak: the highest learning rate the schedule ramps up to before it starts easing back down. If the schedule is a car trip, base_lr is your cruising speed.

## Why it matters

The peak sets how boldly the model learns during the main stretch of the run. Too low and training crawls and may never reach good quality in the time you have. Too high and training gets unstable: the loss can spike, wobble, or blow up entirely.

## Good defaults

- Typical range is **1e-4 to 5e-4** (that is 0.0001 to 0.0005).
- For fine-tuning or continue runs, use a **lower** peak so you refine the model gently instead of overwriting what it already knows.
- It **must be greater than or equal to min_lr** (the floor the schedule decays to); the form will not accept a peak below the floor.

## Gotcha

- The right peak depends on the optimizer and batch size. If you change either by hand, the old peak may no longer fit. When in doubt, let a recipe set it.
