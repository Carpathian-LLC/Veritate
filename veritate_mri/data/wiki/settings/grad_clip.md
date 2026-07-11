---
title: Gradient Clip
summary: A safety cap on how big any single update can be, so one wild step cannot knock training off the rails.
tags: training, settings
---

# Gradient Clip

The gradient is the size and direction of the correction the model wants to make each step. Gradient clip is a **cap on how big that correction is allowed to get**. If a step tries to push harder than the cap, it gets scaled back down to the limit. It is a seatbelt for training.

## Why it matters

Every so often a single batch produces a huge, unlucky gradient: a spike that would shove the model far in one direction and undo a lot of progress, sometimes crashing the run. Clipping catches those spikes and keeps each step within a sane range, which makes training much more stable.

## How the dial works

- **Lower** = a stricter cap = spikes are trimmed harder, so training is calmer but individual steps are gentler.
- **Higher** = a looser cap = the model is freer to take big steps, at more risk of a destabilizing spike.

## When to change it

- Leave it at the recipe's default for most runs; a common value is around 1.0.
- Tighten it (lower the number) if you see the loss occasionally spiking or the run going unstable.

## Gotcha

- Clipping treats the symptom, not the cause. If you constantly hit the cap, the real fix is usually a lower learning rate, not an ever-tighter clip.
