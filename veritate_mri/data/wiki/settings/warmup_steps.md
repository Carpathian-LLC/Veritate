---
title: Warmup Steps
summary: How many steps the run spends ramping the learning rate up from zero to its peak, so training starts gently instead of lurching.
tags: training, settings
---

# Warmup Steps

Warmup steps set how long the run spends easing the learning rate up from zero to its peak (base_lr) at the very start. Instead of hitting full learning speed on step one, the model builds up to it. It is the gentle pull-away before the car reaches cruising speed.

## Why it matters

A freshly started model is fragile. Taking a big learning step immediately can throw it off and make the loss spike or blow up. Warming up gradually lets the model find stable footing first, then pick up speed. More warmup = a slower, safer ramp.

## Good defaults

- The recipes set a sensible number for you; leave it alone for most runs.
- Larger models and higher peak learning rates generally benefit from **more** warmup.
- On a **continue** run (resuming an already-trained model), warmup is usually irrelevant: the model is already stable, so it is fine to keep this low or at its default.

## Gotcha

- Warmup is measured in steps, not a percentage. If you greatly change the total number of steps in a run, a fixed warmup count becomes a different fraction of the run, so revisit it.
