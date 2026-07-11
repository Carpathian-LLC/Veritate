---
title: Optimizer
summary: The rule the trainer uses to turn each mistake into a weight update; muon reaches the same quality on fewer training bytes than the classic adamw.
tags: training, settings
---

# Optimizer

The optimizer is the update rule. Every step, the model makes predictions, sees how wrong it was, and the optimizer decides how to nudge the model's weights to be less wrong next time. It is the "how to learn from a mistake" part of training.

## Why it matters

Two optimizers can reach the same quality but take very different amounts of data and time to get there. A better update rule means you burn fewer training bytes (and fewer hours) for the same result, which matters a lot on modest hardware.

## The choices

- **adamw** : the classic, dependable default. If you are unsure, this always works.
- **muon** : a newer rule that, on this platform, reached the same quality using about **1.60x fewer training bytes** than adamw (measured 2026-07-03). In other words, it learns more per byte of data.

## When to change it

- Prefer **muon** when you want the most learning out of a limited corpus or a limited time budget.
- Stick with **adamw** if you want the most conventional, widely-documented behavior, or you are reproducing a classic setup.

## Gotcha

- Optimizer choice interacts with learning rate. If you switch optimizers by hand, use a recipe or a known-good learning rate rather than carrying over settings tuned for the other one.
