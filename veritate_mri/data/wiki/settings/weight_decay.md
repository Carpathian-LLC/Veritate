---
title: Weight Decay
summary: A gentle pull that keeps the model's internal numbers from growing too large, which helps it generalize instead of memorizing.
tags: training, settings
---

# Weight Decay

Weight decay is a small, steady pull that nudges the model's weights (its internal numbers) toward smaller values every step. It is a form of regularization, meaning a technique that discourages the model from over-fitting the training data.

## Why it matters

Left unchecked, a model can grow large, spiky weights that memorize the training set instead of learning the general pattern. Such a model looks great on data it has seen and poorly on data it has not. Weight decay counteracts that by keeping the numbers modest, which usually improves how well the model handles new inputs.

## How the dial works

- **Higher** = stronger pull = more regularization. The model is kept simpler, at the risk of under-fitting if overdone.
- **Lower** (or zero) = little to no pull. The model can fit the data more tightly, at the risk of over-fitting.

## When to change it

- Leave it at the recipe's default for most runs; the presets already pick a sensible amount.
- Nudge it **up** if the model does much better on training data than on validation data (a sign of over-fitting).
- Nudge it **down** if the model seems to be under-fitting and never learns the data well.
