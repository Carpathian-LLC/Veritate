---
title: Batch Size
summary: How many rows of text the model studies together in one step; bigger is faster and steadier but uses more memory.
tags: training, settings
---

# Batch Size

Batch size is how many rows (chunks of training text) the model processes together in a single step before it updates. Instead of learning from one example at a time, it looks at a stack of them at once and averages the lesson.

## Why it matters

- **Speed**: processing a stack together is more efficient than one-at-a-time, so bigger batches finish the run in less wall-clock time.
- **Stability**: averaging over more examples gives a smoother, less noisy update each step, which can make training steadier.

## Weak-hardware angle

Every row in the batch has to be held in memory at once, so **a bigger batch uses more memory (VRAM)**. On a modest machine this is often the first thing you have to turn down to make a run fit. If a run runs out of memory, lowering batch size is the usual first fix.

## When to change it

- Raise it when you have memory headroom and want faster, smoother training.
- Lower it when a run will not fit, or crashes with an out-of-memory error.

## Gotcha

- Batch size and learning rate are linked: much larger batches often want a slightly higher learning rate. If you change batch size a lot by hand, revisit the learning rate rather than assuming the old one still fits.
