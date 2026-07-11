---
title: Sequence Length
summary: How many bytes of context the model reads at once per row; longer context teaches more per step but costs more memory.
tags: training, settings
---

# Sequence Length

Sequence length is how much text the model sees in one row, measured in bytes (roughly characters). It is the size of the window the model reads before predicting what comes next. A short sequence is like reading one sentence; a long one is like reading a whole paragraph before answering.

## Why it matters

A longer sequence lets the model learn longer-range patterns and gives it more to learn from in each step. If you want the model to track context across long passages, it needs a long enough sequence to have seen that context during training.

## Weak-hardware angle

Longer sequences cost more memory, and the cost grows fast. Here the memory grows **linearly** with sequence length when flash-attention is available, but **quadratically without it** (doubling the length nearly quadruples that part of the cost). On a modest machine, sequence length is one of the biggest memory levers, so keep it only as long as you actually need.

## When to change it

- Raise it when your task genuinely needs long context and you have memory to spare.
- Lower it to make a run fit, or to train faster when short context is enough.

## Gotcha

- A model rarely handles context much longer than it trained on. If you need long conversations at use time, train with a long enough sequence.
