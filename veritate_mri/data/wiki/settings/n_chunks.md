---
title: TBPTT Chunks Per Step
summary: How many chunks of text each step covers; more chunks means more bytes learned per step without raising memory.
tags: training, settings
---

# TBPTT Chunks Per Step

Each training step processes a long sequence in pieces called chunks. **n_chunks** sets how many chunks a single step marches through, which means **how many bytes of text the model learns from per step**. More chunks = more text covered each step. TBPTT is short for "truncated backpropagation through time," the method of learning across a long sequence a piece at a time.

## Why it matters

Covering more bytes per step lets the model move through the corpus faster and see longer stretches of context in one go. It is a way to do more per step.

## Weak-hardware angle (the pleasant surprise)

- Raising n_chunks does **not** raise memory. It processes chunks in sequence, so you get more bytes per step for free on the memory side.
- The knob that controls memory is a different one: **bptt_window**, which sets how many chunks stay live for the backward pass. If you want to grow context while watching memory, raise n_chunks and keep bptt_window small.

## When to change it

- Raise n_chunks to push more bytes through each step, especially when you have plenty of data and want to move through it faster.
- Keep an eye on **bptt_window** (not this knob) if memory is your constraint.
