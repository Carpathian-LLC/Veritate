---
title: batch size
date: 2026-07-27
tags: [settings, training]
summary: How many independent stretches of text the model reads at once before each weight update.
---

# batch size

The number of separate rows of training text processed side by side in one step.

## what it does

Each row is an independent sample drawn from the corpus. The loss is averaged over all of them, so a larger batch gives a steadier estimate of which direction the weights should move: less noise per step, fewer steps needed, more memory and more math per step.

Batch size is the main lever on how fully a GPU is used. A GPU running one row at a time spends most of its time waiting.

The value sets the row count in the data loader and feeds the memory plan in `veritate_core/plugin/mem_planner.py`, which refuses a configuration that cannot fit before the run starts.

## range and default

Any integer of 1 or more. No upper limit is enforced in code; the memory plan is the real ceiling.

Manifest defaults track model size: 32 at `veritate_10m`, 12 at `veritate_80m`, 24 at `veritate_200m`, 16 at `veritate_400m`, 8 at `veritate_800m`, 4 at `veritate_1b3`, and 1 for every trainer of 13B and above.

Running auto tune replaces the manifest value with the batch size that measured the highest tokens per second on this box, and that measured value is what the form shows afterwards.

## when to change it

Raise it while throughput keeps improving and the memory plan still fits. Stop when either stalls. A large jump in batch size changes the effective gradient scale, so raise the learning rate along with it or move in moderate steps.

Lower it when a run aborts on memory, before reaching for anything else.
