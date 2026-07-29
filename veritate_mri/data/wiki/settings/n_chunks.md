---
title: chunks per step
date: 2026-07-27
tags: [settings, training]
summary: How many back-to-back sequence-length chunks make up one training step.
---

# chunks per step (n_chunks)

The number of consecutive `seq`-byte chunks the model walks through before the optimizer updates the weights.

## what it does

The data loader draws rows that are `seq * n_chunks` bytes wide. The training step walks that row one `seq`-wide chunk at a time, accumulating loss, then takes a single optimizer step for the whole row.

Bytes consumed per step:

```
bytes per step = batch_size * seq * n_chunks
```

This is the knob that raises bytes per step without raising peak memory. Memory is governed by `bptt_window`, which controls how many chunks are held live at once. Raising `n_chunks` costs time per step, not memory.

The width is computed in `veritate_mri/training/veritate_trainer.py::run` as `total_chunk_len = seq * n_chunks`, and the walk happens in `chunked_step` in the same module.

## range and default

Any integer of 1 or more.

Manifest defaults: 4 for `veritate_10m` and `veritate_200m`, 2 for `veritate_80m`, `veritate_400m`, and `veritate_800m`, 1 for `veritate_1b3` and every larger trainer.

## when to change it

Raise it on a memory-tight box to get more data through per optimizer step when batch size cannot go any higher. The optimizer runs once per step regardless, so a higher value also amortizes optimizer overhead across more bytes.

Leave it at 1 when batch size already saturates the device.
