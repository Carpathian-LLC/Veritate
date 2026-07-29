---
title: BPTT window
date: 2026-07-27
tags: [settings, training, advanced]
summary: How many chunks of history one backward pass reaches through, and the main driver of peak activation memory.
---

# BPTT window

BPTT is backpropagation through time: carrying the gradient backwards across chunk boundaries so the model learns from context earlier than the current chunk. This setting is how many chunks that reach covers.

## what it does

A training step walks `n_chunks` chunks in order. This value, call it K, says how many of those chunks are held live before a backward pass runs and their memory is released.

```
backward passes per step = n_chunks / bptt_window   (rounded up)
peak activation memory   scales with bptt_window
```

`bptt_window = 1` frees each chunk immediately: cheapest memory, no gradient signal crossing chunk boundaries. `bptt_window = n_chunks` holds the whole step live: full BPTT, highest memory. Setting it above `n_chunks` behaves the same as setting it equal.

Implemented in `chunked_step` in `veritate_mri/training/veritate_trainer.py`, which clamps the value with `K = max(1, bptt_window)`.

This is the setting the dashboard memory estimator multiplies activation memory by. `n_chunks` does not appear in that estimate at all.

## range and default

Any integer of 1 or more.

Manifest defaults: 4 for `veritate_10m`, 2 for `veritate_80m` through `veritate_800m`, 1 for `veritate_1b3` and every larger trainer.

## when to change it

Raise it when the model needs to connect information across chunk boundaries, which is the common case on the `recurrent` and `memory` trunks where state is threaded from one chunk to the next. Watch memory: the increase is close to linear.

Lower it to 1 as the first move when a run aborts on memory but batch size must stay where it is.
