---
title: runs, checkpoints, and steps
date: 2026-07-27
tags: [concepts, basics, training]
summary: What one training step does, what a checkpoint saves, and what lives in a model directory.
---

# runs, checkpoints, and steps

Three words carry most of the training vocabulary on this platform.

## a step

One step is one weight update. The trainer reads a batch of text, predicts the next byte at every position, measures how wrong it was, computes gradients, and applies one optimizer update.

Bytes consumed by a step:

```
bytes per step = batch_size * seq * n_chunks
```

`total_steps` sets the length of the run. Step count, not wall-clock time, is what every schedule is written against: the learning rate, the warmup ramp, and the decay tail are all functions of the current step out of the total.

## a run

A run is one launch of a trainer, and it owns one directory under `models/<name>/`. That directory is the whole record of the run:

```
models/<name>/
  config.json          the shape, the training arguments, the parameter count
  train.csv            one row per logged step: loss, learning rate, gradient norm
  checkpoints/         step_<N>.pt, the weights at each save
  hooks/step_<N>/      the interpretability dumps taken at each save
  veritate.bin         the exported engine artifact, when one has been generated
```

`train.csv` is what the training charts read. `hooks/` is what the interpretability views read. Nothing in the dashboard reads a trainer's console output, so a run is fully inspectable after the fact.

A **scratch** run starts from random weights. A **continue** run resumes from an existing model's latest checkpoint and keeps its shape, trunk, and vocabulary; only the training-loop settings are open to change.

## a checkpoint

A checkpoint is the model's weights written to disk at a given step, plus the arguments needed to rebuild it. Saves happen every `ckpt_every` steps.

Every save goes through one function, `save()` in `veritate_mri/training/save.py`, and that function always does the same three things: write the checkpoint, write or refresh `config.json`, and run the full dump suite into `hooks/step_<N>/`. No trainer writes a checkpoint by itself, so every checkpoint on the box carries the same artifacts and the dashboard can render any of them without special cases.

Checkpoints are written to a temporary name and renamed into place, so an interrupted save leaves a stray temporary file and never a corrupt checkpoint.

The dump suite is what makes a checkpoint more than weights. It captures which neurons fired for which concepts, what each layer predicted partway through the stack, reading and reasoning scores, and sample generations. Because every checkpoint has one, a run can be replayed step by step to watch a capability appear.

## from checkpoint to serving

A checkpoint runs directly under PyTorch through `POST /backends/pytorch`. Exporting it with `POST /export/<name>` produces a `veritate.bin` that the compiled C engine serves, which is faster and leaves the accelerator free for training. Export expects a canonical dense trunk; other trunks train and serve under PyTorch.
