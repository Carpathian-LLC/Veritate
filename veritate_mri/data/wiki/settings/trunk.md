---
title: trunk
date: 2026-07-27
tags: [settings, training, architecture]
summary: Which model body the run builds: the arrangement of layers that does the thinking.
---

# trunk

The body of the model: the stack of layers between the byte inputs and the byte predictions. The trunk decides how the model mixes information across a sequence, and therefore what it costs to train and to run.

## what it does

Each choice builds a different class from `veritate_core/`. The selection happens once, in `veritate_mri/training/veritate_trainer.py::run`, and is baked into the checkpoint. A continue run keeps the trunk of the model it continues.

| value | class | how it mixes information |
|---|---|---|
| `dense` | `model.py::Veritate` | Plain transformer. Every layer attends over every byte. Simplest, best understood, cost grows with the square of sequence length. |
| `patched` | `model_patched.py::VeritatePatched` | Cheap local blocks run on every byte; expensive global blocks run only on patch anchor bytes. Global cost drops by the patch stride. |
| `hybrid` | `VeritatePatched` with a recurrent global mixer | Patched layout, but the global blocks carry a running state instead of attending. |
| `looped` | `VeritatePatched` with repeated global blocks | The same global block runs several times. Extra depth with no extra parameters. |
| `recurrent` | `model_recurrent.py::VeritateRecurrent` | Fixed-size state per attention head, updated byte by byte. Constant memory and constant time per byte at decode, no matter how long the text is. |
| `memory` | `model_memory.py::VeritateMemory` | A fast-weight branch mid-stack that writes only when a byte is surprising, giving the model a scratchpad that survives across chunks. |

## default

No trainer manifest sets `trunk`, so the field renders empty and the trainer falls back to `dense` from `RESERVED_STR_FLAGS` in `veritate_trainer.py`. A run with the field left alone is a dense transformer.

## when to change it

`dense` is the right answer for a first run and for any comparison against published numbers. Pick `patched` when sequence length, not parameter count, is the cost driver. Pick `recurrent` when decode speed on long conversations matters more than peak quality. `memory` and `looped` are research choices: measure them against a dense baseline of the same parameter count before trusting a result.

Export to the C engine expects a canonical dense trunk. A recurrent, patched, or memory model trains and runs under PyTorch but is not exportable to a `.bin` today.
