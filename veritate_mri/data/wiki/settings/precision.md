---
title: precision
date: 2026-07-27
tags: [settings, training]
summary: The number format the training math runs in: bf16 halves memory, fp32 keeps every digit.
---

# precision

How many bits each number carries during training.

## what it does

`bf16` (brain float 16) keeps the same range as a 32-bit float but far fewer digits of detail. Activations take half the memory and the math runs faster on hardware with bf16 units, which is most modern GPUs and Apple silicon. `fp32` keeps full detail at double the memory.

Gradients and the optimizer state stay in fp32 in both modes. Only the forward and backward math runs in the selected format, through `torch.autocast`.

`veritate_core/plugin/hardware.py::resolve_precision` maps the request to a torch dtype. When the device cannot do bf16, the request quietly becomes fp32 rather than failing, so a run started with `bf16` on a plain CPU box trains correctly, just slower and larger.

## options and default

| value | meaning |
|---|---|
| `bf16` | 16-bit forward and backward, fp32 master weights |
| `fp32` | full 32-bit throughout |

Valid values are fixed in `PRECISIONS` in `trainers/common/vanilla_trainer.py`. Anything else stops the run with a clear error.

Every trainer manifest defaults to `bf16` except `veritate_10m`, which defaults to `fp32`. Auto tune sets `fp32` when the box has neither CUDA nor Apple silicon.

## when to change it

Leave it on `bf16`. Switch to `fp32` for a very small model where reduced detail dominates the signal, or when chasing a numerical bug and full precision removes a variable.
