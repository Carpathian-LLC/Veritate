---
title: xIELU as a drop-in for GELU
date: 2026-05-05
tags: [activation, training, perplexity, lut]
summary: xIELU beat GELU on a matched 10.77M-param byte-level run. Inference cost is identical via INT8 lookup; training cost is 2.83x slower.
---

GELU is the default activation in transformer FFNs. xIELU is a recent variant that smooths the negative side differently and has a slightly different gradient near zero. Worth a head-to-head.

## the test

Same architecture, same seed, same data, same step count: 10.77M-param byte-level transformer on TinyStories, 8000 steps. Only the FFN activation function changed.

| activation | final val loss | val ppl |
|---|---|---|
| GELU | 0.7499 | 2.117 |
| xIELU | 0.7373 | **2.090** |

xIELU pulls ahead at step 100 and stays ahead through step 7900. Not a noise-level difference; the gap is consistent across the second half of training.

## the cost question

Two costs to think about: training and inference.

**Training:** xIELU is 2.83x slower per step in eager bf16 PyTorch, because the reference implementation uses `torch.where` plus `expm1`, both of which break the fused-kernel path GELU enjoys. A custom CUDA kernel would close most of this gap; we have not written one.

**Inference:** identical to GELU. Veritate runs the activation as an INT8 lookup table at decode time, so the activation function is one memory load regardless of which curve it implements. The 2.83x training overhead does not propagate to the engine.

This asymmetry is the whole reason the swap is interesting: a small quality win at zero inference cost, paid for once in extra training time.

## recommendation

Ship xIELU on the next 80M run. The 2.83x training overhead is one-time and bounded; the perplexity win compounds across every model trained from here on.

## caveat

Tested only at 10.77M on TinyStories. Behavior at 80M / 1B and on harder data is not validated. The mechanism (smoother negative-side gradient) generalizes in the literature, but the numerical win does not always transfer at scale. Re-validate on the next model class before committing.

## takeaway

Activation choice is a free hyperparameter at inference time when the kernel is a LUT. That changes which experiments are worth running: anything you can fold into a 256-entry table costs nothing to ship, so the bar for trying it is low.
