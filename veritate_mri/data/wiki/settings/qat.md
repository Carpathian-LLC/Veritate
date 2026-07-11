---
title: QAT (INT8 Training)
summary: Trains the model while pretending its numbers are squeezed down to INT8, so it stays sharp after you shrink it for cheap, fast inference.
tags: training, settings
---

# QAT (INT8 Training)

QAT stands for quantization-aware training. Quantization means shrinking the model's numbers down to small 8-bit integers (INT8) so it runs cheaply and quickly when you actually use it. QAT trains the model while it *pretends* to be quantized, using "fake-quant" math, so it learns to stay accurate once the real shrink happens.

## Why it matters

Normally, squeezing a model to INT8 after training loses some quality. With QAT on, weights, activations, and norms all run under fake-quant INT8 during training, so the finished checkpoint **exports straight to an INT8 engine binary** with little quality loss. On a continue run, it fine-tunes an existing model into INT8 (use a low learning rate, around 1e-5).

## Weak-hardware angle

- QAT adds work during training: the fake-quant steps cost extra compute and a little extra memory.
- Because of that overhead, it is **usually left off for CPU training**, where the cost is not worth it.
- The payoff is at deployment, not training. Only turn it on when you actually plan to **run the finished model quantized** for fast, cheap inference.

## When to change it

- **On** when your end goal is an INT8 model for efficient inference.
- **Off** for ordinary training, or any CPU run, where you are not deploying quantized.
