---
title: Number Precision
summary: How many bits each number in the model uses; bf16 halves memory on hardware that supports it, while fp32 is the safe, exact choice on plain CPUs.
tags: training, settings
---

# Number Precision

Precision is how much detail each number inside the model carries. **fp32** stores numbers in 32 bits (full detail). **bf16** stores them in 16 bits (half the detail, half the memory). Bf16 is short for "brain float 16," a compact number format.

## Why it matters

Halving the bits roughly halves the memory the model needs, which can be the difference between a run fitting on your machine or not. But smaller numbers only help if your hardware can do math on them natively.

## Weak-hardware angle (important)

- **bf16 is only faster on hardware with real bf16 support**: modern GPUs and Apple Silicon. There it saves memory *and* runs quickly.
- On a **plain CPU without bf16**, the chip has to fake the format in software. That is slower and can actually use more memory, so **fp32 is the better choice on a CPU**.
- The platform now protects you: on a machine without real bf16, it **automatically downgrades bf16 to fp32**, so you will not accidentally pay the emulation penalty.

## When to change it

- On a GPU or Apple Silicon machine: **bf16** to save memory and go faster.
- On a CPU-only machine: **fp32**.

## Gotcha

- Precision affects memory more than final quality here. Don't reach for fp32 expecting a smarter model; reach for it when you want exactness or you are on a CPU.
