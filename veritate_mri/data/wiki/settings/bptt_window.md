---
title: BPTT Window
summary: How far back in a long sequence the model actually learns from each step; smaller windows save memory, the full window learns the most but costs the most.
tags: training, settings
---

# BPTT Window

When a step processes a long sequence, it is broken into chunks. The BPTT window sets **how many of those chunks stay "live" for learning**, meaning the trainer traces its mistakes back through them and updates accordingly. BPTT is short for "backpropagation through time," the process of learning across a sequence.

Picture reading a long paragraph and then being asked to fix your understanding: do you re-examine only the last sentence, or the last few sentences, or the whole paragraph? The window is how far back you look.

## Why it matters

Looking back over more chunks lets the model learn longer-range patterns, but every live chunk has to be held in memory for the backward pass. So the window is a direct memory-versus-learning dial.

## Weak-hardware angle

- **1 (frozen)** : only the current chunk learns. Lowest memory. The earlier chunks still feed context, they just are not traced back through.
- **4 (balanced)** : a middle ground; learns some longer-range structure without a big memory bill.
- **n_chunks (full BPTT)** : every chunk stays live. Learns the most across the sequence but uses the **most memory**.

## When to change it

- On a tight machine, keep it at **1** or **4**.
- Raise it toward the full value only when you have memory to spare and need long-range learning.
