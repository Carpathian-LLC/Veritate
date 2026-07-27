---
title: sequence length
date: 2026-07-27
tags: [settings, training]
summary: How many bytes of context the model sees at once, and the context window baked into the trained model.
---

# sequence length

The width of one chunk of text in bytes. Because this platform trains on raw bytes, 1024 means 1024 characters of plain ASCII text, fewer for text with multi-byte characters.

## what it does

Two effects, one temporary and one permanent.

During training it sets how far back the model can look while predicting the next byte. Attention cost on a dense trunk grows with the square of this number, so doubling it roughly quadruples the attention work per row.

Permanently, it sizes the learned position table stored in the checkpoint. A model trained at 1024 has 1024 position slots and cannot attend beyond them later, whatever the inference code asks for. This is why the field appears on a scratch run and not on a continue run: the shape is fixed once the model exists.

## range and default

Any positive integer. The corpus must hold at least `seq * n_chunks + 2` bytes or the loader refuses to build a row.

Every trainer manifest defaults to 1024 except `veritate_10m`, which defaults to 512.

## when to change it

Raise it when the task needs longer context than 1024 bytes, and accept the cost: on a dense trunk that cost is quadratic. The `patched` and `recurrent` trunks exist precisely to make long sequences affordable, so pair a large `seq` with one of them rather than paying full attention cost.

Lower it to fit a bigger batch on a small box, or when the corpus is made of short records anyway.
