---
title: 43x faster top-K sampling
date: 2026-05-05
tags: [sampling, kernel, decode, performance]
summary: Replaced the per-token selection sort in sample_token with a min-heap. Bit-exact result, 43x faster, ~1.2% of decode time clawed back.
---

`sample_token` ran a partial selection sort over the 256 logits to pick the top-K. Selection sort is O(K * V): for each of K positions, scan the remaining vocab and pick the max. At V=256 and typical K, that is up to thousands of compare-and-swap ops per token. ~13 microseconds per call.

A min-heap of size K does the same job in O(V * log K): walk the vocab once, push when the heap is small or when the new value beats the current min, pop otherwise. ~1 microsecond per call.

## why it is bit-exact

Selection sort returns the K largest values; a min-heap of size K, after one pass over the input, holds those same K values. The order inside the result does not matter for sampling; the threshold (the smallest of the top-K) is identical between the two algorithms. So downstream temperature scaling, softmax, and multinomial draw see the same input distribution. No tolerance, no oracle drift.

## numbers

- ~13 us -> < 1 us per call (43x).
- ~1.2% of total decode time saved on the 9800X3D at pos=200.
- Code lives in `engine/src/model.c::sample_token`.

## why it was worth doing

Sampling looks like the kind of thing you do not need to optimize: the matmuls dominate. But on a 0.83 ms decode budget, every microsecond outside the matmul shows up. A 1.2% trim is small in isolation and meaningful when you are stacking ten of them. The branchless heap also has predictable cache behavior, which the selection sort did not.

## takeaway

Read the hot path linearly with a profiler. Anything that is not a matmul and runs every token is a candidate. Replace O(K*V) with O(V log K) when V is fixed and small (V=256 here, byte vocab).
