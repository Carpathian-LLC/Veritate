---
title: QuaRot INT4 end-to-end
date: 2026-05-05
tags: [quantization, int4, kernel, avx-512, vnni]
summary: Per-head Hadamard rotation + per-row INT4 quantization, bit-identical AVX-512 kernel, +0.45% perplexity vs INT8 on TinyStories. INT4 ships at 1.15x the speed of INT8.
---

INT4 weights cut model size in half versus INT8 and let more of the network fit in L3. The naive way (just round harder) loses too much quality. **QuaRot** wraps the matmul in a Hadamard rotation that flattens outliers in the activation distribution before quantization, so the same 4-bit grid covers the signal more faithfully.

This experiment took QuaRot from a synthetic 35% reduction (early proof) to a real-weights, bit-exact, end-to-end shipping path.

## the idea

A Hadamard matrix is a 1/-1 matrix with orthogonal rows. Multiplying a vector by one is cheap (sign flips and adds, no multiplies) and rotates the vector into a basis where extreme values get spread across many dimensions instead of concentrating in a few. Quantization is much kinder to a flat distribution than a spiky one, because the few large outliers stop dictating the scale.

We apply a size-64 Hadamard per attention head before the INT4 matmul, then unrotate after. The rotation is a no-op mathematically (orthogonal: H * H^T = I), so the underlying linear operation is unchanged. Only the quantization grid sees a friendlier distribution.

## the kernel

The hot piece is INT4 packed AVX-512 matmul. Two nibbles per byte, unpacked at the last possible moment, then fed to VNNI (`vpdpbusd`) for the int8 dot products.

The trick that made it fast: AVX-512's 64-bit cross-lane permute (`vpermt2q`). After AVX2-style lane-local unpack splits high and low nibbles into adjacent bytes, the permute interleaves them into the layout VNNI wants in one instruction, with no extra shuffle passes. Bit-identical to a scalar INT4 oracle on both decode shapes (`ffn_up` k=768 n=3072 and `ffn_down` k=3072 n=768).

## numbers

- Synthetic kernel benchmark: 35% reduction vs INT8 matmul.
- Real-weights TinyStories 130K val: INT4-QuaRot ppl 1.6640 vs INT8 1.6565 (+0.45%).
- Decode m=1 latency: INT4 0.83 ms vs INT8 0.96 ms on the 9800X3D (1.15x faster).
- Pipeline: `training/export_quarot_int4.py` produces a version-4 `.bin`. Engine reads it via `engine/kernels/x86_64/matmul_int4.c::matmul_int4_vnni_prep`.

## why it was a win

Two boxes ticked at once. Half the weight memory (cache pressure drops, more model fits in L3) and faster matmul (the smaller INT4 tile is easier on the dispatch unit). Quality cost is sub-1%, well below the noise floor at 80M parameters.

## what is still open

The DLA / decision-trace side channel does not currently emit data on the INT4 path because `prep_b_int4` does not retain the row-major weight buffer the trace reader expects. Cosmetic, not correctness.

## takeaway

INT4 is viable for byte-level transformers if you preprocess the activations. Hadamard rotation is the cheapest known preprocessor (no learned parameters, no scale schedule), and the AVX-512 cross-lane permute makes the unpack-to-VNNI pipeline tight enough that INT4 is faster than INT8 on this hardware, not just smaller.
