---
title: m3 BPTT window cap for VRAM
date: 2026-05-05
tags: [m3, multimind, vram, training]
summary: Configurable BPTT window so the m3 trainer fits within VRAM on smaller GPUs.
---

The m3 trainer was OOMing on cards below the headline target. We capped the BPTT window via a runtime config knob rather than reducing batch size, since the throughput hit is smaller and the loss curve held.

## findings

- Window cap of 256 steps was the sweet spot for 12 GB cards.
- Loss within noise of the uncapped run after 5k steps.
- VRAM reduction was roughly linear with the cap.

## open questions

- Does the cap interact with the slot table's eviction policy?
- Long-context eval has not been re-run; needs another pass before we trust the result.
