---
title: Architecture (Trunk)
summary: The internal shape of the model; different trunks trade speed, size, and how well the model handles long conversations.
tags: training, settings
---

# Architecture (Trunk)

The trunk is the model's internal wiring: the core layout that does the thinking before the final output. Same job (read text, predict what comes next), different internal machinery. The choice affects how fast the model trains, how much it can hold, and how it copes as a conversation gets long.

## The choices (with measured notes)

- **dense** : the standard, canonical transformer. The reliable baseline everything else is compared against.
- **patched** : does its heavy compute once per roughly 4-byte "patch" instead of per byte. Measured **1.82x faster** to the same quality, and fits more parameters at the same speed.
- **recurrent** : keeps a fixed-size running state instead of re-reading everything. Matches attention's quality and stays fast and light no matter how long the conversation gets.
- **hybrid** : patched plus a recurrent global path. The **best measured quality** of all variants (1.70x versus dense).
- **looped** : reuses the same layers over and over for extra depth. Beats dense at equal parameter count but loses to patched and hybrid, and letting it "think longer" at test time did not help (measured 2026-07-05).
- **memory** : a long-context device, meant to hold a long input, not a store of knowledge (measured 2026-07-05).

## When to change it

- **dense** for a safe, well-understood run.
- **patched** or **hybrid** when you want the most quality per unit of compute.
- **recurrent** when conversations get very long and you need steady speed.
