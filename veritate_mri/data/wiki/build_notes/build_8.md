---
title: "Build 8: v12 engine format — MTP byte-0 head + untied lm_head"
date: 2026-05-13
tags: [build, engine, format, mtp]
summary: New engine binary format v12 adds slots for an MTP byte-0 head (mtp.transforms[0], mtp.norms[0]) and an untied lm_head, so plugin-trained MTP models (veritate_85m, future MTP plugins) can deploy to the C engine instead of being PyTorch-only. v9 and v11 binaries load unchanged.
---

## versions

- build: 8
- engine: v1.3.0
- mri: v1.3.0
- format: v1.4.0
- plugins: v1.1.1

## what you have to do

- Existing v9 and v11 model `.bin` files keep loading. No action.
- An MTP-architecture model (anything with `mtp.transforms.*` in its checkpoint, e.g. anything trained via `plugins/veritate_85m`) now exports to v12 automatically when you call `export_checkpoint(name, step)`. Re-export to pick up the engine binary, no retraining.
- The `qat_enabled` workflow is unchanged. The act_boost > 1 guard still applies — a partially-QAT-trained MTP model will be refused at load with the same message as before. Train under QAT long enough that embedding magnitudes shrink to act_boost = 1.

## what changed

- `veritate_engine/v1/src/veritate.h` adds `VERITATE_MODEL_VERSION_MTP = 12` and three model fields (`mtp_present`, `mtp_transform0`, `mtp_norm0_w`) plus a `model_project_byte0` declaration.
- `veritate_engine/v1/src/model.c` accepts v12 in the version table, reads the optional MTP section after `n_out`, skips the tied-from-embed `lm_head_build` when `mtp_present`, and routes every byte-0 logit computation through `model_project_byte0` (memcpy when not present, transforms[0] + norms[0] otherwise).
- `veritate_engine/v1/src/main.c` updates the two bare matmul-against-`lm_head` sites (trace top-predictions, perplexity eval) to project first.
- `veritate_mri/training/export.py` detects MTP keys in the state dict and routes through `_export_checkpoint_mtp`. RoPE trunks still raise — they need a future format with no `pos_emb`.
