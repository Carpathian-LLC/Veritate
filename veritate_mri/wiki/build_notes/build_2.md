---
title: "Build 2: trainers, quantization, and dashboard polish"
date: 2026-05-05
tags: [build, plugins, format, mri]
summary: New 1B-class ternary + MoE trainer (multimind_mega), schema-slot trainer (multimind_m1), expanded quantization modes, focused per-plugin training form, plus VRAM, hosting, and dependency fixes.
---

## versions

- build: 2
- engine: v2.0.0
- mri: v0.1.0
- format: v0.1.0
- plugins: v0.1.0

## what changed

Two new trainers, two new quantization modes, a cleaner training form, and a batch of operational fixes.

- **multimind_mega trainer.** A "moonshot" config that trains a 1B-parameter Mixture of Experts model with ternary weight quantization. Mixture of Experts means each transformer layer has several specialized FFN sub-networks ("experts"); a small router decides which expert handles each byte, so only a fraction of the full 1B weights run per byte. Ternary quantization compresses each weight to one of three values (-1, 0, +1), shrinking the deployed model roughly 5x compared to INT8. Together these target a 1B-class model that fits inside the 96 MB L3 cache after deploy.
- **multimind_m1 trainer.** A schema-slot working-memory adapter. Adds a fixed table of named "slot" vectors that the model reads from and writes to as it scans bytes, so information from earlier in a stream can be reused later. Designed to be initialized from an existing trained base (via the new `init_from` field) and trained on top.
- **New weight quantization modes.** The trainer now supports `int4` (0.5 bytes per weight) and `ternary` (~0.2 bytes per weight) in addition to the original `int8`. Set via the `quant_mode` field on MEGA models. INT8 remains the default for everything else.
- **Per-plugin training form.** The training form on the dashboard now only shows the knobs each plugin actually consumes. MEGA shows MoE knobs, M1 shows slot knobs, M3 shows adapter knobs. The form is also reorganized so checkboxes cluster at the bottom and experimental knobs are flagged "advanced".
- **Form remembers your last config.** Hitting "Start training" now writes the submitted args back into the plugin manifest's `defaults` block, so the form repopulates with your last configuration across sessions. Only schema keys are persisted; per-run fields (corpus, model, step, description) are not.
- **Plugin manifest contract.** `documentation/plugins/contract.md` now lists every recognized field with a sample value, including the MEGA cluster (`quant_mode`, `n_experts`, `router_topk`, `router_aux_loss`, `use_8bit_adam`) and the M1 cluster (`n_slots`, `init_from`, `freeze_base`). Existing manifests are unchanged and continue to work.
- **Model directory names.** The directory naming rule (`<corpus>_<size>_<precision>_<version>`) now accepts an optional trailing variant tag, for example `tinystories_120m_bf16_v1_m1` or `tinystories_1b_bf16_v1_mega`. Old names continue to parse.
- **MRI server now binds 0.0.0.0.** The dashboard is reachable from other machines on the network, not just localhost. Hosting ports were tweaked accordingly.
- **M3 trainer VRAM cap.** The multimind_m3 trainer now uses a configurable BPTT window (the number of bytes it back-propagates through at once), capping VRAM use on smaller GPUs. Set the window size in the form.
- **Torch dependency fix.** `requirements.txt` was pulling a CPU-only build of PyTorch that silently disabled GPU training. Pinned to the correct CUDA-enabled wheel. Reinstall dependencies if you cloned before this fix.

## what you need to do

If you cloned at build 1 and saw CPU-only training despite having a GPU, reinstall:

1. `pip install -r requirements.txt --upgrade --force-reinstall`

Otherwise, nothing required. Existing models, manifests, and configs all continue to load.

To try the new trainers:

1. Open the dashboard and pick "multimind_mega" or "multimind_m1" from the trainer dropdown.
2. Fill in the form. Required fields are marked. Advanced fields are collapsed by default.
3. Click "Start training."

## known issues

- The MEGA trainer runs in pure PyTorch for now. The hand-tuned C/AVX-512 kernels for ternary matmul and MoE dispatch are not yet in the engine, so 1B-class MEGA models cannot be deployed for inference yet. Training works; engine deployment is the next build's work.
- MEGA training throughput on consumer GPUs is bottlenecked by Python-level expert dispatch (per-expert loop, per-step). A 1B/8-expert run on RTX 5070 lands around 600 tok/s. Speeding this up needs grouped GEMM or `torch.compile`; planned, not present.
- The engine is x86-only at this build. ARM64 support is planned, not present.
- Long contexts can drift past the 0.1 ms / byte target on some workloads. Tightening that is on the list.
