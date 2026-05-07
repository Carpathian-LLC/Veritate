---
title: "Build 2 (E): experimental fork starts here"
date: 2026-05-07
tags: [build, experimental, engine, plugins, distillation, arm64, ternary]
summary: Experimental fork from build 1. Adds ARM64/Apple Silicon engine path, ternary BitNet b1.58 (.bin v10), an Ollama-teacher distillation plugin, original build-2 trainers (multimind_mega, multimind_m1), and INT4/ternary QAT modes. The mainline track stays at build 1.
---

## versions

- channel: experimental (the fork marker — single source of truth, drives both the header badge and the footer chip)
- build: 2
- engine: v2.1.0 — adds ARM64/NEON SDOT path and the `.bin` v10 ternary reader. Backward-compatible: still reads `.bin` v3..v9 unchanged.
- mri: v0.1.0
- format: v0.2.0 — adds `.bin` v10 (BitNet b1.58 ternary, 5 trits/byte, decoded to INT8 at load).
- plugins: v0.1.0

## what this build is

Build 1 is the canonical baseline. This build is the **experimental fork** — everything below is a divergent line, kept on the `experimental` branch. Mainline users stay on build 1 and pull experimental work in selectively as it stabilizes.

## what changed since build 1

### engine

- **ARM64 / Apple Silicon support.** NEON SDOT INT8 path. macOS `arm64` build (`veritate_engine/v1/build/build.sh` → `bin/macos/arm64/veritate`). Bit-exact with the scalar oracle.
- **Ternary weights (BitNet b1.58, `.bin` v10).** 5 trits per byte on disk, decoded to `{-1, 0, +1}`-valued INT8 at load time via `load_b_ternary`. ~5× smaller `.bin` than v9 INT8 for the same model. Hot path stays INT8 — same kernels, same speed.
- **Single engine, broad format support.** One binary in `veritate_engine/v1/`, reads `.bin` v3 through v10 (INT8 per-tensor, INT4-packed, per-column INT8, MoD-gated, n_out RMSNorm, `act_boost`, ternary). `v2/` is reserved as an empty scratchpad for future hot-path-changing work.
- **Engine numerical-divergence fix.** `quantize_matmul` in `veritate_mri/export.py` was missing `np.ascontiguousarray(W.T)` — the engine's `prep_b` reads weights as `[in, out]` but PyTorch lays them out `[out, in]`. Without the transpose, INT8 export looked plausible on disk but produced gibberish at runtime. Fix is producer-side; engine code unchanged. All existing `models/*/veritate.bin` need re-export with the patched export.

### trainers and quantization

- **multimind_mega.** "Moonshot" config that trains a 1B-parameter Mixture of Experts model with ternary weight quantization. MoE means each transformer layer has several specialized FFN sub-networks ("experts"); a small router decides which expert handles each byte, so only a fraction of the full 1B weights run per byte. Ternary compresses each weight to `{-1, 0, +1}`, shrinking the deployed model ~5× compared to INT8. Together these target a 1B-class model that fits inside the 96 MB L3 cache after deploy.
- **multimind_m1.** Schema-slot working-memory adapter. Adds a fixed table of named "slot" vectors that the model reads from and writes to as it scans bytes, so information from earlier in a stream can be reused later. Designed to be initialized from an existing trained base (via the `init_from` field) and trained on top.
- **distill_teacher (new).** End-to-end Ollama-teacher distillation flow as a single plugin. Pulls text from any installed Ollama model (e.g. `llama3.1:8b`), packs it as a byte corpus, optionally mixes with an existing public corpus, then trains a vanilla byte-level Veritate student through INT8 QAT → ternary QAT → ternary `.bin` v10 export. The student runs natively on the Veritate INT8 engine. One-button run from the dashboard. `models/_teachers/<sanitized_tag>/teacher.json` records every Ollama teacher used on the machine.
- **New weight quantization modes.** Trainers now support `int4` (0.5 bytes per weight) and `ternary` (~0.2 bytes per weight) in addition to `int8`. Set via the `quant_mode` field on MEGA models. INT8 remains the default for everything else.

### dashboard and UX

- **Per-plugin training form.** Only shows the knobs each plugin actually consumes. MEGA shows MoE knobs, M1 shows slot knobs, M3 shows adapter knobs, distill_teacher shows teacher/corpus knobs.
- **Form remembers your last config.** "Start training" writes the submitted args back into the plugin manifest's `defaults` block, so the form repopulates with your last configuration across sessions. Only schema keys are persisted; per-run fields (corpus, model, step, description) are not.
- **Plugin manifest contract.** `documentation/plugins/contract.md` lists every recognized field with a sample value, including the MEGA cluster (`quant_mode`, `n_experts`, `router_topk`, `router_aux_loss`, `use_8bit_adam`), the M1 cluster (`n_slots`, `init_from`, `freeze_base`), and the distill_teacher cluster (`teacher`, `n_generations`, `mix_with`, `mix_ratio`, `int8_steps`, `tern_steps`).
- **Live-training visibility during long-prep phases.** Plugins that have a long pre-training phase (e.g. distill_teacher's hours-long corpus generation) write `config.json` and a header-only `train.csv` at run start, so the model appears in the Models and Training tabs before any training rows land. A `phase` field in `config.json` advances through `generating_corpus → mixing_corpus → training_int8 → training_ternary → exporting → done`.
- **Model directory names.** The naming rule (`<corpus>_<size>_<precision>_<version>`) accepts an optional trailing variant tag, e.g. `tinystories_120m_bf16_v1_m1` or `tinystories_1b_bf16_v1_mega` or `distill_v1_30m_distill_llama3.1_8b`.
- **MRI server now binds 0.0.0.0.** The dashboard is reachable from other machines on the network, not just localhost.
- **M3 trainer VRAM cap.** Configurable BPTT window (the number of bytes it back-propagates through at once), capping VRAM use on smaller GPUs.
- **Torch dependency fix.** `requirements.txt` was pulling a CPU-only build of PyTorch that silently disabled GPU training. Pinned to the correct CUDA-enabled wheel.

## what you need to do

If you cloned at build 1 and saw CPU-only training despite having a GPU, reinstall:

1. `pip install -r requirements.txt --upgrade --force-reinstall`

To use the new ARM64/macOS engine: `bash veritate_engine/v1/build/build.sh` on the target machine.

To run a distillation: `ollama pull <teacher_tag>` (e.g. `llama3.1:8b`), then on the dashboard pick **Distill Teacher (Ollama → Veritate)**, set `corpus` and `description`, hit Run.

## known issues at this build

- The MEGA trainer runs in pure PyTorch. The hand-tuned C/AVX-512 kernels for ternary matmul and MoE expert dispatch are not yet in the engine, so 1B-class MEGA models cannot be deployed for inference yet.
- MEGA training throughput on consumer GPUs is bottlenecked by Python-level expert dispatch (per-expert loop, per-step). A 1B/8-expert run on RTX 5070 lands around 600 tok/s.
- distill_teacher is **sequence-level KD only**. The teacher's BPE vocabulary doesn't translate to the student's byte vocabulary, so logit-level distillation is not implemented; the student trains under standard byte CE on the teacher's text output.
- Distillation throughput is teacher-bound: a Q4_K_M 8B teacher on M1 generates ~30–40 bytes/sec, so a 3 MB teacher corpus takes ~24 hours. Bigger boxes scale this proportionally.
