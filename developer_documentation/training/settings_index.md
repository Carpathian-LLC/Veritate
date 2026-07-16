---
title: Training Settings Index
summary: Every knob on the Training tab, one line each. Canonical reference for `/trainers/run` args and trainer manifest defaults.
tags: training, settings, reference
---

# training settings index

Ground truth for the schema is [`veritate_mri/web/index.js`](../../veritate_mri/web/index.js) (`TRAINER_SCHEMA`) and the per-trainer `manifest.json` (e.g. [`trainers/veritate_80m/manifest.json`](../../trainers/veritate_80m/manifest.json)). Long-form wiki pages live in [`veritate_mri/data/wiki/settings/`](../../veritate_mri/data/wiki/settings/).

## required (every run)

- **name** — your model slug; the size suffix is auto-appended (`chat80m` → `chat80m_85m`).
- **corpus** — training data file, or a mix like `chat_v1:0.4,chat_v2:0.4,chat_v3:0.2`.
- **size** — headline param count. Multi-size trainers show a dropdown; single-size ones (80m, 200m) hide it.
- **precision** — `bf16` (half memory, native on 5070/A/M) or `fp32`.
- **description** — free text saved into `config.json`. Auto-filled if blank.
- **model_type** — `language` | `code` | `statistical` | `other`. Gates the language dump suite (fluency, reading, grammar, MMLU, HellaSwag, IFEval). `language` and `code` both get the suite; `statistical`/`other` skip it (rule 24a).
- **version** — optional revision tag (`v1`, `v2a`). Shows in description, not folder name.

## recipe + architecture

- **recipe** — one-click preset. `balanced` (muon+dense+wsd), `efficient byte-native` (muon+patched+wsd), `long-conversation` (muon+recurrent+wsd), `classic` (adamw+dense+cosine), `custom` (no-op). Only touches optimizer, trunk, lr_schedule.
- **optimizer** — `muon` (measured 1.60x fewer training bytes vs adamw on this platform, `successes.md` 2026-07-03) or `adamw`.
- **trunk** — architecture. `dense` (canonical transformer), `patched` (byte-patch, 1.82x faster to same quality), `recurrent` (constant-state, O(1) decode), `hybrid` (patched+recurrent, best measured quality, 1.70x vs dense), `looped` (weight-tied depth), `memory` (long-context device).

## training loop (standard)

- **total_steps** — stop condition. More = lower loss, more wall time.
- **batch_size** — rows/step. Higher = faster + more VRAM.
- **seq** — bytes of context per row. Bigger = more per-step learning + quadratic attention cost on no-flash backends.
- **n_chunks** — TBPTT chunks per step. Increases bytes/step without changing VRAM.
- **base_lr** — peak LR. Typical from-scratch: `1e-4` to `5e-4`. SFT/continue: `1e-5`.
- **min_lr** — floor LR after decay. Typical `1e-5` to `1e-6`.
- **warmup_steps** — linear ramp from 0 to peak.
- **lr_schedule** — `cosine`, `linear`, `constant`, `wsd`.
- **wsd_decay_frac** — WSD tail fraction (typical `0.1`). Only when `lr_schedule=wsd`.
- **wsd_decay_kind** — `sqrt`, `linear`, `cosine`. Only when `lr_schedule=wsd`.
- **weight_decay** — regularization strength. `0.1`–`0.18` typical.
- **beta1** / **beta2** — AdamW moments. `0.9` / `0.95` for LMs.
- **label_smoothing** — `0` off, `0.05`–`0.1` reduces overconfidence.
- **grad_clip** — per-step gradient cap. `1.0` typical.
- **ckpt_every** — steps between checkpoints.
- **log_every** — steps between `train.csv` rows.
- **eval_every** — steps between validation runs.
- **eval_iters** — batches per validation. Higher = smoother val curve.
- **seed** — RNG seed.

## shape (plugin-set, rarely edited)

- **hidden** — transformer hidden dim.
- **layers** — depth.
- **ffn** — FFN inner dim (usually 4×hidden).
- **heads** — must divide hidden.
- **vocab** — `256` for byte-level.
- **rope_base** — RoPE theta (`10000` default; higher = longer-context extrapolation).

## experimental / advanced

- **variant** — dir suffix (`_sparse`, `_qat`).
- **n_predict** — MTP head count (`1` vanilla, `2`/`4` for multi-token prediction).
- **mtp_aux_weight** — loss weight on auxiliary MTP heads.
- **router_aux_loss_coef** — MEGA/MoE load-balance coefficient (`~0.01`).
- **alpha** — M1/M3 adapter write strength.
- **inject_layer** — M1/M3 injection layer (`-1` = mid-stack auto).
- **init_from** — M1 base checkpoint to inherit.
- **bptt_window** — BPTT depth in chunks. `1` frozen, `4` balanced, `n_chunks` full BPTT (max VRAM).
- **quant_mode** — MEGA weight quant (`int8`, `int4`, `ternary`). Only when QAT enabled.
- **n_experts** — MEGA MoE experts per block.
- **router_topk** — MEGA experts firing per byte.
- **router_aux_loss** — MEGA load-balance coefficient.
- **slm_ref** — reference model for selective loss.
- **slm_keep** — fraction of tokens kept in selective loss (`0.6` default).
- **state_rule** — recurrent update rule (`gla` default).
- **state_carry** — `off` (independent chunks) or `chunks` (thread state left-to-right within a step). `chunks` needs a `hybrid`/`hybrid_moe`/`recurrent` trunk.
- **activation** — FFN activation (`gelu` default; `silu`/`relu` optional).
- **l1_lambda** — L1 penalty on captured features. `0` off.

## core-plugin checkboxes

- **use_act_ckpt** — activation checkpointing. Trades ~50% activation memory for ~30% throughput. Only when memory-bound (rule 24b).
- **use_8bit_adam** — bitsandbytes AdamW8bit. CUDA-only in practice (silent fp32 fallback on MPS). Saves ~6 bytes/param on optimizer state.
- **qat_enabled** — quantization-aware training (fake-quant matmuls + embeddings + RMSNorm). Continues from an existing model, creates `<name>_qat`.

## continue-mode extras

- **resume** — model to continue.
- **corpus** (optional override) — leave blank to reuse the original.

Everything else on the continue tab mirrors the loop knobs above; shape and architecture are locked from the resume target's `config.json`.

## perfect chat recipe (this box: RTX 5070 12 GB + 32 GB RAM)

This is the fastest sensible config for a genuinely conversational byte model **given the current chat corpus (`chat_v1`+`chat_v2`+`chat_v3` = 4.9 GB, see [`chat_model_80m_plan.md`](chat_model_80m_plan.md))**. Do NOT run this against a mostly-code corpus; the model will not talk.

**Pretrain phase** (`/trainers/run` with `id=veritate_80m`, flow=`scratch`):

| knob | value | why |
|---|---|---|
| name | `chat80m` | ← rename to taste |
| size | `80m` | 85 M params, fits comfortably in 12 GB VRAM |
| precision | `bf16` | native on 5070, halves activations |
| model_type | `language` | full language dump suite |
| optimizer | `muon` | 1.60x fewer bytes to target (2026-07-03) |
| trunk | `hybrid` | 1.70x vs dense, O(1) global-state decode |
| corpus | `chat_v1:0.30,chat_v2:0.30,chat_v3:0.10,fineweb_edu:0.20,openwebtext10g:0.05,py_code_v1:0.05` | chat template present from step 0; keeps enough web/prose for common knowledge |
| total_steps | `30000` | ≈ 0.8–1.7 B tokens at 60k tok/s ≈ 4–8 h |
| batch_size | `16` | 5070 12 GB fits this with hybrid + bf16 comfortably |
| seq | `1024` | matches manifest, fixed shape |
| n_chunks | `2` | amortizes optimizer step, no extra VRAM |
| bptt_window | `2` | balanced grad flow through chunks |
| base_lr | `3e-4` | from-scratch peak (NOT the manifest default `1e-5`, which is for SFT) |
| min_lr | `3e-5` | 10× decay range |
| warmup_steps | `500` | matches manifest |
| lr_schedule | `wsd` | measured winner |
| wsd_decay_frac | `0.04` | manifest default; short sharp tail |
| wsd_decay_kind | `sqrt` | manifest default |
| weight_decay | `0.1` | standard for LMs |
| beta1 / beta2 | `0.9` / `0.95` | LM standard |
| label_smoothing | `0.05` | mild overconfidence tamer |
| grad_clip | `1.0` | LM standard |
| ckpt_every | `500` | resumable + dumps for the Models tab |
| log_every | `50` | smooth `train.csv` |
| eval_every | `250` | val curve without stealing compute |
| eval_iters | `64` | manifest default |
| seed | `0` | reproducible |
| use_act_ckpt | `off` | 12 GB fits `batch=16`; leaving it off buys ~30% throughput |
| use_8bit_adam | `off` | not needed at 85 M; fp32 moments are fine in 12 GB |
| qat_enabled | `off` | pretrain first, QAT afterward if you export to `int8` |

**SFT phase** (after pretrain, flow=`continue`, `resume=chat80m_85m`): swap corpus to `chat_v1:0.40,chat_v2:0.40,chat_v3:0.20`, drop `base_lr` to `3e-5`, `min_lr=3e-6`, `total_steps` extended by ~8k. Multi-epoch OK on 4.9 GB (no memorization at 85 M).

**Speed levers already applied on this box** (see `project_85m_training_perf` memory + `chat_model_80m_plan.md`): pinned+prefetch dataloader, RAM-resident corpus, fused AdamW (auto when `optimizer=muon` falls back to AdamW for 1-D params), `F.rms_norm`, in-forward L1. Measured ceiling ~60k tok/s on this 5070.

**What "perfect" cannot mean here.** The measured wall is honest: one 5070 + 85 M params + 4.9 GB chat corpus produces a small, fluent chatbot, not GPT-4. If you want closer to that ceiling, the next lever is size (200 M scale on this same recipe once the 85 M converses), not more knobs on 85 M (`chat_model_80m_plan.md` §honest ceiling).

**Pushback / rule 9a.** If your corpus is still the mixed-code dump from the earlier run, this config will produce a coder that stumbles at chat, not "insanely fluent chat". Confirm the corpus above (or an equivalent chat-dominant mix) exists on disk before launching, or the 4–8 h of compute is wasted.
