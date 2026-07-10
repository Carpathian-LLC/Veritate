# trainers upstream changeset — 2026-07-07

Commit-ready delta between the local `trainers/` tree and the canonical upstream repo (`Carpathian-LLC/Veritate-Trainers`, branch `main`), as of the last sync recorded in `trainers/.sync_state.json`. The sync path (`veritate_mri/training/sync/trainers_sync.py`) is **pull-only**: a tarball download over https with per-file install/update/force/adopt/skip actions; `trainers/` is not a git repo locally and the platform has no push capability. Mirror this changeset to the canonical repo in one commit, then run `/trainers/git/sync` (Settings, trainers sources panel) and **adopt** each file so `.sync_state.json` re-baselines without overwriting.

Exclude `trainers/.DS_Store` (Finder junk) and `__pycache__/` from the commit.

## edits made 2026-07-07 (this verification round)

### trainers/veritate_800m/manifest.json — fix launch crash + stale standalone-trainer keys

The 800m trainer.py is the shared vanilla shim, but the manifest still carried the old standalone MTP/RoPE trainer's defaults. It lacked `n_chunks` and `bptt_window`, which `vanilla_trainer.run` reads unconditionally, so any launch crashed with `AttributeError`. Removed keys the vanilla trainer cannot build (`n_predict`, `mtp_aux_weight`, `rope_base`; also non-exportable per the v9/v11 format invariant) and `corpus` (a reserved string flag; a manifest default for it is silently ignored). Batch/steps/warmup/eval cadence re-derived by interpolation between the adjacent 400m and 1b3 manifests; `optimizer=muon` per the measured platform default (research ledger 2026-07-03, 1.60x fewer bytes to target).

```diff
-  "description": "... vocab=256). Defaults enable activation checkpointing and 8-bit AdamW.",
+  "description": "... vocab=256). Defaults enable activation checkpointing and the Muon optimizer; batch/steps/eval cadence interpolated between the 400M and 1.3B trainers.",
-    "corpus": "fineweb_edu",
-    "total_steps": 100000,
-    "batch_size": 64,
+    "total_steps": 30000,
+    "batch_size": 8,
     "seq": 1024,
-    "n_predict": 4,
-    "mtp_aux_weight": 0.1,
-    "rope_base": 10000.0,
+    "n_chunks": 2,
+    "bptt_window": 2,
     "base_lr": 0.0003,
     "min_lr": 3e-05,
-    "warmup_steps": 2000,
+    "warmup_steps": 1000,
     "lr_schedule": "cosine",
+    "optimizer": "muon",
-    "ckpt_every": 500,
+    "ckpt_every": 1500,
-    "eval_every": 200,
-    "eval_iters": 64,
+    "eval_every": 1500,
+    "eval_iters": 8,
```

### trainers/veritate_80m/manifest.json — restore scratch-flow pretrain defaults

The scratch defaults were left at the chat80m campaign's phase-3 SFT values (base_lr 1e-5, wsd, 52k steps); a fresh scratch run at SFT LR wastes the run. Restored pretrain-shaped defaults per the campaign plan (`developer_documentation/training/chat_model_80m_plan.md`: pretrain = cosine to min, launch guide total_steps ~30000; measured bs12/seq1024 kept). base_lr 4e-4 interpolated between the measured 10m value (6e-4) and the 400m manifest (3e-4); min_lr = peak/10 per the family convention. `label_smoothing`/`weight_decay` left as shipped (campaign-measured; see label_smoothing note below).

```diff
+  description: appended "Scratch defaults target a from-scratch pretrain (cosine, Muon); anneal/SFT phases override LR per run."
-    "total_steps": 52000,
+    "total_steps": 30000,
-    "base_lr": 1e-05,
-    "min_lr": 1e-06,
+    "base_lr": 0.0004,
+    "min_lr": 4e-05,
     "warmup_steps": 500,
-    "lr_schedule": "wsd",
+    "lr_schedule": "cosine",
+    "optimizer": "muon",
```

### trainers/veritate_10m, veritate_200m, veritate_400m, veritate_1b3, veritate_3b manifest.json — Muon default

Added `"optimizer": "muon"` to `defaults` (one line each, after `lr_schedule`). Muon is the measured platform default (adopted 2026-07-03; used across all 10m arms and the full chat80m campaign; `optimizer` is already in the dashboard `TRAINER_SCHEMA` and in `vanilla_trainer` `RESERVED_STR_FLAGS`, with AdamW fallback when the platform lacks the helper). **Not** applied to 13b and larger: their AdamW state exceeds unified memory and relies on the mem-planner's NVMe-paged optimizer tier, which the muon path in `build_optimizer` bypasses; `adamw` there is the memory-planner-compatible default.

### trainers/veritate_200m, veritate_400m manifest.json — description accuracy

200m claimed "Defaults enable activation checkpointing and 8-bit AdamW" while setting neither; 400m claimed both while setting only act-ckpt. Descriptions now match the defaults.

## pre-existing local divergence (also to mirror)

Files whose SHA differs from `.sync_state.json` (`modified`) or that upstream has never seen (`untracked`). One line each.

| file(s) | state | rationale |
|---|---|---|
| `common/vanilla_trainer.py` | modified | shared trainer gained reserved flags (`optimizer`, `trunk`, `activation`, `state_rule`, `state_carry`, `slm_ref`, `slm_keep`, `l1_lambda`), trunk dispatch (patched/hybrid/hybrid_moe/looped/recurrent/memory), state-carry streaming path, WSD schedule, muon + NVMe-paged optimizer builders, mem-planner feasibility gate |
| `readme.md` | modified | friendly-tour rewrite matching the current manifest schema |
| `veritate_10m/{manifest.json,trainer.py}` | modified | trainer.py collapsed to the 36-line vanilla shim; manifest carries the measured 10m experiment defaults (bs32/seq512/12k steps/cosine 6e-4, QAT on) + this round's muon default |
| `veritate_80m/{manifest.json,trainer.py}` | modified | shim + this round's pretrain-default restore |
| `veritate_200m/{manifest.json,trainer.py}` | modified | shim + description fix + muon default |
| `veritate_400m/{manifest.json,trainer.py}` | modified | shim + description fix + muon default |
| `veritate_800m/{manifest.json,trainer.py}` | modified | standalone MTP/RoPE trainer replaced by shim; manifest fixed this round (was launch-crashing) |
| `veritate_1b3/{manifest.json,trainer.py}` | modified | shim + muon default |
| `veritate_3b/{manifest.json,trainer.py}` | modified | shim + muon default |
| `veritate_13b/{manifest.json,trainer.py}` | modified | shim; conservative adamw/paged defaults kept |
| `veritate_50b/{manifest.json,trainer.py}` | modified | shim; conservative paged-tier defaults kept |
| `veritate_70b … veritate_1t` (10 dirs, 20 files: 70b, 100b, 120b, 160b, 200b, 250b, 350b, 500b, 700b, 1t) | untracked | new size trainers completing the canonical ladder; all are the same shim + a conservative bs1/seq1024/cosine manifest sized for the mem-planner's NVMe-paged tier, nothing CUDA-specific |
| `veritate_1b/` (3 files), `veritate_85m/` (4 files), `veritate_tool_sft/` (2 files) | deleted locally | superseded by the one-trainer-per-size canonical set (1b3, 80m); upstream deletion is part of this changeset |

## verification run 2026-07-07

- Hooks: every trainer.py is an identical shim into `vanilla_trainer.run`; the single checkpoint path is `save.save(...)` (`common/vanilla_trainer.py:746`), which is `veritate_mri/training/save.py::save` re-exported through `veritate_core.plugin` — full dump battery (probe, classroom, grades, reading_comprehension, math, grammar, reasoning, concepts, surprise, quant_kl, writing_health, generation) on every checkpoint, model-type gated. No trainer writes `.pt`, calls `dump_*`, or appends CSV directly.
- Flags: `state_rule` (gla default; delta/pinned validated in `model_recurrent`, clean `ValueError` on bad value), `state_carry` (off default, `chunks` validated + trunk-gated), `trunk`, `optimizer` all parse on every trainer via `RESERVED_STR_FLAGS`; unknown dashboard flags are dropped by `parse_known_args` (verified with a dashboard-style argv).
- All 19 manifests JSON-parse; all trainer `.py` files ast-parse; all 19 manifests dry-parsed through `vanilla_trainer.parse_args` with the full required key set present.

## edits made 2026-07-07 (200m pretrain throughput round)

### trainers/veritate_200m/manifest.json — measured pretrain defaults

Closes the "200m defaults are experiment-scale" gap below. Grounded by an MPS throughput sweep at the real hybrid shape (270.5M true params as trunk=hybrid, seq=1024, bf16, muon; M3 Ultra 256GB): batch 24 x seq 1024 x n_chunks 4 = 98,304 tokens/step at a measured 14,150 tok/s; act-ckpt off (measured -17% tok/s, peak memory only 4.3GB); n_chunks 4 amortizes the ~1.6s fixed Muon step cost (n_chunks 2 loses 19% tok/s). total_steps 20400 = ~2.0B tokens. base_lr 3e-4 / min_lr 3e-5 by family interpolation (80m 4e-4, 400m 3e-4); wsd_decay_frac 0.2 so a 2B-to-6B extension resumes from a longer stable phase; eval/ckpt cadence stretched to pretrain scale.

```diff
-    "total_steps": 2500,
-    "batch_size": 32,
-    "seq": 256,
-    "n_chunks": 8,
+    "total_steps": 20400,
+    "batch_size": 24,
+    "seq": 1024,
+    "n_chunks": 4,
-    "base_lr": 0.0001,
-    "min_lr": 1e-06,
-    "warmup_steps": 200,
+    "base_lr": 0.0003,
+    "min_lr": 3e-05,
+    "warmup_steps": 500,
-    "wsd_decay_frac": 0.33,
+    "wsd_decay_frac": 0.2,
-    "ckpt_every": 500,
+    "ckpt_every": 1000,
-    "eval_every": 250,
-    "eval_iters": 64,
+    "eval_every": 500,
+    "eval_iters": 16,
```

## known gaps (flagged, not changed)

- `label_smoothing` appears in every manifest and in the trainers contract, but nothing consumes it (`vanilla_trainer` and `veritate_core/model.py` never read it). Implement or remove platform-wide; removing touches the contract and dashboard schema.
- `veritate_3b` total_steps 400000 is aspirational at the measured ~550 tok/s for 2.5B-dense on this box (~90 days); harmless as a default.
- `veritate_core/plugin/optim.py::build_muon` calls `torch.optim.Muon` with no availability guard; torch 2.12.0 (current) has it, older torch would raise at run start. Platform-side, not a trainer file.
- `state_rule`/`state_carry` are CLI-only reserved flags; the dashboard `TRAINER_SCHEMA` has no fields for them yet, so they are deliberately absent from manifests.
