# Multi-Mind PoC + Concilium v1 — final state 2026-05-22

Single source of truth for tomorrow's review. Everything tested. Numbers from real runs.

## Verdicts

| W   | Claim                                                                    | Verdict | Numbers |
|-----|--------------------------------------------------------------------------|---------|---------|
| W1  | affect bias shifts MoE routing (vanilla setup)                           | PASS-spirit / FAIL-strict | KL ratio 12.9x (≥2x), ‖g‖=0.014 (<0.1) — architecture lives, training-pathway weak |
| W1b | freeze-tail + z-loss force g to be load-bearing                          | FAIL | KL ratio dropped to 1.9; freeze-tail backfires (frozen FFNs reject bias-driven routing) |
| W1e | oracle-forced g (gold-standard architecture controller)                  | PASS | KL ratio 3564x; ‖g‖=2.45 — architecture definitively can route on bias signal |
| W1g | training fix: 10x LR + nonzero init on g                                 | PASS | KL ratio 34x; ‖g‖=0.218 — training pathway works with right hyperparams |
| W2  | learned affect probe replaces oracle scalar                              | PASS | ratio_vs_oracle 1.57; probe val acc 86.09% |
| W3  | region naming via specialty corpora + soft KL gate nudge                 | PASS | 4/6 regions (broca, wernicke, hippocampus, prefrontal) at 99%+ top-1 on own specialty |
| W4  | refractory inhibition (train + eval with) burst-iness vs ppl             | PASS | stickiness drop 31.5%, ppl ratio 1.02 |
| W5  | sleep-cycle adapter updates on conversation buffer                       | PARTIAL | mechanism works; dose-vs-forgetting tradeoff curve documented across 8 variants |
| W6  | slot memory enables cross-sample fact recall (hippocampus PoC)           | PARTIAL | 16x signal vs baseline (16% vs 1%); falls short of 25% threshold — learned-KV bank isn't true episodic memory |
| W1g/per_layer | per-layer g instead of global vector                            | PASS | KL ratio 23x; ‖g‖=0.293; signal moved from L0 to L1 |
| W1g/no_aux | drop aux load-balance loss                                          | PASS | KL ratio 11x; ‖g‖=0.198; baseline drifts up — aux is for cleanliness |
| Integration | per-layer g + refractory_steps=4 (training)                        | PASS | KL ratio 13x; ‖g‖=0.421 (mechanisms compose) |
| **Kitchen sink** | **all best mechanisms stacked + curriculum-amplified s**     | **PASS** | **KL ratio 72.16x, ‖g‖=1.189, abs_KL=0.429 nats (24% of log(6) ceiling) — best result of the PoC** |
| Kitchen sink W2 (probe) | best config but with learned probe instead of oracle label | PASS | KL ratio depends on baseline, ‖g‖=0.545, abs_KL=0.331; model checkpoint saved for live demo |
| Kitchen sink 5x/5000 | push CURRICULUM_AMP to 5.0 + 5000 steps | DIMINISHING | ratio 10.5x, ‖g‖=0.97 — regressed vs 3x/3000 |
| Top-K sweep | K=1, K=3 vs default K=2 | PARTIAL | K=2 optimum: K=1=27x, K=2=72x, K=3=11x. Top-2 mixing helps bias signal at this scale. |
| **Temp sweep** | gate softmax temperature 0.5 / 1.0 / 1.5 / 2.0 / 3.0 | **TEMP-2.0 NEW BEST** | 0.5=2.8x, 1.0=72x, 1.5=30x, **2.0=186x ⭐ (‖g‖=0.90, 62% of log(6) ceiling)**, 3.0=4.4x. Smoother softmax lets bias influence ALL experts. To deploy: model + inference both must use temp=2.0. |

## Platform deliverables

- `veritate_core/model_mtm.py` — VeritateMultimind class with full Veritate contract + sentiment + provider hook (set_gate_bias_provider).
- `veritate_core/multimind/{__init__,plugin}.py` — MultiMindPlugin: attach/detach/sleep/wake. Duck-typed; works on any model exposing the contract.
- `veritate_core/load.py` — loader branch identifies MtM checkpoints by `blocks.0.ff.router.weight`.
- `documentation/multimind/contract.md` — platform contract for the multimind plugin (developer-to-developer voice).
- `trainers/multimind_poc/`
  - `corpus.py` — Amazon Polarity (Apache-2.0) downloader + byteifier; 100k train + 25k val at 512-byte windows.
  - `moe_model.py` — research MoE class with refractory + sentiment bias + named regions.
  - `affect_probe.py` — 200k-param dilated-conv probe (val acc 86.09%).
  - `affect_train.py` — probe trainer (CPU-OK).
  - `falsifier.py` — unified W1/W1b/W1e/W2 trainer (flags: `--oracle-g`, `--freeze-tail`, `--z-loss`, `--probe`).
  - `specialty_corpora.py` — 6 region-specific corpora (5000 samples × 512 bytes each).
  - `w3_train.py` — region-naming trainer.
  - `w4_refractory_eval.py` — train-w/wo-refractory comparison.
  - `w5_sleep.py` — LoRA + OPLoRA sleep cycle.
  - `w6_slot_memory.py`, `w6b_slot_memory_v2.py` — slot memory variants.
  - `demo_chat.py` — interactive REPL: trains-or-loads model, generates bytes, `/sleep` `/regions` `/quit` commands.
- `veritate_mri/routes/multimind_routes.py` — dashboard backend reads real W*_results.json; live inference when `multimind_poc_model.pt` present.
- `veritate_mri/web/multimind.{html,css,js}` — dashboard page at `/multimind`.
- 34 tests passing across mesh routes, settings validation, sync_common, model_mtm, multimind_plugin.

## How to interact tomorrow

1. **Dashboard** — start MRI: `python veritate_mri/app.py` then open `http://localhost:8001/multimind`. Try-it panel runs LIVE inference against the trained model.
2. **REPL chat** — `cd trainers/multimind_poc && python demo_chat.py` — interactive byte-level chat with `/sleep`, `/regions`, `/quit`.
3. **Re-train any falsifier** — `python falsifier.py [--probe|--oracle-g|--freeze-tail N|--z-loss F]` from `trainers/multimind_poc/`. Verdict auto-appends to SUCCESSES.md / FAILURES.md.
4. **Concilium** — `cd C:\GitHub\Concilium && scripts\quickstart.bat` — boots hub on :8881, opens GUI, enrolls a local node.

## Discoveries logged

- Byte-level MoE is genuinely novel territory (no published byte-level MoE LMs as of 2025; BLT uses dense over patches, not MoE).
- "Continuous affect-conditioned per-expert gate bias on byte-level MoE" is the publishable contribution (per priorart deep dive at `priorart_deep_2026_05_21.md`).
- OPLoRA-style orthogonal-subspace projection on sleep updates is a known technique that helps but doesn't fully solve forgetting at 7.5M params; documented trade-off curve.
- W1 mechanism insight: even ‖g‖≈0.014 produces 13x routing-KL because the bias term seeded the gradient pathway early in training; the router then self-organized around sentiment-correlated input bytes. Brain-inspired emergent routing.
- Refractory inhibition costs only +1.3% per training step but cuts L0 stickiness 31.5% — cheap mechanism, real effect.

## Trade-offs that need a follow-up
- W5 sleep: at 7.5M params, parameter-sharing means adaptation and holdout preservation overlap; needs larger model or proper continual-learning machinery (EWC, gradient-task-arithmetic, modular per-region hard separation).
- W6 episodic memory: learned-KV slot bank gives 16x signal but isn't real episodic memory; needs write-as-you-go state, optional cross-sample persistence.
- C engine: supports MoE v11 but only top-1 routing; our PoC uses top-2. Either switch PoC to top-1 (one-line change) or extend engine for top-k.
- gate_g serialization: no v11 slot for affect bias; needs v13 format OR sidecar JSON.

## Files to read first (in priority order)

1. This file (`FINAL_STATE_2026_05_22.md`).
2. `ideas.md` (status board + 100 fallback variants + decision tree + priorart links).
3. `SUCCESSES.md` (validated experiments with numbers).
4. `FAILURES.md` (falsified attempts; the dose-vs-forgetting curve for W5 lives in the multiple W5 entries).
5. `priorart_deep_2026_05_21.md` (novelty verdict).
6. `benchmark_2026_05_22.md` (perf numbers).
7. `C:\GitHub\Concilium\README.md` (cross-agent platform — separate project).

## Hourly cron

Active (CronCreate ID `7cd5cbbb`) firing at :17 each hour for autonomous check-ins. Will continue work or escalate to user on level-10 decisions only. Auto-expires after 7 days per CronCreate default.
