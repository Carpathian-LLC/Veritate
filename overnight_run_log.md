# Overnight run log

## Headlines
- 2026-06-13: 200M Market LLM (`marketllm_200m`) training on MPS bf16 — byte model on the
  1.31B-token crypto series corpus, `model_type=statistical`, batch 12, seq 1024,
  ckpt/val every 500. Launched via the canonical `trainer_runner` path (`.plugin_pid.json`
  written -> dashboard shows the run). total_steps corrected to ~165k (Chinchilla 4B tokens).
- Market subsystem cleaned: dead GBDT serving removed, package deduped, docs corrected,
  experimental S3-URL feature deleted, codec scalar oracle preserved. Tests 8/8 green.

## 2026-06-13 — Market LLM 200M launch + subsystem cleanup

Goal: train a byte-level Veritate LLM on raw market tape (Chinchilla-optimal for the data on
hand), clean the market subsystem, and wire the dashboard so the run is visible.

Corpus: full byte-series corpus built from all `external_data` (no caps) via
`tools/build_series_corpus.py` -> crypto 1.31B tokens (200 pairs) + stocks 0.012B (501
tickers). 3 bytes/bar (`series_codec`). Chinchilla math: 200M wants ~4B tokens (~3 epochs),
400M ~8B, 800M ~16B. Decision: train 200M now on current data; pull more later toward 16B
for an eventual 800M.

Training: `marketllm_200m`, scratch, corpus=crypto, batch 12 (fixed), seq 1024, n_chunks 2
(effective 2048/step -> 24,576 tokens/step), bf16, act-ckpt on, `model_type=statistical`
(skips meaningless language probes), ckpt/val every 500. Launched through the
`trainer_runner` spawn path (PID file for dashboard recovery), NOT mutating the synced
trainer manifest. 8-bit AdamW unavailable (no bitsandbytes) -> torch AdamW fp32; fine at
200M. Early: step 1 loss ~5.8 (random init), step 50 loss ~2.46, ~5.9k tok/s, ~4.1 s/step.
Log: `.plugin_run.log`; per-step CSV: `models/marketllm_200m/train.csv`.

Sizing correction (caught while monitoring): n_chunks=2 doubles tokens/step, so the initial
total_steps=328k = 8B tokens = ~6 epochs = ~16 days (over-Chinchilla, memorization risk).
Corrected to ~165k steps = 4.05B tokens = Chinchilla-optimal 200M (~3 epochs). ETA ~8 days
at the measured rate; usable checkpoints every 500 steps; stoppable anytime.

Cleanup (subagents, each read preflight first): removed dead GBDT serving routes
(/market/forecast,scan,live,hindcast,backtest,status) + helpers/constants; deduped the
base->minutes map, symbol lists, MINUTES_PER_YEAR; corrected doc drift (false isolation +
gating claims, byte-model-only framing); deleted the experimental S3 corpus dashboard
feature (panel + `/market/corpus` + `market_corpus_s3_url`) with data kept; restored the
`series_codec` scalar oracle the dedup pass over-deleted (preflight rule 24). py_compile +
8/8 tests pass.

UI: market.html multi-instrument minis -> separate actual/model lines with a legend; live
chart reworked (model-pick line, log scale, moves to top when live / hidden when off);
latest-checkpoint badge. Corpus library: coming-soon placeholders for market data +
category grouping (Chatting/Autocomplete/Facts/Statistics) with blue headers.

Docs: `documentation/training/launching_runs.md` (launch contract: model_type, required
flags, gotchas). `market_llm_data_manifest.md` at repo root (S3 hosting paths, pulled by
hand).

Do next: relaunch at 165k steps; confirm steady-state ETA; watch loss + checkpoints + disk
(ckpt every 500 = many dump dirs). Stand up the 1s/tick big data pull toward 16B for an
800M run (needs a disk go-ahead; ~TB, 1.3TB free).
