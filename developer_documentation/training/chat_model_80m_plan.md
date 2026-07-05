# 80M byte-level chat model — build plan

The first SCALED model, built to actually be chatted with (the 10M runs were architecture-measurement rigs, ~100x too small to converse). Uses the winning stack from the 10M campaign on real conversation data.

## architecture (decided, measured)

- Trunk: `hybrid` (patched local attention + constant-state recurrent global mixer). Best of all 10M arms (val 0.9707, 1.70x vs dense, O(1) global decode state). See `research_successes.md` 2026-07-04.
- Optimizer: `muon` (1.60x fewer bytes to target). See `research_successes.md` 2026-07-03.
- Shape: veritate_80m manifest (`layers=12` GLOBAL blocks, hidden=768, ffn=3072, heads=12, seq=1024). At the hybrid trunk this is **121.8M params** (12 global + 2 enc + 2 dec = 16 blocks + recurrent gating); params are cheap on 256GB, FLOPs are the wall, so more params/FLOP is the intended trade.
- Pre-launch gate (preflight 24d): hybrid at 80M shape passed CPU shape+finite smoke (loss/grad finite at T=1024 and T=37). MPS stability smoke (bs12/seq1024, several fwd/bwd, no NaN) MUST run right before launch — the E4 NaN-at-real-width lesson.

## data (built, verified)

~4.9 GB conversation data now on disk (was 6 MB), same byte template everywhere (`<|system|>`/`<|user|>`/`<|assistant|>`/`<|end|>`, NUL delimiter, in-content NUL rejected), three registers:
- `chat_v1` (SmolTalk `all`): 2.0 GB, 597,580 conversations — instruction-following/everyday.
- `chat_v2` (Tulu-3 SFT mixture minus the CC-BY-NC no_robots subset): 2.0 GB, 902,049 conversations — knowledge/precision/refusals.
- `chat_v3` (SODA, CC-BY-4.0, two-party speakers→user/assistant adapter): 838 MB, 1,172,002 casual dialogues — the plain chit-chat register ("REALLY good at chatting", not just Q&A).

## training phases (recommended mix)

| phase | corpus spec (multicorpus `stem:w,...`) | LR | steps (guide) |
|---|---|---|---|
| pretrain | `fineweb_edu:0.68,openwebtext10g:0.15,chat_v1:0.04,chat_v2:0.04,chat_v3:0.02,py_code_v1:0.07` | cosine to min | ~25-40k (~0.8-1.7B tok, ~10-20h at ~23k tok/s hybrid) |
| midtrain (anneal) | `chat_v1:0.20,chat_v2:0.20,chat_v3:0.05,fineweb_edu:0.45,openwebtext10g:0.10` | low, WSD decay tail | ~8-12k |
| SFT | `chat_v1:0.40,chat_v2:0.40,chat_v3:0.20` | small | multi-epoch OK (4.9 GB supports it without memorization) |

Chat template is present from pretrain step 0 (10% chat in the mix) so the model never sees the tags as out-of-distribution at SFT.

## launch (dashboard, when GPU free)

`POST /trainers/run` with `id=veritate_80m`, args: `name=chat80m`, `size=80m`, `precision=bf16`, `optimizer=muon`, `trunk=hybrid`, `corpus=<pretrain spec above>`, `total_steps≈30000`, `model_type=language`, plus manifest defaults (batch/seq/schedule). Consider `batch_size` up from the manifest's 12 (RAM is abundant) for throughput; keep fixed shapes (MPS 24c).

## honest ceiling

One M3 Ultra will not produce an Opus-level model — a documented compute wall (`research_failures.md` 2026-06-27), not a tuning gap. Target: a small, fast, genuinely conversational byte model, made the best-possible-per-FLOP by the efficiency stack. If the 80M base converses, scale to 160-200M and fold in the DeepSeek levers (MoE for capacity-per-FLOP, MTP for byte-signal density) at the scale where their evidence begins (`efficient_architecture_research.md` wave 3).
