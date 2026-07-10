# research successes

Validated results with the evidence that proved them. Entries are research outcomes, not bug fixes. Entries arrive here from `ideas.md` when an idea clears its falsifier; index at `research.md`.

## 2026-07-10: chat200m chat-phase SFT PASSES all four gates — the applied lessons hold at scale (identity native, copy skill reinforced, RAG copy works, servable)

- Tested: one combined gated chat-phase SFT resuming chat200m from the base step_20400 to step_24400 (~4k steps, ~390M tokens, 2e-5 WSD -> 2e-6). Mix: chat 50% (v1/v2/v3) / grounded_v3 20% / chat_identity_v1 10% / chat_recall_v1 8% / fineweb+owt 12% anchor. The pre-registered HARD gate (failures.md 2026-07-08 lesson): the conversation-copy needle must NOT erode; all four surfaces scored sampled; roll back if the copy skill drops.
- Gate result, all four PASS (final step_24400, greedy/low-temp, strict read): (1) NEEDLE conversation-copy 1.00 @190B / 0.83 @475B — UP from the base 1.00/0.67, the exact OPPOSITE of the 80M identity-SFT collapse (0.92 -> 0.17); mid-run 21000 hit 1.00/0.92, final 1.00/0.83 (decay-tail noise, still well above base). (2) IDENTITY bare, no persona line: "My name is Veritate" 3/3, "made by Carpathian" 3/3 — the 80M scored 0/8 bare at every temp and needed the serving-side persona crutch. (3) GROUNDED alien-fact read-off-the-page 3/3 greedy (quantum sail -> Lena Voss, Miravel -> Mount Kessler, Brindlemere -> blue honey); the 80M was ~0/3. (4) CHAT register natural, empathy INTACT ("rough day" -> "Yeah, it was pretty tough. But I'm glad to have you home") where the 80M lost empathy after its identity SFT.
- Why it worked (the thesis, confirmed): the copy + identity skills were dosed into the PRETRAIN from step 0 (grounded_v3 2.5%, identity 2%, recall 4%), so the base already sat at 1.00/0.67 copy and native identity; the SFT's diverse grounded data (invented entities, 4 families) REINFORCED copying rather than replacing it with a narrow template. In-pretrain dosing beats late-phase SFT for a small model, exactly as failures.md 2026-07-06/08 retry conditions predicted.
- Servable (the platform requirement): v13-hybrid fp16 bin exported (541MB, act_boost=1, 20 blocks / 16 global), capabilities.chat marked trained@24400, loads + chats on BOTH the C engine and PyTorch backend through /hybrid/chat. Honest weakness measured: field SELECTION inside a compound context sentence is imperfect (asked what a bridge "spans", it returned the completion year instead of the river — read the passage, mis-selected); single-fact contexts are 3/3. This is grounded_v3 family-3 (multi-fact selection) territory, a known next-dose direction, not a regression.
- Meaning: chat200m is a categorical step up over chat80m on every axis the 80M was weak (self-knowledge, reading handed facts, empathy retention, copy skill) AND avoids the interference failure that the 80M's late SFT caused. The strategy of baking every ledgered lesson into the pretrain and gating the SFT hard is validated end to end at 270M.
- Run: `models/chat200m_200m/` steps 0-24400 one lineage (base 20400, chat SFT 24400). Needle JSONs: session scratchpad (`needle_200m_20400/21000/24400.json`). Transcripts in `worklog.md` 2026-07-10.

## 2026-07-09: chat200m pretrain clears the 80M's finished pretrain well before its own ends (the scale-up works)

- Tested: the full efficiency + throughput stack at 2.2x the flagship size. `veritate_200m` manifest at trunk=hybrid = 270,510,336 params (measured; manifest "202M" is the dense estimate), muon, bf16, seq 1024, batch 24, n_chunks 4 (the +68% amortization lever, paper 6), 9-stem mix with every ledgered dose from step 0 (chat 12%, py_code 6%, chat_recall_v1 4%, grounded_v3 2.5%, chat_identity_v1 2%, knowledge base 73.5%). 20,400 steps, dashboard-launched, one clean run.
- Result: 2.005B tokens in 39.9h (1.66 days) at a sustained 14.1-14.3k tok/s (paper 6's cost model predicted the operating point to 0.3%). Val fell 1.70 -> 0.812, monotone, zero instability across every 2k milestone; WSD decay tail delivered the expected final drop (0.862 @16k -> 0.812 @20k). Zero DUMP FAILED; all 14 dump families incl. generation.json present at step_20400 (the multicorpus stem fix holds on a 9-stem mix). Optimizer + lineage clean for a resume phase-change.
- Scaling verdict (the falsifiable claim): at matched training tokens the 200m beats the 80M inside the 80M's pretrain window (1.044 vs 1.085 @200M tok; 0.987 vs 0.993 @395M; 0.937 vs 0.954 @590M), then continues past the point where the 80M's pretrain PLATEAUED (0.942 at its full 737M-token budget) down to 0.812 with 1.2B+ more tokens spent. The bigger model clears the smaller one's finished pretrain before its own is 40% done. Caveat: single run per model, different corpus mixes (the 200m's val overlaps its training mix only 37% vs the 80M's 68%, so the 200m's lead is conservative); this validates the scale-up recipe, not a scaling law.
- Meaning: the composed stack (paper 1) plus the throughput tuning (paper 6) plus the dosing lessons (failures.md 2026-07-06/08) compose into a stable, on-schedule, 2-day 270M pretrain on one M3 Ultra, with all Round-1 lessons structural from step 0. Base for the chat-phase SFT (grounded + identity + register), which is the surface where chat quality and the RAG copy skill actually get made.
- Run: `models/chat200m_200m/` (steps 0-20400 one lineage). Curve: `train.csv`. Plan + gates: `developer_documentation/training/chat_model_200m_plan.md`. Chronology + throughput analysis: `worklog.md` 2026-07-08/09 and `research/amortizing_the_optimizer_step.md`.

## 2026-07-05: three-phase 80M build produces the first conversing byte model (chat80m)

- Tested: the full scaled pipeline on the campaign's winning stack — `veritate_80m` manifest at trunk=hybrid (121.8M params) + muon + bf16, three dashboard phases via resume: pretrain 30k steps (6-corpus mix, 10% chat so the byte template is in-distribution from step 0; val 1.695 -> 0.942), midtrain anneal 10k (45% chat, 2e-5 WSD sqrt tail; val -> 0.681 on the shifted mix), SFT 8k (pure chat 40/40/20, 1e-5 -> 1e-6; final val 0.647). ~16k tok/s throughout; ~9h wall total; zero DUMP FAILED across 96 checkpoints; optimizer state carried across phases (muon momentum survives resume).
- Result: the model CONVERSES. Greedy CPU decode on the SFT checkpoint: greeting answered naturally ("I'm doing well, thank you. How are you?"), emotional register handled with a dialogue-appropriate follow-up ("What do you mean?"). Conversational form/register at byte level is confirmed at 80M with ~470M total training tokens.
- Honest boundary (measured, same smoke): factual recall fails — "capital of France" degenerates into a repetition loop; the model has answer-shape without world knowledge. Consistent with the documented compute wall (~470M tokens is ~1/100 of a knowledge-bearing budget), amplified by greedy decode. Not a pipeline defect: the register phases worked exactly as designed; knowledge needs tokens/scale (160-200M next), sampling helps loops.
- Meaning: phase recipe validated end-to-end (template-native pretrain -> register anneal -> SFT polish), resume-as-phase-change works (corpus + LR pivots on one lineage), and the hybrid trunk trains stably for 48k steps at 80M-scale in bf16. This is the base for the long-context memory campaign measurements.
- Run: `models/chat80m_80m/` (steps 0-48000 one lineage). Chat smoke: first-conversation transcript above; needle degradation curve: `experiments/v2/longctx/chat80m_needle_curve.json`.

## 2026-07-04: composed trunk (E5 hybrid) beats both parents

- Tested: `trunk=hybrid` = patched local attention + constant-state recurrent global mixer on patch slots, muon, canonical 10M shape (15.9M params at ~dense per-byte FLOPs), fineweb_edu, 12000 steps, single delta vs `e2patched` (its attention-global parent) and vs `e1muon` (dense).
- Result: best final val of all arms (0.9707 vs patched 0.9776, recurrent 0.9900, dense 0.9990); 1.70x wall-clock to dense-final quality; 1.15x to patched-final; ahead of patched at 98/120 matched evals; throughput 79.5k tok/s (113 percent of dense). Falsifier cleared on both conditions.
- Meaning: the composition is not additive-only, it stacks: patch-level compute cost, more params per FLOP, AND the global path carries O(1) state (no KV growth on the long-range component). This is the novel trunk the campaign was aiming at; it becomes the default research architecture going forward.
- Run: `models/e5hybrid_10m_qat/`. Single seed; margins over dense are large, margin over patched (0.007 final, 1.15x) is under the 5 percent band: needs a second seed before the beats-patched claim is reported externally (agent_roe seed rule). The beats-dense claim is comfortably outside noise.

## 2026-07-04: constant-state trunk at byte level (E3) = quality parity with attention

- Tested: `trunk=recurrent` (gated linear recurrence, scalar-per-head decay, O(1) state per head) vs dense attention, both muon, canonical 10M shape (recurrent 10.95M, +8.6 percent from the output gate, disclosed), fineweb_edu, 12000 steps.
- Result: per-step quality BEATS attention (final val 0.9900 vs 0.9990; ahead 117/120 matched evals; reaches dense-final at step 8600). Wall-clock: 18 percent slower per step (57.9k vs 70.3k tok/s, unoptimized chunkwise PyTorch), so at equal wall-clock it sits +0.011 behind: inside the +-0.03 parity band. Falsifier (worse by more than 0.03 at equal wall-clock): survived.
- Meaning: attention is not required for byte-level quality at this scale, and the constant-state trunk's decode memory is O(1) in position vs a KV cache growing without bound. This is the conversation-length lever.
- Run: `models/e3recur_10m_qat/`. Single seed; margin under 5 percent, so the parity claim (not the win claim) is the reportable result per agent_roe seed rule.

## 2026-07-04: boundary-patched trunk at byte level (E2) = 1.82x wall-clock savings

- Tested: `trunk=patched` (SpaceByte-style, global blocks on spacelike-boundary slots, seq/4) vs dense, both arms muon, canonical 10M shape, fineweb_edu, 12000 steps. Patched = 15.0M params at roughly dense per-byte FLOPs.
- Result: patched reaches the dense arm's final val (0.9990) at step 8600 / 6679s = **1.82x wall-clock savings**; final val 0.9776 vs 0.9990 at equal steps; ahead at 105/120 matched evals; realized throughput 128 percent of dense. Falsifier (no bpb win at equal wall-clock, or under 70 percent throughput): cleared on both.
- More parameters per FLOP is real at byte level: the patched arm carries 49 percent more params yet steps faster, because global attention/FFN runs on 4x fewer positions.
- Stack so far (measured, 10M): Muon 1.60x x patching 1.82x = ~2.9x wall-clock vs AdamW dense.
- Runs: `models/e1muon_10m_qat/` (baseline), `models/e2patched_10m_qat/`. Single seed each; margins large.

## 2026-07-03: Muon optimizer at byte level (E1) = 1.60x byte savings

- Tested: canonical 10M byte model, fineweb_edu, 12000 steps, identical arms except optimizer. AdamW final val 1.0375; Muon (2D hidden weights, RMS-matched lr, torch native) final val 0.9990, reaching the AdamW-final quality at step 7500 = 1.60x fewer bytes to target. Ahead at 115/120 matched evals from step 500 on; ~3 percent step-time overhead. Falsifier was under 1.15x: cleared.
- Muon lags during warmup (first ~400 steps); judge nothing before step 500.
- Adopted as the default optimizer for subsequent experiment runs (`optimizer=muon` on `/trainers/run`). Single seed; effect size (60 percent) far exceeds the 5 percent multi-seed threshold in agent_roe.
- Runs: `models/e1adamw_10m_qat/`, `models/e1muon_10m_qat/` (train.csv curves).

## 2026-07-02: curated Python byte corpus at scale

- Built: `trainers/corpus/py_code_v1_train.bin` (12.0 GiB) + val (251 MiB) from codeparrot-clean. 1.55M files, every file passes `ast.parse`, exact-deduped, flat uint8 byte stream compatible with existing loaders. 35 of 54 source shards still unconsumed (more available on demand).
- Plus ~200K GPT-4-quality instruction pairs downloaded (Magicoder OSS-Instruct + Evol-Instruct sets, ~580 MB).
- Evidence: build meta in `py_code_v1_train.bin.meta.json`; delimiter and UTF-8 validity spot-verified.

## 2026-07-03: instruction SFT lifts a base model fast (pipeline proof)

- Tested: chat-template SFT of a 0.5B code base on the downloaded instruction sets, prompt-masked loss, MPS bf16.
- Result: HumanEval pass@1 (held-out, unit tests executed) went 0 percent to 15 percent by step 1000 (~16M tokens). Confirms the SFT pipeline, the eval harness, and that data quality is the operative lever at small scale.
- Caveat: base model was subword (BPE), off the byte-level mission; run retired in favor of byte-level work. Checkpoint: `models/qwen25coder_0p5b_sft/`.

## 2026-07-02: measured training throughput ceiling on M3 Ultra

- Measured: 0.5B transformer, bf16, MPS: ~4,858 tok/s eager, ~6,484 tok/s with torch.compile at fixed shapes (bs32/seq512). 1.5B: ~1,947 eager / ~2,187 compiled. MLX is the same tier as torch-MPS (not a lever). GPU is FLOP-bound at ~70 percent of realizable bf16 compute.
- Implication: wall-clock for any plan computes directly from these numbers; framework changes buy nothing, torch.compile buys ~35 percent, fixed shapes are mandatory (see failures 2026-07-03).
