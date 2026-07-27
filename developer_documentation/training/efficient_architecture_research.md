# efficient architecture research program

Research map for a byte-level model that is foundationally cheaper to train and run, holds full conversations, and writes knowledge into memory at inference time instead of via full-corpus gradient re-feed. Outcomes land in the root ledgers `../../successes.md` / `../../failures.md` (indexed by `../../research.md`). Nothing here is declared impossible; every lever carries a falsifier and gets tested.

## ranked levers (2026-07-03 literature pass, citations at bottom)

| lever | expected win | evidence scale | engine export | status |
|---|---|---|---|---|
| Muon optimizer (2D weights) | 1.5-2x tokens-to-loss | 135M-4B + production | yes (trunk unchanged) | E1 running (adamw arm live, muon arm queued) |
| Byte patching (SpaceByte-style boundary rule) | 3-6x effective train, 3-4x decode FLOPs | 250M at this FLOP class | v12 format needed | E2 built + smoked (`trunk=patched`), queued |
| Constant-state trunk (gated linear recurrence, RWKV/GLA class) | O(1) decode state vs KV cache: ~1000x memory, ~7x FLOPs at 32k-byte conversations | byte-level proof at 353M (MambaByte) | v12 viable (SSD kernel prototype in-tree) | E3 built + smoked (`trunk=recurrent`), queued |
| Surprise-gated neural memory (Titans MAG class) | knowledge acquisition without gradient re-feed | 170M floor, below that unproven | PyTorch-only | E4 queued |
| Teacher-distilled + difference-sampled data | ~2x pretrain compute | 200M-1.2B | yes | after E1/E2 |
| MoR adaptive depth | ~1% NLL at equal FLOPs, 2x throughput | 135M-1.7B | no | second wave (overlaps patching) |
| Fine-grained MoE | large only at big budgets | extrapolated | no | skip (MPS gather/scatter unproven) |
| Ternary training below 3B | negative (2x width penalty) | 100K-70B | export path exists | killed for training, kept for export |
| Backprop-free rules | none demonstrated on LM | toy only | n/a | killed (see failures ledger) |

## wave 2 levers (2026-07-04 literature pass: data, distillation, consolidation)

| lever | expected win | evidence | cost here | status |
|---|---|---|---|---|
| Surprise-selective loss (RHO-1 class: mask low-excess-loss tokens vs a reference model) | 5-10x domain, +6.8 percent general at 1B | RHO-1 2404.07965, ESLM 2505.19893 | near zero: surprise already logged, e1muon is the reference | E6 planned (EXP A arm 3) |
| Curriculum into WSD decay (ascending quality; instruction tail in decay; min_lr at 1/3 peak) | +1-2 percent at 1.5B, standard in 2025-26 small-model recipes | 2511.18903, MiniCPM, Trillion-7B, 2406.14491 | data ordering + one hyperparam | E6 planned (EXP A arm 2) |
| Sleep consolidation (replay session, self-distill fast-weight-ON logits into slow weights, reset fast weights) | GSM-Infinite 74.2 to 81.2 at 2B; beats SFT at 3.6-4.8x less wall-clock | 2605.26099, 2606.03979, Letta 2504.13171 | hours at 10M; composes with E3+E4 | E5 planned (after E4 verdict) |
| Sequence distillation at volume (14B generates, 72B judges) | validated in-project (0 to 15 percent HumanEval; 800M chat) | 2510.01631: small rephrasers beat 70B generators | teacher throughput-bound (~10 MB/day at 32B; use 14B) | decay-tail data for E6 |
| Byte-level logit KD (ULD/ALM) | real at 1B+ warm-started; zero evidence at or below 100M from-scratch | ALM 2503.20083, BLD 2604.07466 | high (Ollama exposes top-k only) | deferred; retry if seq-KD + SLM plateau |
| Byte tax quantified | plain byte trails BPE ~4.5 pts at compute-matched 180M; parity holds only WITH patching | 2605.12928, SpaceByte, BLT | n/a | patching confirmed as entry fee (matches E2) |

## wave 2 recursion levers (2026-07-04: test-time-compute axis)

| lever | expected win | evidence | status |
|---|---|---|---|
| Looped global block on the patched trunk (weight-tied, per-loop gain, randomized R, test-time R sweep) | 1.5-3x param efficiency on reasoning-flavored evals; ~0 on raw bpb; patching's seq/4 slots absorb the loop FLOP cost | Ouro 2510.25741 (1.4B x R4 = 4B-dense reasoning), RRT 2410.20672, 2502.17416; unproven under 135M | E7 planned (`trunk=looped`) |
| MoR-lite per-slot recursion router (expert-choice, fixed capacity = fixed shapes) | recovers loop FLOPs, ~2x throughput at parity | MoR 2507.10524 at 135M-1.7B | E8, only if E7 shows signal |
| Pause/filler bytes | small, QA-only | 2310.02226 at 130M/1B | deferred |
| COCONUT latent CoT | needs curated CoT curriculum | 2412.06769 at 124M | skip; retry with byte CoT SFT data |
| TRM/HRM port, student self-play below 1B | none on LM | see failures ledger | killed |

Honest bound from this axis: recursion adds knowledge MANIPULATION, not storage (Ouro's MMLU regression). It is ~10-15 percent of the "frontier intelligence in a tiny model" ask, multiplicative with the rest of the stack. Knowledge lives in params, the E4 memory, and retrieval.

## wave 3: DeepSeek-class levers (2026-07-04)

Literature pass over the DeepSeek reports (V2 2405.04434, V3 2412.19437, DeepSeekMoE 2401.06066, R1 2501.12948) plus independent small-scale MoE/MTP replications, mapped onto this box's real constraints: FLOP-bound GPU, abundant RAM (params are cheap, FLOPs are the wall), byte vocab=256, 10-200M params, MPS fixed-shape rule. Every DeepSeek gain below was measured at 2B-671B; NONE of the mechanisms has published evidence at or below 1B active params, so the multipliers here are extrapolations and every experiment is a falsification, not a bet.

Baseline for all wave-3 A/Bs: `e5hybrid_10m_qat` (patched local attention + constant-state recurrent global mixer, 15.9M params at ~dense per-byte FLOPs, muon, fineweb_edu, 12000 steps, 79.5k tok/s, final val 0.9707).

### ranked levers

| lever | what it does | measured gain @ scale | expected mult here | evidence quality | MPS port risk |
|---|---|---|---|---|---|
| DeepSeekMoE (fine-grained + shared experts, aux-loss-free balance) | more total params at fixed active FLOPs: split FFN into m fine experts, top-mK route, isolate K_s always-on shared experts; V3 replaces the balance aux-loss with a per-expert routing bias nudged each step | 2401.06066: 2B MoE (0.3B active) hits Pile 1.808 = dense-upper-bound 1.806, beats GShard 2.9B at 1.5x less compute; 16B ~= LLaMA2-7B at 40% compute | 1.0-1.5x wall-clock, HIGH variance (could be <1) | strong >=0.3B active; ZERO at 10M; OLMoE/JetMoE confirm >=1.3B active | MEDIUM-HIGH: capacity-dense einsum routing is fixed-shape (MPS-safe) but pays a capacity-factor FLOP tax the box can't hide with sparse kernels; dead-expert risk at 10M |
| Byte-level MTP (extra head predicts byte t+2) | denser training signal per position + self-speculative decode; DeepSeek-V3 keeps causal chain with a sequential MTP module (shared embed + shared head + one block) | V3: +consistent benchmark deltas @15.7B/671B; second-token accept 85-90% -> 1.8x decode TPS. Gloeckle 2404.19737: n=4 code +12% HumanEval @13B, byte-level used n=8; 3x self-spec decode | 1.1-1.3x data-eff IF it fires at 10M + separate 1.5-1.8x decode axis | strong at scale; NEGATIVE/needs-registers below ~1B (2505.10518, babylm 2025) | LOW: one extra linear head, fixed shapes, MPS-trivial; NOT engine-exportable (v11 bans MTP, preflight rule 40b) |
| MLA (KV compression to latent) on the local blocks | compress K/V into a d_c latent, decoupled RoPE key; shrinks KV cache, claims >=MHA quality via low-rank bottleneck | V2 2405.04434: KV cache -93.3% vs 67B, train cost -42.5%, gen throughput 5.76x; d_c=512=4*d_h, cache ~= GQA-2.25-groups | ~1.0x here (redundant) | strong as an INFERENCE KV lever at many-heads/long-context; near-zero relevance at 10M/512-window | LOW-MEDIUM: up/down projections + decoupled RoPE code; no fp8 needed |
| FP8 / quantized training | Fprop/Dgrad/Wgrad GEMMs in E4M3 with 1x128 activation tiles, 128x128 weight blocks, fp32 accumulate every 128 | V3: <0.25% rel-loss error vs bf16 @~1T tokens | n/a | strong but Hopper-tensor-core-specific | DEAD: MPS has no fp8 dtype or kernels; bf16 is the floor. Kill, do not test |
| R1 GRPO / RL-from-base | R1-Zero does pure GRPO RL on the base; distillation to small models is SFT on 800k R1 traces, which BEATS direct RL on <7B students | 2501.12948: distill-SFT >> direct-RL at 1.5-32B | n/a pre-SFT | strong; matches the 2511.04902 kill (self-improve <1B collapses, ~0 base success) | POST-SFT ONLY: nothing usable at 10M pre-SFT; the usable form is teacher-trace SFT distillation, already the wave-2 decay-tail plan |

Genuine non-architectural levers in the reports: (a) data curation at 14.8T high-quality tokens and the two-stage WSD-with-batch-ramp schedule (batch 3072->15360, min_lr ~1/10 peak) - already covered by wave-2 curriculum/RHO-1 and the trainer's WSD shape; a batch-ramp is a mild free tweak. (b) DualPipe bidirectional pipeline overlap: irrelevant on a single GPU, zero transfer. (c) node-limited/device-limited routing: a multi-node comms trick, irrelevant single-box.

Void note: the tentative "E8 = MoR-lite" reserved in the wave-2 recursion table is dead on arrival - it was conditional on E7 showing signal, and E7 (`trunk=looped`) was falsified (see failures ledger 2026-07-05). E8 is reclaimed below.

### E8 - fine-grained MoE FFN on the hybrid trunk

- Exact change: replace the FFN inside the hybrid trunk's global blocks with a DeepSeekMoE FFN: 8 routed fine experts each at 1/4 the dense FFN width + 1 always-on shared expert at 1/4 width, top-2 routed. Routing is capacity-based DENSE dispatch (GShard/Switch 2006.16668 / 2101.03961 form): one-hot dispatch mask -> `[num_experts, capacity, hidden]` scatter via einsum, batched expert GEMM, einsum combine. capacity_factor=1.25, overflow dropped, underflow zero-padded. FIXED shapes throughout (MPS rule 24c). Aux-loss-free balance (V3 2412.19437): per-expert routing bias `b_i` added to affinity for top-k SELECTION only (not the combine weight), nudged +-gamma each step by over/under-load; tiny sequence-wise balance aux-loss alpha=0.01 (2401.06066) as backstop.
- Params/FLOPs vs hybrid baseline: active FFN width per token = 2*(1/4) routed + 1*(1/4) shared = 0.75x dense FFN, so per-token matmul FLOPs land NEAR the hybrid baseline (slightly under ex-routing); total FFN params ~3x (8+1 quarter-experts vs 1 dense). The honest cost the box cannot hide: the dense-capacity dispatch computes expert GEMMs over `num_experts * capacity` slots including padded/dropped ones, and the two dispatch/combine einsums are extra FLOPs - a real capacity-factor tax (~1.25x + einsum) that sparse-kernel GPUs avoid and MPS does not. So compare at EQUAL WALL-CLOCK, never equal active-FLOPs.
- Pre-registered falsifier: MoE arm fails to beat the hybrid baseline's final val (0.9707) at equal wall-clock, OR expert utilization collapses (any expert's token share <0.02 or >0.5 sustained past step 4000, i.e. dead/hot experts at 10M), OR realized throughput <70% of hybrid. Any of these kills fine-grained MoE at 10M-scale on this box; retry condition = repeat at >=80M active where DeepSeekMoE's own evidence begins to apply.
- Run config (POST /trainers/run): `output_name=e8moe_10m_qat`, `trunk=hybrid_moe`, `optimizer=muon`, `corpus=fineweb_edu`, `steps=12000`, canonical 10M shape, `model_type=language`. Single delta vs `e5hybrid_10m_qat`. Log per-expert token share to the dashboard each eval. Seed rule: any win under 5% needs 3 seeds before reporting (agent_roe).
- Build status (2026-07-04): MoE FFN implemented (`veritate_core/model_moe.py::MoEFFN`, component doc `developer_documentation/platform/model_moe.md`), wired as `trunk=hybrid_moe` (VeritatePatched `global_ffn="moe"`). Verified at the real 10M shape: unit combine/aux/share, aux-loss-free bias revives dead experts (time-averaged min share 0->0.073 over 800 steps), MPS fwd+bwd finite (no E4-class NaN), full dump/hook battery passes. No run launched yet.
- MEASURED FLOP TAX (decisive for sequencing): MoE/hybrid throughput = 0.715 (32.0k vs 44.8k tok/s, MPS bs32/seq512, AdamW+QAT) -- right on the pre-registered <70% kill line, clearing by only 1.5 points, and muon may push it under. MPS has no sparse-routing kernels, so the capacity dispatch pays for `num_experts*capacity` padded+dropped slots a datacenter GPU skips: MoE's "free params" premise is PARTLY DEFEATED by this box. Params 1.52x hybrid (cheap on 256GB) but the FLOP side is at the edge of acceptable. Consequence: MoE is NOT bet on the primary chat model. The 80M chat baseline uses plain hybrid+muon; `hybrid_moe` is tested as a controlled A/B at 80M afterward (where DeepSeekMoE's quality evidence begins) so the throughput tax can be weighed against a real quality delta. If it clears neither 70% throughput NOR a bpb win at 80M, MoE is dead on this hardware regardless of scale.

### E9 - byte-level multi-token prediction head

- Exact change: add ONE MTP module (DeepSeek-V3 depth D=1) predicting byte t+2. Module = shared byte embedding + one global trunk block + linear projection, tied to the shared byte-0 output head; sequential (reads the main path's residual, preserves the causal chain). Training loss = main next-byte CE + lambda * t+2 CE, lambda=0.3 for first 8000 steps then 0.1 (V3 schedule). At inference the head is either discarded (main model unchanged, rule 11a) or run as a 2-byte speculative drafter. Byte level is MTP's best case: bytes are low-information targets, so a t+2 head densifies the signal more than it does for subword tokens (Gloeckle 2404.19737 used n=8 heads specifically for their byte model).
- Params/FLOPs vs hybrid baseline: main decode path UNCHANGED (rule 11a: the head exposes `project_byte0`, consumers never branch). Training adds ~one-global-block forward/backward for the MTP module (a few percent wall-clock); params +one global block. Inference-time exportability: BLOCKED - the v11/v9 `.bin` format bans MTP heads (preflight rule 40b, exporter raises early); the MTP head is a training-signal + PyTorch-speculation device only until a v12 format ships.
- Pre-registered falsifier: main-task final val bpb not improved vs hybrid baseline (0.9707) at equal wall-clock (the densification claim), AND byte t+2 speculative acceptance <40% (the decode-lever claim) - if BOTH fail, byte-MTP is dead at 10M, matching the "MTP needs registers / helps only at scale" small-model evidence (2505.10518, babylm 2025). If only the val claim fails but acceptance is high, keep it as a decode-only lever. Retry for the densification claim: add register tokens (2505.10518) or retry at >=80M.
- Run config: `output_name=e9mtp_10m_qat`, trunk=hybrid + mtp flag, `optimizer=muon`, `corpus=fineweb_edu`, `steps=12000`, canonical 10M shape, `model_type=language`. Single delta vs `e5hybrid_10m_qat`. Log t+2 acceptance rate at each eval via a paired speculative pass. Must pass the dump suite at real run shape before launch (rule 24d); the MTP path is a known new-variant crash surface.

### E10 - MLA on the hybrid trunk's local attention (redundancy check)

- Exact change: replace the patched local blocks' MHA with MLA (V2 2405.04434): compress per-head K/V into a shared latent `c_KV` of dim d_c = 4*d_h, up-project for K/V, carry a small decoupled RoPE key of dim d_h/2 for position. Global recurrent path UNCHANGED (it already carries O(1) state, no KV). This directly tests the registered question: on the hybrid trunk the global long-range path is already constant-state recurrence, so MLA can only help the LOCAL blocks - which run on a bounded 512-byte window where the KV cache is already tiny.
- Params/FLOPs vs hybrid baseline: params ~neutral (latent down+up projections replace K/V projections, roughly offsetting at d_c=4*d_h). Training FLOPs ~neutral-to-slightly-up (extra projection matmuls). Decode KV memory on the local window drops from `2*n_h*d_h` to `~4.5*d_h` per position - but the local window is bounded at 512 bytes and the box has 256GB RAM, so this buys nothing at this scale.
- Pre-registered falsifier (framed to CONFIRM redundancy): MLA arm must beat the hybrid baseline's final val (0.9707) by >0.005 at equal wall-clock, OR show a measurable decode-memory/latency win at conversation-length eval (>=4096-byte context). If it clears neither, MLA is confirmed REDUNDANT on the hybrid trunk (the recurrent global path already owns the long-range/memory axis, and the local window is too short and the box too RAM-rich for KV compression to matter) and is killed for this architecture. Expected outcome: ~1.0x, redundant. Retry condition: an attention-heavy trunk (no recurrent global path) at long context, where KV cache actually dominates.
- Run config: `output_name=e10mla_10m_qat`, trunk=hybrid + mla-local flag, `optimizer=muon`, `corpus=fineweb_edu`, `steps=12000`, canonical 10M shape, `model_type=language`. Single delta vs `e5hybrid_10m_qat`. Decoupled RoPE + out-of-place masks only (MPS rule 24c). Seed rule applies to any sub-5% delta.

## experiments

- E1 Muon vs AdamW: canonical ~20M, same corpus/schedule, measure bytes to reach the AdamW arm final val bpb. Falsifier: under 1.15x savings.
- E2 boundary-patched trunk vs dense at equal FLOPs and wall-clock, both arms on the E1 winner. Fixed patch-slot count (MPS fixed-shape rule). Falsifier: no bpb win at equal wall-clock, or realized throughput under 70 percent of dense.
- E3 constant-state trunk vs dense, params-matched, 3 seeds. Falsifier: mean val bpb worse by more than 0.03 at equal wall-clock.
- E4 surprise-gated memory: inject invented facts once, quiz after 8k distractor bytes, memory on/off/matched-dense. Falsifier: recall lift under +10 points absolute, or plain-text bpb regression over 0.05.

Composition target: E1 x E2 x E3 stack multiplicatively if each clears its falsifier; the stack is the answer to the 2026-06-27 compute-wall failure (retry condition: a 10x-or-better lever).

## sources

BLT 2412.09871, Fast-BLT 2605.08044, SpaceByte 2404.14408, H-Net 2507.07955, MoR 2507.10524, MoE scaling 2402.07871, BitNet 2402.17764 / 2407.09527 / 2504.12285, Muon 2502.16982 / 2505.02222, distillation scaling 2502.08606, MiniPLM 2410.17215, MambaByte 2401.13660, RWKV-7 2503.14456, Gated DeltaNet 2412.06464, Mamba-2 2405.21060, Titans 2501.00663, TTT 2407.04620, ATLAS 2505.23735, LaCT 2505.23884, memory layers 2412.09764, sparse memory finetuning 2510.15103, Hymba 2411.13676, forward-forward 2301.01452, Mono-Forward 2501.09238.

Wave-3 (DeepSeek-class): DeepSeek-V2 2405.04434, DeepSeek-V3 2412.19437, DeepSeekMoE 2401.06066, DeepSeek-R1 2501.12948, MTP (Gloeckle) 2404.19737, MTP-needs-registers 2505.10518, OLMoE 2409.02060, JetMoE 2404.07413, GShard 2006.16668, Switch Transformer 2101.03961, self-improve-collapse-below-1B 2511.04902.
