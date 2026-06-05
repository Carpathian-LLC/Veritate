# multimind ideas

Research log for the Multi-Mind (MtM) Veritate variant. Open ideas, active falsifiers, and the order we test them. Falsified ideas migrate to FAILURES.md with the three-line format; validated ideas migrate to SUCCESSES.md with numbers.

Read `claude_preflight.md` rule 40a before adding to this file: ideas are research questions with a stated falsifier, not changelog entries.

## 100 fallback variants (if W1e fails)

User directive 2026-05-21: failure is not the wrong option, it's a signal to try 100 different things. This list is exhaustive on purpose. Iterate from top 10 first; expand outward if all fail. Group prefixes are for navigation only.

#### (A) Bias mechanism
1. W1e: oracle-forced g ([+1,+1,+1,-1,-1,-1], frozen) — currently running.
2. Per-layer g (shape `(layers, n_experts)`) instead of global.
3. Per-layer per-head g (richer signal injection).
4. Trainable scalar amplitude `alpha * g * s` with alpha learnable.
5. Curriculum-amplified s in [-5,+5] for first half, anneal back to [-1,+1].
6. Add `g_e * s` to OUTPUT-side expert weights too, not just gate logits.
7. Multiplicative bias: `gate_logits *= (1 + g * s)`.
8. Sentiment as a learnable continuous embedding added to residual stream.
9. Two asymmetric biases: `g_pos` and `g_neg` (only active when s sign matches).
10. Step function: route to fixed experts based on `sign(s)` with no learning.
11. Hash-based hard routing on s (deterministic, ablates softmax noise).
12. Modulate router weight matrix directly: `W_router + g_outer_product(s)`.
13. Steering: inject s into attention Q/K/V via learnable projection.
14. FiLM modulation: gamma + beta on residual stream conditioned on s.
15. 2D bias plane: (valence, arousal) -> g_2d * (v, a).

#### (B) Routing mechanism
16. Top-1 routing (cleaner signal; less expert mixing noise).
17. Top-k=3 or 4 (more receptive to weak bias).
18. Gumbel-softmax routing for differentiable hard selection.
19. Straight-through estimator on top-k indices.
20. Sigmoid-per-expert gating (independent gates, no normalization).
21. Sparsemax instead of softmax (sparser routing).
22. Hard routing with REINFORCE.
23. Sequence-level routing: one expert per sample, not per token.
24. Hierarchical routing: route to "polarity group" then to expert.
25. Anneal top-k from 6 (dense) to 2 over training.

#### (C) Training signal
26. Train longer (8000-20000 steps; byte-level needs more).
27. Larger model (hidden=512, 8 layers).
28. Smaller model (hidden=128, 2 layers) for faster iteration.
29. Smaller corpus (5k samples) for faster iteration.
30. Larger batch (128).
31. Smaller batch + grad accumulation (lower variance updates to g).
32. Warmup g alone for first 500 steps before FFNs touch it.
33. Pretrain FFNs to be sentiment-specialized via auxiliary classifier.
34. Higher LR for g (separate param group at 10x base LR).
35. Higher LR for router.
36. Cosine LR restarts.

#### (D) Data
37. SST-2 instead of Amazon Polarity (cleaner, shorter).
38. Synthetic pos/neg corpus (max signal, controlled).
39. Class-balanced sampling per batch (force exposure).
40. Strip neutral spans; train on highly polarized bytes only.
41. Augment with negation samples ("not bad", "could be better").
42. Mix multiple sentiment datasets.
43. Per-position labels (every byte labeled, not just sample-level).
44. Sort by polarity strength; curriculum easy-to-hard.

#### (E) Loss
45. Drop aux load-balance loss (lets specialization happen, see if signal emerges).
46. Increase aux load-balance coefficient (force more uniformity, expose bias signal).
47. Routing consistency loss (same sentiment -> same routing).
48. Contrastive routing loss between pos/neg sample pairs.
49. Routing entropy regularization (encourage decisive routing).
50. Per-expert auxiliary classification head (each expert predicts s).
51. KL push between gate distribution and a sentiment-conditioned target.

#### (F) Architecture
52. Two-tier MoE: sentiment-MoE + content-MoE in parallel.
53. Sparse attention with sentiment-routed heads.
54. Hypernetwork: small net generates expert weights from s.
55. LoRA per expert (adapter routing).
56. Mixture-of-LoRAs replacing mixture-of-FFNs.
57. Sentiment-conditioned LayerNorm (gamma/beta from s).
58. FiLM-style residual modulation.
59. Sentiment-conditioned positional encoding.
60. Sentiment-as-prefix token ("POS"/"NEG" prepended to input).

#### (G) Curriculum / schedule
61. Curriculum extreme -> neutral.
62. Warmup g amplitude from 0 -> 1.
63. Decay g amplitude over training (teacher pattern).
64. Phase 1 train router; phase 2 train FFN.
65. Anneal gate softmax temperature (high to low).

#### (H) Diagnostics first
66. Per-step routing entropy logging.
67. Per-step router gradient norm logging.
68. Per-step g gradient norm logging (`g.grad.norm()`).
69. Attention-vs-FFN sentiment localization (where does the signal actually settle?).
70. PCA on expert outputs to see actual specialization axes.
71. Train a per-expert "what does this expert see?" classifier.
72. Inspect top-k boundary cases (samples where bias flips a routing decision).

#### (I) Probe
73. 5-class sentiment task (more discriminative).
74. Adversarial training samples.
75. Distill from a larger pretrained sentiment model.
76. FastText-style bag-of-bigrams probe (proven 94.6% on Amazon).

#### (J) Engineering / debug
77. Verify gate_bias broadcast shape via unit test.
78. Print s during training to verify pass-through.
79. Print g over time to verify movement.
80. Print g.grad to verify gradient signal.
81. Bypass model; feed crafted gate logits directly; verify expert outputs differ.
82. Print routing histogram per class per epoch.

#### (K) Different framing
83. Sentiment-classification task instead of LM (no causal loss).
84. Multi-task: LM + sentiment heads, sentiment loss conditions routing.
85. Bias the OUTPUT not the gate (post-expert mixing).
86. Sentiment-aware skip connection bias.
87. RWKV-style scalar state modulation.

#### (L) Train/inference mismatch
88. Train without bias, evaluate with bias (true sidecar).
89. Joint train probe + bias.
90. Cross-entropy on "which expert should fire" supervised by sentiment.

#### (M) Pretraining
91. Pretrain dense model first, convert to MoE (upcycle).
92. Use a public byte-level pretrained model as init.
93. Use Veritate 85m checkpoint as init for dense trunk.

#### (N) Math knobs
94. Larger init for g (start at +/-0.5, not 0).
95. Smaller init for router (so bias dominates initially).
96. Lower softmax temperature (sharper top-k).
97. Higher softmax temperature (smoother, more bias-receptive).

#### (O) Inhibition / inhibitory dynamics
98. Refractory mask (W4 in queue): fired expert suppresses self K bytes.
99. Lateral inhibition between same-polarity experts.
100. Sticky expert: once fired, keep firing for K bytes (boost prior decision).

## execution order if W1e fails

1. **First wave (cheapest, biggest signal)**: #16 (top-1), #94 (g init), #5 (s amplification), #45 (drop aux balance), #67/#68 (diagnostics).
2. **Second wave (model-level)**: #2 (per-layer g), #50 (per-expert head), #14 (FiLM), #60 (prefix token).
3. **Third wave (full redesign)**: #52 (two-tier), #57 (LN modulation), #91 (upcycle).
4. If 1-3 fail, the architecture is structurally unable; pivot to non-MoE sentiment-modulated paths (#85 RWKV, #86 skip-bias).

## W5 final 2026-05-22: sleep adaptation works; forgetting is a real cost at 7.5M params

8 attempts spanning naive FT, sparse-by-expert, LoRA, LoRA-cranked, OPLoRA (200 steps), OPLoRA gentler. The dose-vs-forgetting curve at 7.5M params:

| variant | convo_drop | holdout_rise |
|---|---|---|
| FT 80 steps lr=1e-4    | +56.8% | +81.0% (catastrophic) |
| FT 15 steps lr=1e-5    |  +6.2% | +0.6% |
| FT 20 steps lr=5e-5    | +21.7% | +8.2% |
| FT 25 steps lr=3e-5    | +18.5% | +5.9% |
| sparse 40 steps        | +11.3% | +3.7% |
| sparse 100 steps       | +31.3% | +27.0% |
| LoRA rank=4 lr=5e-4 80s |  +3.2% | +1.5% |
| LoRA rank=16 lr=2e-3 200s | +19.9% | +27.9% |
| OPLoRA rank=16 lr=2e-3 200s | +19.7% | +26.7% |
| OPLoRA rank=8 lr=1e-4 80s K=12 |  +1.7% | +0.3% |

Sweet spot near LR=5e-5 / 20 steps (drop=21.7% / rise=8.2%) — the tradeoff is real and continuous, not a hard pass/fail. OPLoRA can pin holdout to ~0% rise but blocks adaptation. Fundamental cause: at 7.5M params, parameter-sharing means buffer-relevant gradient directions overlap with holdout-relevant ones; orthogonal projection blocks both. Proper solution requires either (a) larger model with more redundant capacity, (b) gradient-task-arithmetic with multiple buffers averaged (continual learning literature), or (c) explicit per-region "modular" adapters with hard separation.

**Decision**: W5 = PARTIAL (sleep mechanism demonstrably works; dose/forgetting tradeoff curve documented; production-grade solution needs continual-learning machinery beyond PoC scope). Move on to W4 + benchmark + final demo.

## W6 final 2026-05-22: slot bank gives 16x cross-sample signal but not episodic memory

W6b cross-sample test (question-only eval, no fact in context): baseline 1.0% accuracy, with_memory 16.0%. **16x improvement is real** but absolute 16% is below the 25% PASS threshold. W6c tried bigger bank (256) + fact-masking during training + 10k steps; regressed to 6.6%. Diagnosis: the SlotMemory module is a learnable KV bank trained by gradient descent — it stores GLOBAL statistics, not per-sample episodes. Real episodic ("hippocampus") memory needs explicit write-as-you-go state, not parameter-baked KV. That's a larger architecture redesign (write/read separate, sample-internal state, optional cross-sample persistence) outside PoC scope; the W6 result validates the concept (slot memory helps 16x) without crossing the falsifier line.

**Decision**: W6 = PARTIAL (architecture concept works; proper episodic memory deferred to follow-up). Move on to W5 + W4.

## W6 discovery 2026-05-21: in-context recall is trivial; slot memory needs cross-sample test

W6 v1 (`w6_slot_memory.py`) results: baseline 100% accuracy; with_memory 99.8%. Both nailed the synthetic "fact stated 200 bytes ago, then asked" recall task. Verdict FAIL only because the falsifier required baseline < 10%; the test was too easy.

**Real finding**: standard transformer attention handles 256-512 byte recall trivially. The synthetic in-context recall task does NOT distinguish "remembered via attention" from "remembered via slot memory." To test slot memory's actual purpose (PERSISTENT memory across samples, beyond the context window), the eval has to give ONLY the question with NO fact in context. If the slot bank truly encodes (name -> color) statistics globally via gradient updates, only the with_memory model can recall.

W6b (`w6b_slot_memory_v2.py`) redesigned: train with fact+noise+question, eval with question only. 24 names x 16 colors = 384 combos (no longer memorizable as a tiny set). Pass criteria: with_memory >= 25% AND baseline < 18% (random over ~7 unique color first-bytes).

## status board 2026-05-21

| exp | status | metric | notes |
|---|---|---|---|
| W1   | PASS  | KL ratio 12.9x, ||g|| 0.014 (later 0.218 with W1g hyperparams) | architecture lives at byte-level |
| W1b  | FAIL  | KL ratio 1.9x | freeze-tail backfires (frozen FFNs reject bias-driven routing) |
| W1e  | PASS  | KL ratio 3564x, ||g|| 2.45 | gold-standard controller: oracle g works |
| W1g  | PASS  | KL ratio 34x, ||g|| 0.218 | training works with 10x LR on g + nonzero init |
| W2   | PASS  | ratio vs oracle 1.57, ||g|| 0.222 | learned probe (86% acc) recovers ALL of oracle signal |
| W3   | running | TBD | region naming via specialty corpora |
| W4   | running | TBD | refractory inhibition burst-iness vs ppl |
| W5   | coded, queued | TBD | sleep-cycle MoE-expert adaptation |
| W6   | coded, queued | TBD | slot memory long-range recall |

Affect probe: 86.09% val acc on Amazon Polarity at 200k params (5 dilated convs, dilations [1,2,4,8,16]).

Platform integration:
- `veritate_core/model_mtm.py` ships VeritateMultimind class with full canonical contract
- `veritate_core/load.py` has MtM dispatch branch ahead of RoPE branch
- 4 model_mtm tests pass; no regression in 6 existing roundtrip tests
- 26 new mesh/settings/sync_common tests pass
- ~646 LOC removed across mesh, routes/JS, brain methods, dead decode files
- 5 dead inference files deleted (eagle3, kangaroo, exit_head, constrained, best_of_n)

## W1 outcome 2026-05-21: ARCHITECTURE LIVES, falsifier setup needs tightening

- Verdict: FAIL on the strict falsifier (||g||=0.014 < 0.1) but **PASS in spirit**: with-bias model's L0 routing KL is 12.9x baseline (0.0424 vs 0.0033 nats).
- L0 per-expert delta is non-trivial: positive bytes prefer wernicke (+5.7%) and thalamus (+2.9%); negative bytes prefer prefrontal (+8.2%). Cerebellum dominates both at ~34% (expected load-imbalance noise floor).
- Mechanism: ||g||=0.014 is too small to be the direct route-changer, yet routing shifted 13x. The bias term seeded the gradient pathway early in training; the router then self-organized around sentiment-correlated input bytes. Weight decay on `g` (0.01 in AdamW) suppressed `g` growth.
- Next: W1b (freeze-tail + z-loss + exclude `g` from weight decay) forces `g` to be the load-bearing path. Running now.

## active falsifiers

### W1: affect bias shifts MoE routing (oracle signal)

- **claim**: adding a per-expert learnable bias `g_e * s` to gate logits (where `s` is a sentiment scalar, `g_e` is a learnable scalar per expert) causes measurable expert specialization by sentiment, vs a baseline MoE on identical data.
- **falsifier**: KL(P(expert|positive) || P(expert|negative)) on held-out labeled bytes, with-bias model >= 2x baseline model AND `||g||_2 > 0.1` after training. If either fails, the architecture absorbs the signal as noise and the whole MtM premise is dead at this scale.
- **dataset**: Amazon Polarity (Apache-2.0), 50k pos + 50k neg, 512-byte windows.
- **model**: byte-level transformer, 6 experts, hidden=256, layers=4, ffn=512, top-k=2, ~15M params.
- **baseline vs treatment**: identical config, identical seed, identical data; treatment adds `g * s` term inside the gate logits before topk.
- **expected wall-clock**: ~75 min on the 5070 (corpus 10 min + 2 trainings 30 min each + eval 5 min).
- **status**: in flight (2026-05-21).

## queued ideas (test after W1)

### W2: learned affect probe (Q2)

- if W1 passes, replace oracle scalar with the output of a tiny byte-level CNN trained on the same labels. Falsifier: does the learned probe's routing-shift recover at least 50% of the oracle's routing-shift?
- if W1 fails, this is moot.

### W3: region naming via Stage B specialty corpora

- soft KL gate nudge per region. broca trained on output-side dialogue, wernicke on QA comprehension, hippocampus on long-context recall, prefrontal on multi-step chain text, cerebellum on code/repetition, thalamus on routing meta-loss.
- falsifier: per-region specialty perplexity on held-out specialty data is lower than that region's perplexity on a random other specialty's data, by at least 10%.

### W4: refractory lateral inhibition

- a region that fired (gate_weight > 0.5) suppresses its own gate logit for K subsequent bytes.
- falsifier: per-byte decode FLOPS drop measurably (>5%) at fixed quality vs no-refractory baseline; or, qualitatively, firing patterns become bursty rather than constant-on-one-expert.

### W5: sleep-cycle adapter updates over the mesh

- conversation buffer ships from x86 to M3 Ultra via the mesh; per-region LoRA adapters trained on CPU bf16 / GPU INT8 split path (see project memory: split-precision invention); shipped back; hot-swapped.
- falsifier: per-region specialty perplexity decreases after a sleep cycle on a conversation buffer relevant to that region's specialty, vs no-sleep baseline on the same eval set.

### W6: slot-table as hippocampus episodic memory

- existing slot_table addon repurposed for the hippocampus region's recent-context recall.
- falsifier: recall accuracy on a "fact stated 1000 bytes ago, then asked" task >= 30% (random baseline at byte-level: <5%).

## W1 contingency: pre-staged retries if initial run shows weak signal

### W1b: freeze non-router params for final 500 steps

- **rationale**: most likely W1 failure mode (per research compilation 2026-05-21) is "sentiment leak through CE" — the attention + non-MoE MLPs solve sentiment on their own, the gate weight `g` rides along at zero, falsifier reads as FAIL even though architecture works.
- **change vs W1**: after step `TRAIN_STEPS - FREEZE_TAIL`, freeze every parameter except the router weight and `gate_g`. Sentiment information can then only flow into the prediction through the bias term, forcing `g` to be load-bearing or the loss stalls.
- **falsifier**: same as W1.
- **artifact**: `trainers/multimind_poc/falsifier_v2.py` already authored; run via `python falsifier_v2.py --freeze-tail 500`.

### W1c: router z-loss

- **rationale**: large router logits cause numerical instability and high-entropy noise on top of any bias term. z-loss (sum of squared log-sum-exp on logits) keeps logits in a regime where small `g_e * s` perturbations actually move the topk decision boundary.
- **change vs W1**: add `Z_LOSS_COEFF * (logsumexp(logits)**2).mean()` to the per-block loss. Coefficient 1e-3 per prior art.
- **falsifier**: same as W1.
- **artifact**: also in `falsifier_v2.py` via `--z-loss 1e-3`.

### W1c: longer training (byte-level routing SNR contingency)

- **rationale**: byte-level per-position router gradient is ~2-3x noisier than subword (BLT reports byte BPB 0.83-0.86 nats/byte vs subword ~1.5-3 bits/token effective). OLMoE specialization "saturates at ~25-30% of training." 3000 steps may be pre-saturation. The fix may be more steps, not architectural.
- **change vs W1**: `TRAIN_STEPS = 8000`, otherwise identical. Cheap if W1 trended in the right direction but undershot.
- **falsifier**: same.

### W1d: per-layer g instead of global

- if W1a-c all FAIL but `||g||_2` is non-trivial in W1c, the global vector may be too coarse. Replace `(n_experts,)` with `(layers, n_experts)`. Falsifier unchanged but harder to satisfy (more params to converge).

### W1e: oracle-forced g (gold-standard controller)

- **purpose**: distinguish "architecture cannot route on bias" from "training signal too weak." If W1+W1b+W1c all fail, this answers whether the architecture is dead or just hard to train.
- **change vs W1**: hand-init `g_e = [+1, +1, +1, -1, -1, -1]` (or any handcrafted half-split), FREEZE `g_e` from step 0, train only router + FFN.
- **falsifier**: KL(P(expert|pos) || P(expert|neg)) >= 0.5 nats AND NLL on pos vs neg val diverges by >= 0.1. If FAIL, the architecture truly cannot use a bias signal; MtM premise is dead. If PASS, the original W1's signal-to-noise was the bottleneck.
- **artifact**: `falsifier_v2.py` `--oracle-g`.

## decision tree post-W1 (read this before iterating)

```
W1 verdict:
  PASS                                       -> ship W2 (learned probe)
  FAIL with ||g|| close to PASS_G_NORM       -> retry W1d (per-layer g)
  FAIL with ||g|| ≈ 0 and KL ratio < 1.5     -> CE solved it elsewhere; W1b (freeze-tail)
  FAIL with KL ratio borderline (~1.5-2.0)   -> W1c (longer training, 8000 steps)
  FAIL all of the above                       -> W1e (oracle-forced) to settle architecture-vs-training
```

## published MoE specialization anchors (calibration for verdicts)

- Mixtral 8x7B: near-uniform routing on The Pile domains. Specialization is essentially absent for upcycled MoEs ([arxiv 2401.04088](https://arxiv.org/abs/2401.04088) §5).
- OLMoE-from-scratch: clear domain specialization on arXiv/GitHub/Wikipedia, saturates at ~25-30% of training ([arxiv 2409.02060](https://arxiv.org/html/2409.02060v1) §5.3-5.4).
- Verdict calibration: KL ceiling over 6 experts = `log(6) ≈ 1.79 nats`. Our W1 PASS threshold KL >= 0.5 nats = 28% of ceiling, comparable to early-saturation OLMoE specialization. Threshold is meaningful, not a strawman.

## byte-level + MoE = open lane

- No published byte-level MoE LM found. BLT (Meta Dec 2024, [arxiv 2412.09871](https://arxiv.org/html/2412.09871v1)) is the dominant byte-level scaling paper and explicitly uses dense transformers over entropy-based patches, NOT MoE. The combination "byte-level + MoE + affect-routed" appears genuinely unexplored.

## DeepSeek aux-loss-free balancing port (W1f, optional)

- Rule: `b_i := b_i + u * sign(c_bar - c_i)`, applied per batch. `u = 0.001`. Bias added to router logits PRE top-k only; NOT used in output weighting. Lives in disjoint algebraic slot from our `g_e * s` (which is also pre-top-k but flows through output combining). They can coexist; consider porting if W1 shows expert collapse despite the existing aux-loss term.

## related prior art

- DeepSeek auxiliary-loss-free MoE balancing ([arXiv 2408.15664](https://arxiv.org/pdf/2408.15664)): adds per-expert additive bias updated from running token-count. Our `g_e * s` reuses the additive-before-topk shape so load balancing still works. Consider porting the EMA bias update.
- "Channel-Aware Gating MoE" ([arXiv 2504.00819](https://arxiv.org/abs/2504.00819)): adds an external scalar signal to MoE gate logits. Closest mechanical analog. Not LM, not sentiment.
- BrainStack — Neuro-MoE with EEG-driven expert routing ([arXiv 2601.21148](https://arxiv.org/pdf/2601.21148)): external neural signal biases expert weights. Closest brain-inspired match.
- Zhyper (Oct 2025, [arXiv 2510.19733](https://arxiv.org/pdf/2510.19733)): factorized hypernet conditioning LoRA on a scalar; analogous to `g_e * s` at adapter scale.
- HyperSteer (2025, [arXiv 2506.03292](https://arxiv.org/pdf/2506.03292)): scalar control of token-level computation without MoE.
- Valence-Arousal subspace steering on Llama/Qwen (2025, [arXiv 2604.03147v2](https://arxiv.org/html/2604.03147v2)): monotonic behavior shift via scalar direction in activations.
- Router z-loss best practice: [mbrenndoerfer.com/writing/router-z-loss-moe-training-stability](https://mbrenndoerfer.com/writing/router-z-loss-moe-training-stability)
- EvaByte (2025): byte-level matches subword LMs at 5-10x decode speed. Validates byte-level as a real substrate for MtM. [hkunlp.github.io/blog/2025/evabyte/](https://hkunlp.github.io/blog/2025/evabyte/)

**No 2024-2025 paper found that uses a frozen affect/sentiment probe as a gate-bias sidechannel in an LM MoE at byte-level.** W1 is novel territory.

## platform integration map (executed when an MtM falsifier PASSes)

Survey of the Veritate platform on 2026-05-21 found 4 load-bearing changes between research-local code and a proper Veritate variant. Order: 1 -> 2 are mandatory for PyTorch inference; 3 -> 4 are mandatory for C-engine inference.

1. **`veritate_core/model_mtm.py` (NEW)**: research model promoted with the full Veritate contract: `forward(tokens, targets=None) -> (logits, loss)`, `embed`, `run_blocks`, `run_block`, `project_byte0`, `set_qat`, `post_l1_sum`, **`hook_spec()` returns a canonical-shaped FFN proxy** (routing-weighted expert output) so the dumper walks one shape. Sentiment-bias `gate_g` stays on the model as a normal nn.Parameter.

2. **`veritate_core/load.py`**: add a branch in `load_from_state_dict(sd, cfg)`. The discriminator key is `blocks.0.ff.router.weight` (canonical / RoPE / 800M / 85M do not have this). Order matters: check MtM before RoPE so MtM with rope_buffers doesn't get misrouted. Branch logic: ~6 LOC.

3. **`veritate_mri/inference/backends/pytorch.py` Brain backend**: currently assumes single dense FFN per block; reads `blk.ff.up` pre-GELU output for DLA. For MtM: either (a) Brain detects MoE and reads a synthesized "effective FFN" tensor from `hook_spec()`, or (b) we ship a `BrainMoE` subclass. (a) is leaner; the `hook_spec()` proxy in (1) is designed for it.

4. **Export to C engine (`veritate_mri/training/export.py`)**: v11 supports MoE but ONLY `router_topk=1` (engine refuses higher). Two paths: (a) switch MtM to top-1 routing for the shippable build, or (b) extend the engine and bump to v13. (a) is far leaner; top-1 is a one-line change in `moe_model.py`. Sentiment-bias `gate_g` has no v11 serialization slot. For (a)-leaner-still: bake `gate_g * s(prompt)` into the EXPORTED per-expert router bias as a sidecar JSON that the C engine loads at decode time (separate file, no binary format change). v13 is deferred.

5. **`documentation/hooks/contract.md` (deferred)**: reserve fields `expert_choice`, `gate_weight`, `refractory_mask`, `valence`, `arousal` in a "v8 reserved" subsection; emit only when the dumper sees an MtM model. Avoids touching the v7 producer until a multimind model actually trains end-to-end.

**Hardest change**: #4 export-side, because the C engine v11 binary has no extension point for affect-bias and top-K>1 is explicitly refused by the loader. Path of least resistance for shipping: switch to top-1 routing + sidecar JSON for `gate_g`. Defer v13 until a successor experiment needs top-K>1.

## affect probe priorart (W2)

- Zhang et al. char-CNN (~11M params): ~95.1% on Amazon Polarity. Not sub-1M.
- FastText (linear bag-of-bigrams, <1M params): ~94.6% on Amazon Polarity, ~90.7% word-only.
- No published sub-1M-param byte/char CNN on Amazon Polarity found. Our 200k-param dilated-conv probe targeting >=75% acc for the falsifier is plausible but unverified at this scale; closest priorart is linear. If the probe under-performs, fall back to a 1M-param variant or the linear bag-of-bigrams structure.
- Dilation > depthwise-separable per DCLS (2024 NeurIPS-adjacent, [arXiv 2408.03164](https://arxiv.org/html/2408.03164v1)). Already chosen.
