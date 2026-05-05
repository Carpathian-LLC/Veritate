# Research notes

Cited literature. Updated 2026-04-28 with architectural alternatives to the transformer for
80M / INT8 / CPU-decode / sub-ms targets. Existing quantization, analog, and SIMD notes
preserved at the bottom.

# ------------------------------------------------------------------------------------
# Key takeaways for Veritate
# ------------------------------------------------------------------------------------

1. **Mamba-2 / Mamba-3 SSM is the strongest first port.** O(1) per-token state, INT8
   already shipped (BitMamba-2 hit 112.9 tok/s on Xeon AVX-512 at 255M, ~9 ms/tok),
   matmul-dominated kernels. Direct fit for our VNNI path.
2. **RWKV-7 "Goose" is the second port.** Pure recurrence, vector-valued gating,
   constant memory and time per token, mature CPU runtime (rwkv.cpp, llama.cpp).
   Beats Mamba-class models at 3B with fewer training tokens; small-scale evidence is
   weaker but trendlines hold.
3. **Hybrid: Griffin / Hawk pattern (recurrent + 1k-window local attention).** Matches
   Llama-2 with 6× less training data; the local attention restores recall that pure
   SSMs lose at small scale. Highest expected quality at 80M.
4. **BitNet b1.58 ternary + T-MAC LUT kernels** is the orthogonal compression axis,
   not an architecture replacement. Combines with any of the above.
5. **MoE / MoD / Hyena / SpikeGPT / dLLM / HRR** are interesting but do not fit our
   constraints today (irregular memory, FFT cost, no INT8 story, tiny scales).

# ------------------------------------------------------------------------------------
# Sub-quadratic recurrent / SSM family
# ------------------------------------------------------------------------------------

## Mamba / Mamba-2 / Mamba-3

### Math
Selective state-space recurrence: `h_t = A_t h_{t-1} + B_t x_t; y_t = C_t h_t`.
A, B, C are input-dependent (selective). Mamba-2 restricts A to a scalar-times-identity,
yielding the "state-space duality" form, which collapses the recurrence to a sequence of
matmuls (i.e. our existing kernel). Mamba-3 adds (a) exponential-trapezoidal
discretization (drops the short causal conv), (b) complex-valued rotation in the state
update (data-dependent rotary), (c) MIMO outer-product → matmul state update, raising
arithmetic intensity at decode without growing state.

### Inference cost vs Veritate transformer
- Per-token decode: O(d · N) for SSM update + O(d²) for projections, where N = state
  dim (16-256). At our d=768 this is ~6-25× cheaper per layer than full attention
  prefill, but our transformer is already cached → for decode-only the comparison is
  closer to parity per layer. The win is *no KV cache growth* and a smaller working set.
- BitMamba-2 (1.58-bit) on Xeon AVX-512: 112.9 tok/s @ 255M (~8.8 ms/tok), 46.8 tok/s
  @ 1B (~21 ms/tok). At 80M scaled linearly: ~3 ms/tok ternary; INT8 with our kernel
  density would be ~1.5-3 ms/tok bound, i.e. same order as our 0.836 ms transformer.

### Quality
- Mamba-1 matches Transformer++ from 125M to 1.3B perplexity-for-perplexity.
- Mamba-3 1.5B: +0.6 pp avg downstream over Gated DeltaNet; MIMO adds another +1.2 pp.
  Same perplexity as Mamba-2 at half the state size.
- INT8 W8A8 quantized Mamba-2 (SSDi8): "near-lossless" vs FP16, 1.4× speedup.

### CPU friendliness
- Selective scan = sequence of scalar-times-identity recurrences → fully vectorizable.
- INT8 state quantization shipped: Decoupled Scale Quantization (DSQ).
- Hand-written AVX-512 kernels exist (BitMamba-2 ref impl).

### Open source
- Reference: https://github.com/state-spaces/mamba
- Mamba-3 paper: https://arxiv.org/abs/2603.15569 (ICLR 2026)
- Mamba-2 paper: https://arxiv.org/abs/2405.21060
- Mamba-1 paper: https://arxiv.org/abs/2312.00752
- BitMamba-2 C engine (AVX-512 + NEON): https://github.com/jserv/bitmamba.c
- Cross-platform benchmark: https://engrxiv.org/preprint/view/6686

### Validation cost on Veritate harness
~2-3 weeks. Replace attention block with selective-scan kernel; reuse FFN. Need (a) new
INT8 selective-scan kernel (vpdpbusd-friendly), (b) port weights from a pretrained
Mamba-2 130M, (c) run scalar oracle, (d) bench.

## RWKV-7 "Goose"

### Math
Generalized delta-rule with vector-valued gating:
`s_t = s_{t-1} (diag(w_t) - kᵀ_t a_t kᵀ_t) + vᵀ_t kᵀ_t`, output `o_t = s_t r_t`.
Time-mixing replaces softmax attention; channel-mixing replaces FFN. State is a
matrix-valued running buffer. Capable of recognizing all regular languages — strictly
beyond TC⁰ transformer expressivity under standard complexity conjectures.

### Inference cost
- O(d²) per token, constant memory. No KV cache.
- 80M model: ~5-10 ms/tok on consumer CPU with rwkv.cpp at INT4. Veritate-grade
  hand-tuned VNNI: estimate 1-2 ms/tok at INT8.

### Quality
- 2.9B Goose hits 3B SOTA on multilingual at 3.1T training tokens (vs Llama-3 8B's 15T).
- 0.19B / 0.4B / 1.5B / 2.9B model suite released.
- Outperforms Mamba-2 at matched compute on most downstream tasks.

### CPU friendliness
- Pure recurrence; no scan. Inner loop is vector × matrix → vector. Maps cleanly to
  vpdpbusd. INT4/INT8 quantization in rwkv.cpp ships today.
- Watch-out: weight outliers can break naive Q4_0 (known issue in rwkv.cpp).

### Open source
- Paper: https://arxiv.org/abs/2503.14456
- Model weights: https://huggingface.co/BlinkDL
- C inference: https://github.com/RWKV/rwkv.cpp (INT4/5/8 + FP16)
- llama.cpp also supports RWKV-6/7 natively now.

### Validation cost
~2 weeks. RWKV is the closest architectural fit: vector-state recurrence is exactly
what our matmul kernel does. Port a 0.19B Goose checkpoint, distill to 80M.

## Griffin / Hawk (DeepMind, gated linear recurrence + local attention)

### Math
RG-LRU (Real-Gated Linear Recurrent Unit): `h_t = a_t ⊙ h_{t-1} + (1 - a_t) ⊙ x_t`,
where `a_t = σ(W_a x_t)`. Hawk = pure RG-LRU stack. Griffin interleaves RG-LRU blocks
with sliding-window local attention (window ~1024).

### Inference cost
- RG-LRU per token: O(d) — element-wise. Cheaper than SSM scan, no state matrix.
- Local attention: O(d · w) where w = window. Bounded.
- Per-block decode: ~2-3× cheaper than full attention at our seq=256.

### Quality
- Hawk > Mamba on downstream at matched scale.
- Griffin matches Llama-2 with 6× fewer tokens.
- Length-extrapolates well beyond training context.

### CPU friendliness
- RG-LRU is element-wise sigmoid + multiply-add → trivial AVX-512 BF16/INT8.
- Local attention reuses our existing transformer kernels with a window mask.

### Open source
- Paper: https://arxiv.org/abs/2402.19427
- PyTorch impl: https://github.com/fattorib/hawk-pytorch
- DeepMind has not released weights publicly; reproduction efforts exist.

### Validation cost
~3 weeks. Two new kernels (RG-LRU element-wise, masked local-window attention).
Higher quality bet than pure recurrent but more code.

## xLSTM / mLSTM (Beck et al.)

### Math
mLSTM: matrix-valued cell state `C_t = f_t C_{t-1} + i_t v_t kᵀ_t`, output via
`h_t = C_t q_t / max(|n_t|, 1)`. sLSTM variant adds scalar memory mixing; the LLM use
the mLSTM-only ("xLSTM[1:0]") form.

### Inference cost
- Constant-memory recurrence per token. O(d²) per layer.
- xLSTM 7B reports 32× faster than transformer at 32K context.
- At 80M: bandwidth-bound on reading the matrix state from L1/L2.

### Quality
- xLSTM[1:0] outperforms Llama / Mamba / RWKV-4 on PALOMA on 568/571 (99.5%) text
  domains at matched training compute.
- xLSTM 7B: comparable downstream to similarly-sized LLMs, much faster decode.

### CPU friendliness
- Matrix state read every token = 2 MB-ish for d=768. Fits in our 96 MB L3 trivially
  but pollutes L1. Tiling becomes important.
- Linear-attention math = matmul + element-wise → vectorizes cleanly.

### Open source
- xLSTM main: https://github.com/NX-AI/xlstm
- xLSTM 7B: https://arxiv.org/abs/2503.13427
- Original paper: https://arxiv.org/abs/2405.04517

### Validation cost
~3-4 weeks. mLSTM kernel is novel; no off-the-shelf INT8 implementation. Higher risk.

## RetNet (Microsoft)

### Math
Multi-scale retention: `S_n = γ S_{n-1} + kᵀ_n v_n; o_n = q_n S_n`, with γ a per-head
decay. Three equivalent forms (parallel for training, recurrent for decode, chunkwise
for both).

### Inference cost
- O(1) decode latency, batch-size invariant.
- Recurrent state: d × d per head. Matrix add + matmul per token.

### Quality
- Comparable to transformer at 1.3B-7B; behind newer Mamba-2 / RWKV-7 on most evals.
- Largely superseded but the chunkwise form influenced everything after.

### Open source
- Paper: https://arxiv.org/abs/2307.08621
- Survey: https://arxiv.org/abs/2506.06708
- Reproducible PyTorch: https://github.com/fkodom/yet-another-retnet

### Validation cost
~2 weeks but lower expected ceiling than Mamba-3 / RWKV-7. Skip unless we need a stepping
stone.

## Gated DeltaNet (NVIDIA, ICLR 2025)

### Math
Combines linear-attention delta rule (precise memory write) with a forget gate:
`S_t = α_t (S_{t-1} - k_tᵀ (S_{t-1} k_t - v_t)) `. Already integrated as the linear
component of Qwen3-Next.

### Inference cost / quality
- Beats Mamba-2 and DeltaNet on language modeling, retrieval, and long-context recall.
- Memory-bound on GPU at decode (FLOP/B ~0.87) — same on CPU but our L3 hides it.

### Open source / paper
- Paper: https://arxiv.org/abs/2412.06464
- Code: https://github.com/NVlabs/GatedDeltaNet

### Validation cost
~2-3 weeks. Considered the strongest pure-linear baseline as of 2026.

# ------------------------------------------------------------------------------------
# Long convolution / Hyena family
# ------------------------------------------------------------------------------------

## Hyena Hierarchy

### Math
Interleave implicit long convolutions (filter parameterized by an FFN) with element-wise
gating. Forward via FFTConv, O(L log² L) per sequence.

### Inference cost
- Training and *prefill*: 5× faster than dense attention at L=8K.
- **Autoregressive decode is the problem.** A naive Hyena is O(L) per generated token
  because the convolution kernel must be re-applied. SSM-form distillations exist
  ("Scavenging Hyena", arXiv:2401.17574) but the inference story is still weak vs
  Mamba/RWKV.
- "Flash Inference" (arXiv:2410.12982) gets to O(L log² L) total but irregular FFT
  memory access does not match VNNI matmul.

### Quality
- Reaches transformer quality with 20% less training compute at L=2K.

### CPU friendliness
- Poor for our target. FFT is bandwidth-intensive, complex, and the convolution does
  not use vpdpbusd. INT8 FFT exists but is exotic.

### Open source
- Paper: https://arxiv.org/abs/2302.10866
- Code: https://github.com/HazyResearch/safari

### Validation cost
High. Would require a custom INT8 FFT kernel. Not recommended at our scale.

# ------------------------------------------------------------------------------------
# Conditional compute (MoE / MoD)
# ------------------------------------------------------------------------------------

## Mixture of Experts (MoE)

### Math
FFN replaced by N expert FFNs; router picks top-k per token. Activated params << total
params.

### Inference cost
- At Veritate scale (80M total) with 8 experts of 10M, top-2 routing: per-token compute
  ≈ 25M activated. Saves ~3× FFN flops.
- **Caveat**: at small scale dense often beats MoE for the same active params (HF MoE
  blog). The win is total-params-to-compute ratio, which we are not capacity-starved on.
- Routing introduces irregular memory access — bad for SIMD batch=1 decode.

### Quality
- At ≥1B scale, MoE matches dense with ~4× less compute. Below 500M the literature is
  thin and mostly negative.

### Open source
- Mixtral, OLMoE, DeepSeek-V2 — all >100M active.

### Validation cost
Medium. Routing dispatcher in the inner loop is a SIMD pessimization at our scale.
**Recommendation: skip for v1. Revisit if we scale to 200M+.**

## Mixture of Depths (MoD)

### Math
Top-k router selects which tokens enter each block; remaining tokens take residual
shortcut. Static computation graph (k fixed).

### Inference cost
- 0.5 budget = ~half FLOPs. Decode-time savings real if router is cheap.
- Known caveat: under some settings MoD *increases* latency (control flow cost).

### Quality
- Lower loss at fixed train budget than vanilla, isoFLOP across 60M-3B.

### Open source
- Paper: https://arxiv.org/abs/2404.02258
- Follow-up MoD-routing: https://arxiv.org/abs/2412.20875

### Validation cost
Low (it's an attention modification, not a new arch). Worth trying as a v3 add-on
*after* the base architecture is chosen.

# ------------------------------------------------------------------------------------
# Compression-as-architecture (BitNet, T-MAC, LUT-NN)
# ------------------------------------------------------------------------------------

## BitNet b1.58

### Math
Weights ∈ {-1, 0, +1}; activations INT8. Multiplication degenerates to sign + add.

### Inference cost
- bitnet.cpp on x86 AVX-512: 2.37×-6.17× over llama.cpp Q4. 100B model at 5-7 tok/s on
  one CPU.
- BitNet b1.58 2B4T: 2B params, 4T tokens, matches FP16 LLMs of similar size.
- 1-bit AI Infra (arXiv:2410.16144) shows lossless inference is achievable on CPU.

### Quality
- Matches FP16 baselines starting at 3B. Below 1B the gap reopens.
- "BitNet b1.58 Reloaded" (arXiv:2407.09527) closes the gap for smaller nets via better
  training recipe.

### CPU friendliness
- **Best of any architecture in this list for CPU.** Multiplication → sign + add → no
  multiplier needed. T-MAC LUT kernels generalize this further (see below).

### Open source
- Microsoft framework: https://github.com/microsoft/BitNet
- Paper: https://arxiv.org/abs/2402.17764
- 2B4T technical report: https://arxiv.org/abs/2504.12285

### Validation cost
~1-2 weeks. **This is orthogonal to architecture choice.** Apply to any of the above.
Effectively replaces our INT8 path with INT2 (sign + add).

## T-MAC (Microsoft, lookup-table mpGEMM)

### Math
Decomposes weight × activation into bit-wise table lookups. Mixed-precision matmul
becomes index + add over a precomputed centroid table.

### Inference cost
- 4-5× over llama.cpp at 3B BitNet. 20 tok/s single-core, 48 tok/s 4-core on Surface
  Laptop 7. Linear scaling FLOPs with bits.
- Raspberry Pi 5: 11 tok/s on 3B BitNet.

### CPU friendliness
- TBL/PSHUFB intrinsic-friendly. AVX-512 vpermb is the x86 native op.

### Open source
- https://github.com/microsoft/T-MAC
- Paper: https://arxiv.org/abs/2407.00088

### Validation cost
~2 weeks to integrate as a backend kernel. **Strongest pure-CPU compression path.**

## LUT-NN / LUT-DLA / Vec-LUT

- LUT-NN (arXiv:2302.03213): centroid learning + table lookup at inference.
- LUT-DLA (arXiv:2501.10658): deep learning accelerator framework.
- Vec-LUT (arXiv:2512.06443): unified LUT across parallel tokens, 2025.

These move the math entirely off the multiplier, replacing matmul with `gather + add`.
Promising for analog-AI fungibility (Mythic-style flash storage).

# ------------------------------------------------------------------------------------
# Byte-level architectures (relevant since Veritate is byte-level)
# ------------------------------------------------------------------------------------

## MambaByte

### Summary
Token-free Mamba operating on raw bytes. MambaByte-972M outperforms other byte-level
models, competitive with subword Mamba via speculative decoding with a subword draft.
- Paper: https://arxiv.org/abs/2401.13660 (COLM 2024)
- Direct fit if we pick Mamba as the base architecture and want to stay byte-level.

## SpaceByte

### Summary
Multi-scale byte transformer with larger blocks inserted only at word boundaries (space
characters). Beats MegaByte at 2.7× less compute. Subword-quality at byte-level.
- Paper: https://arxiv.org/abs/2404.14408 (NeurIPS 2024)

## Byte Latent Transformer (BLT, Meta)

### Summary
Dynamic patches sized by entropy of next byte. **Up to 50% fewer FLOPs at inference vs
Llama-3 at matched training.** Byte-level matches tokenization performance for the first
time at scale (8B / 4T bytes).
- Paper: https://arxiv.org/abs/2412.09871
- Code: https://github.com/facebookresearch/blt

### Veritate fit
BLT's local-attention encoder/decoder + global patch model is heavier than we need at
80M, but the **entropy-patching idea is portable**: variable compute per byte is a
direct latency win on a sub-ms decode budget.

## Multiscale Byte Language Models (MBLM)

- arXiv:2502.14553 — hierarchical byte modeling for million-byte sequences.

# ------------------------------------------------------------------------------------
# Cutting-edge and outrageous (with paper backing)
# ------------------------------------------------------------------------------------

## Discrete Diffusion Language Models (dLLMs)

### Math
Forward process masks tokens, reverse process denoises in parallel. Bidirectional
attention; multi-token updates per step.

### Inference cost
- **Parallel decode is the headline.** Seed Diffusion: 2,146 tok/s on H20 GPU.
  10× speedup over autoregressive at matched quality on code.
- For Veritate: GPU-shaped wins. On CPU the gain is muted because we're already
  generating one token at a time. *Could* fit if we redefine "token" as a 16-byte block.
- Survey: https://arxiv.org/abs/2506.13759
- Block-diffusion for dLLMs: https://arxiv.org/abs/2509.26328 (Fast-DLLM v2, 2025)

### CPU friendliness
- Bidirectional attention = full attention every step → expensive.
- KV-cache tricks: "delayed KV caches" claim 2-10× for long sequences.

### Recommendation
Skip for now. Autoregressive byte-level dLLMs are a 2026 research target; not yet
mature. Watch.

## Energy-based language models

- EDLM (arXiv:2410.21357, ICLR 2025): EBM at full sequence level for each diffusion step.
  Closes the dLLM-vs-autoregressive perplexity gap.
- "Autoregressive LMs are secretly EBMs" (arXiv:2512.15605): theoretical bridge.
- Skip for now. No CPU inference story.

## Hyperdimensional Computing / Vector Symbolic Architectures

### Math
Compositional structures via circular convolution `c = a ⊛ b` and superposition (sum)
in d=10K dimensional vectors. Binding is invertible with the inverse of `⊛`.

### Inference cost / fit
- **Hrrformer** (arXiv:2305.19534): linear-time self-attention via HRR.
  Up to 280× faster training on Long Range Arena.
- Hyperdimensional Probe (arXiv:2509.25045, 2025): VSA-based decoding of LLM
  representations.
- Vector ops are pure SIMD (FFT for circular convolution, element-wise add). INT8 HRR
  is mostly unexplored.

### Recommendation
Strong moonshot candidate. **Hrrformer is the practical hook**: replace softmax
attention with HRR binding/superposition; keep the FFN. 80M HRR LM with no published
baselines at our scale → high upside, high risk.
- Tutorial code: https://github.com/MahmudulAlam/Holographic-Reduced-Representations
- FCAI repo: https://github.com/FutureComputing4AI/Learning-with-Holographic-Reduced-Representations

## Spiking neural networks (SNNs) for language

- **SpikeGPT** (arXiv:2302.13939, ICLR 2025 poster): 46M / 216M params, RWKV-style
  attention-free, binary activations. **32.2× fewer ops on neuromorphic hardware.**
- **SpikeLLM** (arXiv:2407.04752, ICLR 2025): 7B-70B; -25.5% WikiText2 perplexity gain
  at W4A4 over OmniQuant baseline.
- **SpikingBrain** (arXiv:2509.05276, 2025): 7B linear + 76B hybrid-MoE.

### CPU fit
- Binary activations = bit ops. Sparse events in time. Hardware win is on Loihi /
  TrueNorth, **not on x86**. CPUs can't natively benefit from temporal sparsity in
  dense matmul kernels.
- Could be revisited if we ever target an event-driven coprocessor.

### Recommendation
Defer until Veritate looks at neuromorphic hardware (v5+ analog roadmap entry).

## TokenFormer (treat parameters as tokens, ICLR 2025)

### Math
Replace all linear projections with token-parameter cross-attention ("Pattention").
Input tokens are queries; weights are key/value tokens.

### Performance
- 1.4B test ppl 11.77 (vs 11.63 transformer) at half the training cost.
- Enables progressive scaling without retraining.

### Veritate fit
Interesting for training cost / continual learning. Inference is still attention-shaped
→ no decode-cost win over transformer. Skip for the inference goal; revisit for training
ergonomics.
- Paper: https://arxiv.org/abs/2410.23168
- Code: https://github.com/Haiyang-W/TokenFormer

## Trainable activation functions (drop-in wins)

### xIELU (arXiv:2411.13010, late 2024 / 2025)
- Trainable piecewise integral-of-ELU.
- 1.1B Llama, 126B FineWeb-Edu tokens: **lower perplexity than ReLU² and SwiGLU at
  matched compute and params.**
- One exp() call vs SwiGLU's ~two — slightly cheaper.

### Expanded gating variants (xGELU, xSiLU, xSwiGLU; arXiv:2405.20768)
- Trainable scaling on gates outperforms static GLU.

### CPU fit
Activation cost is 3-5% of forward pass. INT8 LUT for any activation is trivial
(we already plan a GELU LUT). Replacing GELU with xIELU is a free swap.

### Recommendation
**Easy win.** Add xIELU as an alternative activation behind a compile-time switch.

## FlashAttention-3 for decode

- FA3 (arXiv:2407.08608) is **prefill-bound**. Single-token decode sees little
  difference. Memory bandwidth, not compute, is the bottleneck.
- **Flash-Decoding** (PyTorch blog, Dao et al.): split-K attention for decode. Helps
  long-context decode, irrelevant at our seq=256.
- Not applicable to Veritate's transformer baseline.

## Hybrid Mamba-Transformer (Hymba, Nemotron-H)

- **Hymba** (NVIDIA, ICLR 2025): hybrid attention + SSM heads in parallel within each
  layer. 1.5B beats Llama-3.2-3B on commonsense, **3.49× faster, 14.7× smaller cache**.
  Cross-layer KV sharing + sliding-window attention.
- **Nemotron-H** (NVIDIA): replaces 92% of attention with Mamba-2; 3× throughput vs
  Llama-3.1.
- AI21 Jamba: Mamba + transformer mix at 50B+.

### Veritate fit
Same playbook as Griffin but with Mamba-2 as the recurrent half. **Strongest hybrid
candidate** for 80M scale; Hymba's commonsense win at 1.5B is the closest published
analog to our target.
- Hymba paper: https://arxiv.org/abs/2411.13676
- AI21 hybrid commentary: https://www.ai21.com/blog/rise-of-hybrid-llms/

# ------------------------------------------------------------------------------------
# Summary table (Veritate-relevant axes)
# ------------------------------------------------------------------------------------

| Architecture | Per-token cost | KV/state | INT8 ready | SIMD fit | OSS | Quality vs T at <100M |
|---|---|---|---|---|---|---|
| Transformer (baseline) | O(d² + L·d) | growing | yes | excellent | many | reference |
| Mamba-2 | O(d·N + d²) | constant | yes (DSQ, BitMamba) | excellent | mamba, bitmamba.c | match |
| Mamba-3 | O(d·N + d²) | constant | partial | excellent | NVlabs | +0.6 pp @ 1.5B |
| RWKV-7 | O(d²) | constant | yes (rwkv.cpp) | excellent | rwkv.cpp, llama.cpp | match-to-better |
| Griffin/Hawk | O(d) + window | constant + window | partial | excellent | hawk-pytorch | best @ small |
| Gated DeltaNet | O(d²) | constant | partial | excellent | NVlabs | match Mamba-2 |
| RetNet | O(d²) | constant | yes | excellent | yet-another-retnet | weaker |
| xLSTM (mLSTM) | O(d²) | constant | partial | good | NX-AI/xlstm | match-to-better |
| Hybrid (Hymba) | O(d·N + d·w) | bounded | partial | excellent | nvidia (weights) | best published |
| Hyena | O(L log²L) | implicit conv | weak | poor (FFT) | safari | match @ 2K |
| MoE | O(d²/k) | growing | yes | poor (routing) | many | weaker @ <500M |
| MoD | O(0.5·d²) | growing | yes | medium | research | small win |
| BitNet b1.58 | O(d²) but no mul | growing | n/a (intrinsic) | excellent | microsoft/BitNet | match @ 3B+ |
| Hrrformer (HRR) | O(d log d) FFT | constant | unexplored | medium | research | weak signal |
| SpikeGPT | O(d²) sparse | constant | binary | poor on x86 | bitbrain-ai | match-to-weaker |
| dLLMs | parallel | none | weak | poor (full attn) | facebookresearch | match (large) |
| TokenFormer | O(d²) | growing | partial | medium | Haiyang-W | match |

# ------------------------------------------------------------------------------------
# Concrete recommendations for Veritate
# ------------------------------------------------------------------------------------

### Tier 1 — port one of these to the harness next
1. **RWKV-7 Goose 0.19B → distill to 80M.** Lowest-friction port. Existing rwkv.cpp
   gives us a scalar oracle. Vector-matrix recurrence is exactly our kernel.
2. **Mamba-2 130M → 80M.** Selective scan kernel is new code but small. INT8 path
   already proven at adjacent scale.
3. **Hymba-style hybrid.** Mamba-2 head + sliding-window attention head per layer.
   Keeps our existing attention kernel as the local-attention path. Highest expected
   quality at 80M.

### Tier 2 — orthogonal additions, any base
4. **xIELU** as the activation. Free perplexity win.
5. **T-MAC LUT kernel** as a compile-time backend. Matches our v5 JIT direction.
6. **Entropy-patching from BLT** — compute per byte ∝ next-byte entropy. Variable
   per-byte latency, average dominated by predictable bytes.

### Tier 3 — moonshots, watch list
7. **Hrrformer / HRR-attention.** Linear-time, vectorizable, completely off the
   beaten track.
8. **BitNet b1.58 ternary** as the weight format under any of Tier 1.
9. **Discrete diffusion at byte level** — if a credible CPU paper appears in 2026.

# ------------------------------------------------------------------------------------
# Quantization (INT8 / INT4)
# ------------------------------------------------------------------------------------

### Q8_0 — block quantization with scale
- 32 INT8 values per block, one fp16 scale.
- Dequantize: `float v = int8 * scale`.
- Used by llama.cpp, GGUF, Whisper.cpp.
- ~1% accuracy loss on most LLM benchmarks vs fp16.

### Q4_0 / Q4_K_M — INT4 weights
- 4 bits per weight. 8x smaller than fp32.
- Q4_K_M (used by llama.cpp default) groups 32 values with one scale + one min.
- Loss is real but tolerable. 7B Q4 models fit in 4 GB.

### Activation quantization (the hard part)
- Weights are static — quantize once at conversion time, done.
- Activations change per input. Two strategies:
  - **Dynamic per-token** — recompute scale every forward pass. Good accuracy, small cost.
  - **Static (PTQ)** — calibrate on a dataset, freeze the scale. Faster but more loss.
- Veritate uses dynamic per-token in v1 → v2 considers static for the deployed binary.

### SmoothQuant, AWQ, GPTQ — modern PTQ methods
- AWQ (Activation-aware Weight Quantization) — scales weights based on activation
  magnitude. State of the art for INT4 LLMs as of late 2025.
- We don't need to implement these — just ingest weights from a quantizer.

### QAT for INT4 (2025 state of art)
- Recover up to 70% of accuracy lost vs PTQ; +1-3% on GPQA / MMLU Pro.
- Gemma 3 4B QAT: 8 GB → 2.6 GB at int4 with negligible loss.
- PyTorch torchao + Unsloth ship QAT loops. We ingest weights, no training code needed.

# ------------------------------------------------------------------------------------
# Analog AI hardware — the long game
# ------------------------------------------------------------------------------------

### Mythic AI
- Flash memory cells store weights as conductance values.
- Matmul = Ohm's law (V × G = I) summed by Kirchhoff's law down each column.
- Analog → ADC at the column → digital activation → next layer.
- Effective precision: ~8-bit. Power: 25 TOPS at 3W.
- Status: M1076 chip ships in production. Limited dev kit access.

### IBM Research — phase-change memory (PCM)
- Crystalline vs amorphous state encodes weight value.
- True non-volatile analog matmul.
- HERMES project: 64-tile chip, 14 nm, 12.4 TOPS/W.
- Status: research; not commercial.

### Lightmatter / Lightelligence — photonic
- Light interference replaces electron flow.
- Mach-Zehnder interferometer arrays multiply by phase shift.
- 10× faster than GPU at fraction of the power for matmul.
- Status: Lightmatter Envise shipping to select customers.

### Rain Neuromorphics
- Analog neural net that physically resembles a brain.
- Backed by Sam Altman, OpenAI alumni.
- Status: pre-product.

### Why this matters for Veritate
INT8 code is the bridge. A model written for digital INT8 SIMD maps cleanly onto analog
matmul arrays. The kernel interface stays the same; only the backend changes. Veritate's
`matmul()` function pointer could one day point to `matmul_mythic()` or `matmul_lightmatter()`.

# ------------------------------------------------------------------------------------
# CPU SIMD — what each ISA gives us
# ------------------------------------------------------------------------------------

| ISA | Width | INT8 throughput per cycle | Year | Notes |
|---|---|---|---|---|
| SSE4 | 128 b | ~16 muls (PMADDUBSW) | 2008 | Baseline. Skip unless desperate. |
| AVX2 | 256 b | ~32 muls (vpmaddubsw + vpmaddwd) | 2013 | Universal target. |
| AVX-512 | 512 b | ~64 muls | 2017 | Skylake-X+, Zen 4+. |
| AVX-512 VNNI | 512 b | 64 muls in 1 inst (vpdpbusd) | 2019 | Ice Lake+, Zen 4+. *Big win.* |
| AVX10 | 256/512 b | Same as VNNI | 2024+ | Intel's unified path forward. |
| ARM NEON | 128 b | ~16 muls (smull/smlal) | 2011 | Apple Silicon, Raspberry Pi. |
| ARM SDOT/UDOT | 128 b | 16 muls in 1 inst | 2017+ | Equivalent of VNNI on ARM. |
| ARM SVE2 | 128–2048 b | scalable | 2019+ | NOT on Apple Silicon. |
| Apple AMX | matrix | 512 muls/cycle | 2020 | Undocumented coprocessor. |

# ------------------------------------------------------------------------------------
# Speedup techniques we'll use later
# ------------------------------------------------------------------------------------

### Speculative decoding
Tiny draft model proposes N tokens, big model verifies in parallel. 2–3× speedup at zero
quality cost. Only relevant once Veritate has a real LM forward pass (v3+).

### KV cache + Flash Attention
Standard for transformer inference. Not relevant until v3.

### Kernel fusion
Combine multiple ops into one pass to avoid round-trips through RAM. E.g., matmul + bias
+ activation in one kernel. Veritate already does this implicitly because we hand-write.

### Unrolling + software pipelining
Manually unroll the inner loop and interleave loads with arithmetic to hide latency.
Critical for hitting peak FMA throughput. We'll do this in v2 once we move to pure NASM.

# ------------------------------------------------------------------------------------
# JIT / runtime binary emission — software ASIC at runtime
# ------------------------------------------------------------------------------------

Brainstorm note from the user: what if we wrote machine code directly, bypassing assembly?

Direct answer: writing raw bytes is one level below assembly. Assembly assembles to those
exact same bytes — the CPU sees no difference. So raw bytes for static code is pure
tedium with zero speedup.

The interesting unlock is RUNTIME emission. Generate a kernel with M, N, K (and even the
weight matrix shape) baked in as immediates at the moment we know them. The compiler can
then constant-fold loop bounds, unroll perfectly, and skip every runtime shape check. This
is exactly what an ASIC does — the dataflow is wired in. Doing it in software at startup
gives us the same thing for the duration of a process.

Real systems doing this:
- **AsmJit** / **Xbyak** — C++ libraries for emitting x86 machine code at runtime.
- **Halide** — schedule + IR + code generator; specializes kernels for specific shapes.
- **Tiramisu** / **TVM AutoTVM** — search-based kernel generators.
- **MLIR codegen** — LLVM project's IR for emitting specialized inference kernels.
- **JAX `jit`** — traces, compiles, and caches kernels per input shape.

For Veritate, three escalating options:
1. **Static specialization via #defines** (today): MN K are compile-time constants. The
   compiler unrolls accordingly. We already do this implicitly.
2. **Multi-shape AOT** (v3): build N specialized binaries for the most common shapes,
   dispatch by shape at runtime.
3. **JIT** (v5+): emit machine code at startup once shapes are known. Use AsmJit-style
   approach. Costs ~1 ms at startup for compile, recovered on every subsequent inference.

The win for Veritate would be hardcoding the matmul kernel for the exact (M,N,K) of each
layer in the trained model. No bounds checks, no tail handling, fully unrolled, perfect
register allocation. Likely 1.5–3× over generic templated kernels.

Decision: defer to v5. Too much complexity for v1–v4.

# ------------------------------------------------------------------------------------
# Latency reduction — hiding inference behind the user
# ------------------------------------------------------------------------------------

The user almost never notices model latency that happens BEFORE they hit Enter. The input
phase (typing, thinking, editing) is dead time we can spend on the prefill. Three layered
techniques:

### Streaming prefill (a.k.a. incremental prefill)
As the user types each token, run forward pass on the partial prompt and persist the KV
cache. By the time they submit, the cache is already hot for the entire input — only the
final tokens need to be processed at submit-time. Net effective latency: the cost of the
last 1–3 tokens, not the whole prompt.

- Cost: forward pass per keystroke. Mitigate with debounce (process at word boundaries or
  every 50 ms idle).
- Risk: user edits/deletes — the KV cache becomes invalid for those positions. Solution:
  KV cache is a tree, not a chain. Rewind to the divergence point and reuse the prefix.

### Speculative prefill
Predict what the user is about to type and prefill that too. If they type "what's the
weather in San", predict "Francisco" and prefill its KV. If correct, latency is negative
(work was done before the keystroke). If wrong, throw it away — cost is bounded.

### Speculative decoding (the post-submit twin)
Once the user has submitted, a tiny draft model proposes tokens and the big model verifies
many at once. Standard 2–3× speedup. Combines with streaming prefill: prefill is hot,
decode is fast, output starts appearing in <50 ms.

### Why this is powerful for Veritate
Veritate already targets sub-ms matmul. With streaming prefill on top, the user-perceived
latency is bounded by typing speed, not model speed. This turns "fast inference" into
"inference appears instantaneous."

Adoption decision:
- v1–v2: out of scope (we're benchmarking matmul, not running a model).
- v3 (transformer forward pass): streaming prefill becomes the default UX.
- v4+: explore speculative prefill once we have an autoregressive draft model.

References:
- vLLM "continuous batching" — same idea, server-side.
- llama.cpp `--keep` and prompt caching — primitive form of KV cache reuse.
- Apple "predictive text" on iOS — streaming prefill in production for years.
