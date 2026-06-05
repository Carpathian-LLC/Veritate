# Multi-Mind (MtM) deep prior-art scan, 2026-05-21

Architecture under review: byte-level transformer with 6 named brain-region MoE experts (broca, wernicke, hippocampus, prefrontal, cerebellum, thalamus), routed by a frozen affect probe whose scalar valence is added to gate logits through a learned per-expert bias `g_e`, plus per-region sleep adapters for offline consolidation and a slot-bank hippocampus addon.

## 1. has the exact stack been built before?

No published artifact matches the full stack. The closest near-miss is **Mixture of Cognitive Reasoners (MiCRo / MCR)**, EPFL, June 2025 ([arxiv 2506.13331](https://arxiv.org/abs/2506.13331)). MiCRo partitions the layers of a *pretrained subword LLM* into four expert modules aligned with the brain's language, logic, social-reasoning, and default-mode networks. Routing is by a learned token-level gate; there is no affect sidechannel and no byte-level vocabulary, no sleep adapter, no slot memory. They claim novelty on "interpretable + steerable expert modules aligned to cognitive networks." The naming convention overlaps MtM at the conceptual level (named cognitive regions) but the routing signal, vocabulary, and adaptation mechanism differ.

Other partial matches:

- **BriLLM** (March 2025, [arxiv 2503.11299](https://arxiv.org/abs/2503.11299)): brain-inspired LLM where tokens map to "specialized nodes analogous to cortical areas" via Signal Fully-connected (SiFu) propagation. Not byte-level; not MoE in the standard sense; uses subword tokens at 40k vocab; no affect routing; no sleep.
- **Lilith** (July 2025, [arxiv 2507.04575](https://arxiv.org/abs/2507.04575)): position paper proposing modular LLMs with thinking/memory/sensory/regulatory modules and "chemical signaling" via token messages. No implementation, no benchmarks, no routing math.
- **Modular Agentic Planner (MAP)**, Nature Communications 2025 ([nature.com/articles/s41467-025-63804-5](https://www.nature.com/articles/s41467-025-63804-5)): orchestrates *multiple LLM calls* as brain-region modules (conflict monitor, state predictor, etc.). Multi-LLM agent pipeline, not a single-model architecture.
- **BrainStack** ([arxiv 2601.21148](https://arxiv.org/pdf/2601.21148)): MoE for EEG decoding where each expert handles an anatomical cortical region. Inputs are literal brain signals, not text.

Verdict on 1: the **named-region + byte-level + affect-gated-bias + sleep adapter** combination has not been published. The named-region MoE piece alone is prior art (MiCRo, BrainStack).

## 2. brain-inspired modular LLMs, 2024-2025

- **MiCRo** ([arxiv 2506.13331](https://arxiv.org/abs/2506.13331)): four cognitive-network experts. Most direct comparator.
- **BriLLM** ([arxiv 2503.11299](https://arxiv.org/abs/2503.11299)): 1-2B SiFu-flow model, GPT-1-level generation.
- **MAP** (Nature Comms 2025): multi-LLM brain-region planner.
- **Brain-like Functional Organization within Large Language Models** ([arxiv 2410.19542](https://arxiv.org/abs/2410.19542)): post-hoc analysis showing standard LLM neurons spontaneously cluster into functional networks resembling brain regions. Relevant: validates the *premise* that brain regions are a reasonable factorization for FFN experts.
- **Numenta Thousand Brains Project** ([arxiv 2412.18354](https://arxiv.org/abs/2412.18354), spun off January 2025): sensorimotor learning paradigm, not language; ships "Monty" implementation. Disjoint architecturally but the closest "brain-first" research program. Their commercial arm (NuPIC) ships sparsity-accelerated transformer inference.
- **Anthropic interpretability** ([Sonnet 4.5 emotion-vector report, Nov 2025](https://www.edtechinnovationhub.com/news/anthropic-uncovers-emotion-like-mechanisms-shaping-ai-behavior)): 171 distinct emotion concepts emerge spontaneously as linear directions inside a frontier LLM. Validates the affect-probe approach in MtM W2; Anthropic does not route experts on these signals, they just observe.
- No public DeepMind brain-team or Bengio-group work in 2025 ships anatomical-region FFN experts.

## 3. per-region adapters with sleep / consolidation

The general "wake/sleep" framing has prior art but not in production LLMs with rolling-buffer conversation consolidation.

- **Wake-Sleep Consolidated Learning (WSCL)**, AAAI 2024 ([arxiv 2401.08623](https://arxiv.org/abs/2401.08623)): the seminal modern wake-sleep paper for continual learning. Vision classification, not language. Has a short-term hippocampal buffer; NREM-stage replay consolidates synapses; REM stage explores feature space. No LoRA; no per-region split.
- **Sleep-like unsupervised replay** ([researchgate 366321775](https://www.researchgate.net/publication/366321775)) and **SESLR** ([arxiv 2507.02901](https://arxiv.org/abs/2507.02901)): biological-style sleep replay reduces catastrophic forgetting in CNNs / spiking nets.
- **CL-LoRA** (CVPR 2025): rehearsal-free continual LoRA with orthogonal task subspaces.
- **CoDyRA** ([arxiv 2412.01004](https://arxiv.org/abs/2412.01004)): dynamic rank-selective LoRA where rank is *chosen per task*; explicitly mitigates the rank-vs-forgetting tradeoff MtM hit on W5.
- **Subspace Geometry Governs Catastrophic Forgetting in LoRA** ([arxiv 2603.02224](https://arxiv.org/pdf/2603.02224)): forgetting is governed by the *principal angle between task gradient subspaces*, not rank. Direct prescription for W5: project the sleep update orthogonal to prior task gradients.
- **OPLoRA** ([arxiv 2510.13003](https://arxiv.org/abs/2510.13003)): orthogonal-projection LoRA, prevents forgetting during PEFT.

No production LLM ships periodic per-region adapter consolidation on rolling conversation buffers. Closest in spirit are agentic-memory frameworks (MIRIX, A-MEM, Mem0, Memoria, [arxiv 2502.12110](https://arxiv.org/abs/2502.12110), [arxiv 2512.12686](https://arxiv.org/abs/2512.12686)) — these store text in external stores, not weight updates.

For W5: the prescription is CL-LoRA / OPLoRA style orthogonal subspaces per region, plus dose-tied-to-novelty (lower rank when new gradient subspace is parallel to existing one).

## 4. routing-by-sidechannel (non-token signal added to MoE gate)

This is the strongest novelty axis for MtM. Surveyed sidechannels:

- **Task-conditioned routing** ([arxiv 2603.11114](https://arxiv.org/pdf/2603.11114)): router conditioned on task identity. Discrete categorical sidechannel.
- **Modality-aware routing**, SMAR ([arxiv 2506.06406](https://arxiv.org/pdf/2506.06406)): modality identity added to MoE gate to preserve text capability while adding vision.
- **CLIP-MoE** ([arxiv 2409.19291](https://arxiv.org/pdf/2409.19291)): visual features routed to image-specialized experts. Image embedding sidechannel.
- **CAT-MoEformer** ([arxiv 2605.19997](https://arxiv.org/html/2605.19997)): scenario labels + normalized speed as routing-gate input for beam prediction.
- **Robotics MoE locomotion** ([arxiv 2602.00678](https://arxiv.org/pdf/2602.00678)): proprioception conditions terrain-expert selection.
- **LAR-MoE** ([arxiv 2603.08476](https://arxiv.org/pdf/2603.08476)): latent task vectors guide robotic-imitation expert routing.
- **GET** ([arxiv 2601.15906](https://arxiv.org/pdf/2601.15906)): affective adaptation in foundation models concentrates in the FFN gate projection — empirical mechanistic evidence that the *gate* is the right surgical site for affect, exactly where MtM adds `g_e`.

What is unique to MtM: a **frozen unsupervised affect-probe** (scalar continuous valence) added as a **per-expert additive bias** to the gate logits, on a **byte-level** model. CLIP-MoE conditions on dense vision embeddings; SMAR on a one-hot modality flag; proprioceptive MoE on continuous sensor vectors. No one I found routes on a single scalar emotional valence projected onto a learned per-expert bias vector. This is incremental relative to "sidechannel-conditioned MoE" as a class, novel in the specific signal type.

## 5. slot memory / external KV for LLMs (W6 comparison)

- **Memorizing Transformers** (Wu et al. 2022, [arxiv 2203.08913](https://arxiv.org/abs/2203.08913)): kNN over cached KV pairs. Reads heavy at inference (kNN per token).
- **Infini-attention** (Google, April 2024, [arxiv 2404.07143](https://arxiv.org/abs/2404.07143)): fixed-size compressive memory in attention; 114x compression vs Memorizing Transformers. HuggingFace's reproduction concluded it underperforms in the zero-shot retrieval limit ([huggingface.co/blog/infini-attention](https://huggingface.co/blog/infini-attention)).
- **MemoryLLM** ([arxiv 2602.00398](https://arxiv.org/pdf/2602.00398)): 1B-parameter memory pool, O(1) updates; degrades past 20k tokens. **M+** ([arxiv 2502.00592](https://arxiv.org/html/2502.00592v1)) adds a co-trained retriever, 8x retention.
- **∞-former** (Martins et al. 2022, [arxiv 2109.00301](https://arxiv.org/abs/2109.00301)): continuous-attention long-term memory.
- **RetNet** ([arxiv 2307.08621](https://arxiv.org/abs/2307.08621)): retention via recurrent KV summary; not a slot bank.
- **HMT** (NAACL 2025, [aclanthology 2025.naacl-long.410](https://aclanthology.org/2025.naacl-long.410.pdf)): hierarchical memory layers.
- **EM-LLM** (ICLR 2025): event-segmented episodic memory.

At 10-50M params, the W6 design (rank-64 attention-pooled slot bank, 1 lookup per byte, passive write) is closest to **a stripped Memorizing Transformer with attention pooling instead of kNN**. The "1 lookup per byte" is unusually cheap; the question that decides whether W6 ships is whether 10-50M-param models can extract enough signal from a passively-written bank to justify the bytes per second. Memorizing Transformers needed kNN over real KV; M+ needed a *co-trained retriever*. Passive writes have a known recall ceiling: HuggingFace's negative result on Infini-attention is the cautionary tale.

## 6. AGI-at-small-scale papers, 2025

The closest takes-it-seriously papers:

- **NVIDIA "Small Language Models are the Future of Agentic AI"** ([arxiv 2506.02153](https://arxiv.org/abs/2506.02153)): orchestrated SLMs as agentic system, not a single-model AGI claim, but argues SLMs suffice for most agent tasks.
- **MiCRo** (above): explicitly markets modular cognitive specialization as an interpretability + capability path.
- **BriLLM** (above): explicit brain-inspired architecture targeting "biological AGI" framing.
- **MAP** (Nature Comms 2025): brain-inspired planner argues modular structure unlocks planning.

No paper takes "AGI on a $300-1000 box at 10-100M parameters" seriously as a research target. The closest framings (SLM-agent, MiCRo) accept that frontier capabilities require billions of params and orchestrate or specialize from there. MtM's framing is novel as a research stance even if individual architectural pieces are not.

## 7. the one technique we are missing

**Orthogonal-subspace constraints on the per-region sleep adapter (CL-LoRA / OPLoRA / Subspace-Geometry).**

W5 over-adapted at rank 16 / 200 steps / lr 2e-3. The Subspace Geometry paper ([arxiv 2603.02224](https://arxiv.org/pdf/2603.02224)) establishes that forgetting is governed by the principal angle between the new sleep gradient and the pre-sleep weight subspace, not by rank. The single highest-value borrow: **project the sleep LoRA update orthogonal to the top-k principal components of the pre-sleep region weights before applying it.** This is a 30-line OPLoRA-style change to the sleep step. It decouples the dose knob from the forgetting knob — exactly what W5 needs. Empirically, OPLoRA reports near-zero forgetting at ranks 16-64. This is more leverage than any new region, any sleep schedule tuning, or any slot-bank rework.

## novelty verdict

| MtM piece | verdict |
|---|---|
| Named-region MoE experts | **incremental.** MiCRo and BrainStack ship cognitive-region experts; MtM uses six anatomical names instead of four cognitive networks. The naming taxonomy is unique but the architectural pattern is prior art. |
| Byte-level vocabulary | **incremental.** Meta BLT ([arxiv 2412.09871](https://arxiv.org/abs/2412.09871)) established byte-level as competitive at scale. MtM's specific byte-level + MoE combo is unpublished but the byte-level piece alone is not novel. |
| Frozen affect-probe scalar added to gate logits via learned per-expert bias `g_e` | **genuinely new.** Sidechannel-conditioned MoE exists (task, modality, sensor, vision), but no public work routes on a single scalar continuous affect signal with a learned per-expert additive bias. GET shows the FFN gate is the right surgical site for affect; MtM operationalizes it as routing. This is the most defensible novelty. |
| Per-region sleep adapters on rolling buffers | **incremental.** WSCL established wake-sleep consolidation; CL-LoRA established continual-LoRA. Per-region split is the new wrinkle. Not yet competitive with orthogonal-subspace methods. |
| Slot-bank hippocampus, rank-64 attention-pool, passive write | **incremental.** Memorizing Transformers, Infini-attention, MemoryLLM all cover this design space. MtM's per-byte cheap read is unusual; passive-write recall is a known failure mode. |
| Byte-level + 6 named regions + affect-gated MoE + per-region sleep + slot bank, in one ~50M-param model | **genuinely new as a *system*.** No single paper assembles this stack. Whether the assembly justifies a paper depends on whether the affect-gate result generalizes and whether sleep gets fixed (see section 7). |

Defensible novelty for a paper: **"continuous affect-conditioned per-expert gate bias on a byte-level MoE"** as a self-contained contribution, with the named-region taxonomy and sleep/slot-bank as secondary results showing the same backbone supports interpretable continual adaptation.
