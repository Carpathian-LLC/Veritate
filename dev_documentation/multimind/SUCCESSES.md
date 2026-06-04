# multimind SUCCESSES

Validated research findings for the Multi-Mind PoC, with numbers + falsifier each. Bug fixes, refactors, and plumbing do NOT go here; they live in git history. See `claude_preflight.md` rule 40a.

Each entry follows this shape:

```
### W<N>: <one-line claim>
- falsifier (verbatim from ideas.md): <one line>
- result: <one line with numbers>
- artifact: <path to smoke + stats JSON>
- date: YYYY-MM-DD
```

## entries

### W1g TEMP-2.0 BEST: kitchen-sink + gate softmax temperature 2.0
- falsifier: KL ratio >= 2.0 AND ||g||_2 > 0.1
- result: ratio = **186.52x** (2.6x over previous best 72x), ||g||_2 = **0.8999**, baseline_kl = 0.0059, with_bias_kl = **1.109 nats (62% of log(6)=1.79 ceiling)**.
- mechanism: smoother softmax (T=2.0) lets the bias signal influence ALL experts, not just the top-2. Sweep: temp 0.5=2.8x, 1.0=72x, 2.0=186x best, 3.0=4.4x (non-monotonic; sweet spot at 2.0).
- artifact: dev_documentation/multimind/W1g_temp_2_0.json
- date: 2026-05-23

### W1g KITCHEN SINK: per-layer g + refractory + curriculum-amplified s (best-of-all)
- falsifier: KL ratio >= 2.0 AND ||g||_2 > 0.1
- result: ratio = **72.16x**, ||g||_2 = **1.189**, baseline_kl = 0.0059, with_bias_kl = 0.4290 nats (24% of log(6)=1.79 ceiling, matching OLMoE-from-scratch specialization)
- mechanism stack: per-layer g (matrix shape (layers, n_experts)) + refractory_steps=4 during training + 10x LR on g + Normal(0, 0.1) init + curriculum sentiment amplification 3x->1x over training
- artifact: dev_documentation/multimind/W1g_kitchen_sink.json
- date: 2026-05-22

### Integration: per-layer g + refractory (training time)
- falsifier: KL ratio >= 2.0 AND ||g||_2 > 0.1
- result: ratio = 13.04x, ||g||_2 = 0.4210; mechanisms compose cleanly
- artifact: dev_documentation/multimind/W1g_per_layer_refractory.json
- date: 2026-05-22

### W1g per-layer g
- falsifier: KL ratio >= 2.0 AND ||g||_2 > 0.1
- result: ratio = 23.04x, ||g||_2 = 0.2934; signal moved from L0 to L1
- artifact: dev_documentation/multimind/W1g_per_layer.json
- date: 2026-05-22

### W1g no-aux-load-balance
- falsifier: KL ratio >= 2.0 AND ||g||_2 > 0.1
- result: ratio = 11.03x, ||g||_2 = 0.1979 (baseline drifts up to 0.0059 without aux — aux helps baseline cleanliness, not bias signal)
- artifact: dev_documentation/multimind/W1g_no_aux.json
- date: 2026-05-22

### W1e: oracle-forced g (gold-standard architecture controller)
- falsifier: KL(with-bias) >= 2x KL(baseline) AND ||g||_2 > 0.1
- setup: g_e initialized to [+1,+1,+1,-1,-1,-1] and FROZEN; only router + FFN trained. Tests "can the architecture use a bias signal at all" independent of training pathway.
- result: KL ratio = 3564x baseline; ||g|| = 2.45 (sqrt(6), by construction); baseline KL = 0.0039 nats; with-bias KL = 13.74 nats (effectively deterministic per-class routing).
- interpretation: ARCHITECTURE DEFINITIVELY WORKS. The MtM premise (per-expert bias vector modulating routing) is sound. W1's failure (||g||=0.014) was a training-signal weakness, not an architectural impossibility. Next step: engineer training so g learns to grow to magnitude ~0.5+ via higher LR on g + nonzero init (W1g).
- artifact: dev_documentation/multimind/W1b_results.json (named W1b in the file because falsifier_v2.py reused the W1b label; intended W1e)
- date: 2026-05-21

### W1: affect bias shifts MoE routing (oracle signal)
- falsifier: KL(with-bias) >= 2x KL(baseline) AND ||g||_2 > 0.1
- result: KL ratio = 34.124, ||g||_2 = 0.2181, baseline KL = 0.0047, with-bias KL = 0.1601
- artifact: dev_documentation/multimind/W1_results.json
- date: 2026-05-21

### W2: learned affect probe (replaces W1 oracle scalar)
- falsifier: KL(with-probe)/KL(with-oracle) >= 0.5 AND ||g||_2 > 0.1
- result: ratio_vs_oracle = 1.568, ||g|| = 0.2216, probe_val_acc = 0.8609375
- artifact: dev_documentation/multimind/W2_results.json
- date: 2026-05-21

### W3: region naming via specialty corpora
- falsifier: per-region ppl on own specialty >= 10% lower than on other AND target region is top-1 for >= 50% of own-specialty bytes; >= 50% of regions must pass.
- result: 4/6 regions passed
- artifact: dev_documentation/multimind/W3_results.json
- date: 2026-05-21

### W4: refractory inhibition burst-iness vs ppl
- falsifier: max stickiness drop >= 0.05 AND ppl ratio <= 1.1
- result: max stickiness drop = 0.3152, ppl ratio = 1.0232
- artifact: dev_documentation/multimind/W4_results.json
- date: 2026-05-22

### W1: affect bias shifts MoE routing (oracle signal)
- falsifier: KL(with-bias) >= 2.0x KL(baseline) AND ||g||_2 > 0.1
- result: KL ratio = 23.039, ||g||_2 = 0.2934, baseline KL = 0.0032, with-bias KL = 0.0743
- artifact: dev_documentation/multimind/W1g_per_layer.json
- date: 2026-05-22

### W1: affect bias shifts MoE routing (oracle signal)
- falsifier: KL(with-bias) >= 2.0x KL(baseline) AND ||g||_2 > 0.1
- result: KL ratio = 11.028, ||g||_2 = 0.1979, baseline KL = 0.0059, with-bias KL = 0.0650
- artifact: dev_documentation/multimind/W1g_no_aux.json
- date: 2026-05-22
