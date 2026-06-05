# multimind FAILURES

Falsified research claims for the Multi-Mind PoC. Three-line format per ROE rule 6. Move entries here from ideas.md when an experiment falsifies its claim with a clear number. Bug fixes do NOT go here; they live in git history. See `claude_preflight.md` rule 40a.

Each entry follows this shape:

```
### W<N>: <one-line claim that failed>
- falsifier + numbers: <one line: claim required X, measured Y on artifact Z>
- retry condition: <one line: what would have to change for this to be worth re-running>
- date: YYYY-MM-DD
```

## entries

(empty until W1 verdicts)

### W1: affect bias shifts MoE routing (oracle signal)
- falsifier + numbers: required KL ratio>=2.0 AND ||g||>0.1; measured ratio=12.947 (PASS-side), ||g||=0.0136 (FAIL-side). Absolute KL_with_bias = 0.0424 nats (layer 0 dominated), baseline KL = 0.0033 nats.
- **interpretation (important)**: the KL ratio massively exceeds threshold — routing DID specialize by sentiment in with-bias vs baseline. But ||g|| stayed near zero, meaning the bias term `g_e * s` was NOT the mechanism. Either weight decay (0.01 in AdamW) pulled `g` back faster than gradient could push it, or the "CE found sentiment elsewhere" mode (router learning sentiment from x indirectly). Architecture not falsified; experimental setup needs to force the signal through `g` to certify causality.
- retry condition: W1b (freeze-tail + z-loss) forces `g` to be load-bearing in the last 500 steps. Also try excluding `g` from AdamW weight decay (one-line fix).
- artifact: dev_documentation/multimind/W1_results.json
- date: 2026-05-21

### W1b: affect bias shifts MoE routing with freeze-tail + z-loss
- mitigation: freeze non-router params for final 500 steps; router z-loss=0.001
- falsifier + numbers: required KL ratio>=2.0 AND ||g||>0.1; measured ratio=1.900, ||g||=0.0086
- retry condition: try per-layer g (W1d) or longer training
- artifact: dev_documentation/multimind/W1b_results.json
- date: 2026-05-21

### W4: refractory inhibition burst-iness vs ppl
- falsifier: max stickiness drop >= 0.05 AND ppl ratio <= 1.1
- result: max stickiness drop = 0.3533, ppl ratio = 4.3581
- artifact: dev_documentation/multimind/W4_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 1.17 (drop +56.8%); holdout 2.64 -> 4.77 (rise +81.0%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 2.55 (drop +6.2%); holdout 2.64 -> 2.65 (rise +0.6%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 2.13 (drop +21.7%); holdout 2.64 -> 2.85 (rise +8.2%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 2.21 (drop +18.5%); holdout 2.64 -> 2.79 (rise +5.9%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.72 -> 2.41 (drop +11.3%); holdout 2.64 -> 2.73 (rise +3.7%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.72 -> 1.86 (drop +31.3%); holdout 2.64 -> 3.35 (rise +27.0%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.72 -> 2.63 (drop +3.2%); holdout 2.64 -> 2.68 (rise +1.5%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 2.17 (drop +19.9%); holdout 2.64 -> 3.37 (rise +27.9%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-21

### W6: slot memory enables long-range fact recall (hippocampus)
- falsifier: with-memory acc >= 30% AND baseline acc < 10%
- result: baseline 0.000 vs with_memory 0.000
- artifact: dev_documentation/multimind/W6_results.json
- date: 2026-05-21

### W6: slot memory enables long-range fact recall (hippocampus)
- falsifier: with-memory acc >= 30% AND baseline acc < 10%
- result: baseline 1.000 vs with_memory 0.998
- artifact: dev_documentation/multimind/W6_results.json
- date: 2026-05-21

### W6b: slot memory cross-sample fact recall (hippocampus)
- falsifier: with-memory acc >= 25% AND baseline < 18% on question-only eval
- result: baseline 0.010 vs with_memory 0.160
- artifact: dev_documentation/multimind/W6b_results.json
- date: 2026-05-22

### W6b: slot memory cross-sample fact recall (hippocampus)
- falsifier: with-memory acc >= 25% AND baseline < 18% on question-only eval
- result: baseline 0.112 vs with_memory 0.066
- artifact: dev_documentation/multimind/W6b_results.json
- date: 2026-05-22

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.71 -> 2.18 (drop +19.7%); holdout 2.64 -> 3.34 (rise +26.7%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-22

### W5: sleep-cycle adapter updates on conversation buffer
- falsifier: convo ppl drops >= 20% AND holdout ppl rises < 5%
- result: convo ppl 2.72 -> 2.67 (drop +1.7%); holdout 2.64 -> 2.64 (rise +0.3%)
- artifact: dev_documentation/multimind/W5_results.json
- date: 2026-05-22
