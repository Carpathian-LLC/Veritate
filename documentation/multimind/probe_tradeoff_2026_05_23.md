# Probe accuracy vs MtM signal strength: a real trade-off

Discovered 2026-05-23 by autonomously sweeping three probe variants paired with the same MtM kitchen-sink config.

## Data

| Probe | val acc | training | MtM ‖g‖_2 | MtM kl_max |
|---|---|---|---|---|
| original   | 86.09% | BCE, binary {0,1}, 200k params, 2000 steps | 0.545 | 0.331 |
| big        | 88.38% | BCE, binary {0,1}, 1.6M params, 6000 steps | 0.171 | 0.008 |
| continuous | 85.62% | MSE on tanh, targets {-1,+1}, 200k params, 4000 steps | 0.541 | 0.006 |

## The finding

A higher-accuracy probe produces a WEAKER MtM routing signal. Counter-intuitive but explainable:

- original (86%): BCE-trained, always commits to ±0.99 even on ambiguous inputs. Every training sample gives the MoE gate a STRONG sentiment scalar; ‖g‖ has lots of signal. Routing-KL high.
- big (88%): more capacity + better-calibrated. Outputs lower-magnitude on uncertain inputs. Bias term `g_e * s` weaker; gate has less reason to specialize.
- continuous (85.6%): MSE on tanh deliberately lets outputs land near 0. Same mechanism: weak scalar → weak gate-bias signal.

## Why this matters

The "routing KL ratio" metric REWARDS probes that commit (right or wrong). A probe that says +0.99 on everything wins the routing-KL game even though it's badly miscalibrated.

For real applications, the user picks based on goal:
- Vibrant dashboard demo: original probe (strong contrast between pos/neg routing).
- Honest behavior on neutral content: continuous probe (scalar near 0 when input is ambiguous).
- Best benchmark accuracy: big probe (88.4%).

## Implication

The brain-inspired "amygdala fires only when something happens" intuition ALIGNS with the calibrated-probe behavior: the bias only fires when the probe is confident. The kitchen-sink with original probe is showing a less brain-like overcommitment.

For the brain-region story, the continuous probe is the right default even though its visible routing-KL is smaller. Three model checkpoints are available so the user picks based on goal.

## Decisions for tomorrow

1. Default model: keep binary (current) or switch to continuous? Recommend keep binary for visual clarity in dashboard demos; offer continuous as alternative.
2. Probe deployment: ship binary with default model; ship continuous as switchable.
3. Future: a 3-class probe (positive/neutral/negative) trained on SST-5 would resolve the trade-off cleanly.
