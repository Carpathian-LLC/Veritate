# The classroom dashboard — vision + moonshots

The Live Training tab today shows loss + lr + tok/s. That tells us THAT the
model is learning, not WHAT or HOW. The vision: a metaphorical classroom
where every neuron, every confidence component, every emerging skill is
visible while the model trains.

This composes with [`MULTIMODULE_BRAIN.md`](../MULTIMODULE_BRAIN.md) (7-module
architecture) and [`CONFIDENCE_MATH.md`](../CONFIDENCE_MATH.md) (calibrated
confidence) — both of those docs say "we can see HOW it learns." This doc
operationalizes that visibility.

## What "HOW it learns" actually means

The model isn't a black box that gets better. It's a 12-layer stack of
neurons that develop specializations, form coalitions, and cross thresholds
where new capabilities emerge. The classroom panels make those processes
observable:

| Concept             | What you actually see                                      |
| ------------------- | ---------------------------------------------------------- |
| Neuron growth       | Top-K neurons per layer per checkpoint, change in their activations over time |
| Memory formation    | Which positions in attention heads stabilize early vs late |
| Confidence ramp     | margin/entropy/lens-consistency evolving over training     |
| Skill emergence     | Breakpoints where val perplexity drops on a *category*     |
| Capacity saturation | How much representational space is "used" per layer        |
| Learning rate       | Per-neuron weight change magnitude per checkpoint          |

## Panels to add (shippable, in order)

### Tier 1 — data already exists

1. **Model size meter.** ✓ shipped 2026-04-29 — Reads `config.json` for the
   selected run. Shows: total params (with embed/attn/ffn breakdown), INT8
   bytes (params × 1), INT4 bytes (params × 0.5), and an L3 fit indicator
   (green ≤ 96 MB, red otherwise on 9800X3D). One stable render per run pick.
   ✓ mirrored to Learning tab 2026-04-29 — `#learnSizeMeter` rebinds to the
   timeline picker's value.

2. **Neuron biography.** ✓ shipped 2026-04-29 — Reads
   `probe_step_*.json` per probed checkpoint. Layers × checkpoints grid
   showing top-8 FFN neurons (id + magnitude). Click a cell to see that
   neuron's rank + activation across every probed checkpoint. Falls back to
   "no probe data for this run yet — needs Stage D or retroactive probing"
   when the run pre-dates the rule-4 dump or no probes exist on disk.
   ✓ mirrored to Learning tab 2026-04-29 — `#learnNeuronBio` re-renders on
   timeline pick; click-to-inspect detail box scoped per panel.

3. **Confidence evolution.** ✓ shipped 2026-04-29 — Reads `lens_step_*.npz`
   per checkpoint, computes the four CONFIDENCE_MATH components in the
   browser (no server route), plots them all on one canvas. Margin
   normalized to the run's max-abs for co-plotting; lens-consistency in
   green so the climb is obvious. `residual_stab` is a proxy from
   `residual_norms` until the probe writes the embed-projection signal.
   ✓ mirrored to Learning tab 2026-04-29 — `#cConfEvoL` plots the same four
   series for the picked timeline.

4. **Lens-logit drift.** ✓ shipped 2026-04-29 — Reads `lens_step_*.npz`
   per checkpoint. Per-layer × per-checkpoint table; each cell is the
   top-3 predicted bytes (glyph + softmax %) at that layer's residual
   stream projected through the tied embedding. Watch the picks sharpen
   and stabilize as training progresses.
   ✓ mirrored to Learning tab 2026-04-29 — `#learnLensDrift` re-renders for
   the picked timeline.

5. **Loss heat strip.** Per-byte loss across the val set, color-coded.
   Shows where the model is confused — is loss mostly on punctuation?
   On capitalization? On the third byte after a period?

### Tier 2 — small infra additions

6. **Co-activation graph.** At each checkpoint, compute pairwise
   correlations of top-K FFN neurons across the val set. Visualize as a
   force-directed graph. Watch neurons cluster into circuits. Each circuit
   = a learned subroutine.

7. **Per-neuron learning rate.** Diff weight L2 norm across consecutive
   checkpoints. "Frozen" neurons (no change in last 5 ckpts) vs "alive"
   neurons (still moving). Plot the alive fraction over time — it shrinks as
   the network commits.

8. **Surprise atlas.** Run a fixed paragraph through the model at every
   checkpoint, capture per-byte surprise (negative log probability). Show
   as a 2D heatmap (byte index × checkpoint). Watch surprise gradients
   flatten over training.

### Tier 3 — research-grade

9. **Skill emergence detector.** Maintain a benchmark suite of
   per-capability prompts (vocab category, syntax, simple Q&A, basic logic).
   At every checkpoint, evaluate val perplexity per category. Detect SHARP
   drops — those are skill emergence events. Notify on detection.

10. **Memory saturation per layer.** Hyperdimensional Computing fingerprint:
    project residual stream into 10K-dim binary vectors, measure how full
    the representational space is per layer. Layers with high saturation
    are "done learning"; layers with low saturation have room.

11. **Critical period identification.** At step N, snapshot weights. Re-run
    training on a tiny perturbation of the corpus. Measure how much the
    final model differs. Sharp differences = critical period (data choice
    has outsized impact at this step). Mark these on the loss curve.

## Moonshot ideas (truly unprecedented)

These don't exist anywhere. Each would be a paper.

### M1 — Neuron transcripts

For every FFN neuron, generate a natural-language description of what it
detects, automatically. Run through the corpus, find the top-100 contexts
where the neuron fires, ask a separate small model to summarize the pattern.
Display as: "Neuron 1427: fires on byte sequences ending a sentence with a
period followed by a space. Specialized starting at step 5000."

The Anthropic interpretability team does this offline on closed models
billions of params big. Doing it live during training of an 80M open model,
shown in the MRI per checkpoint — nobody has shipped this.

### M2 — Live ablation theater

Click a neuron in the dashboard, hit "ablate", see the model's output change
in the response area in real time. Watch a story about a girl turn into a
story about a boy because you knocked out the gendered-pronoun neuron.
Educational and visceral.

### M3 — Emergence ringing

Train two models on identical data with different seeds. At every
checkpoint, compute symmetric KL between their predictions on val. When KL
collapses (they start producing the same predictions), that's an emergence
event — the architecture's inductive bias has overpowered seed-dependent
randomness. Plot KL over training. Spikes downward = emergence moments.

### M4 — Learning trajectory replay

Every checkpoint stores a hash of the residual stream activations at the
canonical prompt. Build an interactive 3D embedding (UMAP or t-SNE) of all
checkpoints. The model's "learning trajectory" through hidden state space
becomes a visible path. Watch the model wander through ideas, settle into
basins, escape, settle again.

### M5 — The forgetting curve

For each curriculum stage transition, measure how much the model's prior
abilities decay. Plot per-stage retention of stage-A skills, stage-B
skills, etc. The forgetting curve. Standard in cognitive psychology
(Ebbinghaus 1885), never measured live during ML training. Direct
comparison to human memory consolidation.

✓ first analysis tool shipped 2026-04-29; live dashboard panel pending.
`analysis/forgetting_curve.py` walks `data/models/*curriculum*/grades_step_*.json`,
pairs consecutive stages, and emits `forgetting_curve.json` + a heatmap
PNG. forgetting_pct = (ppl_start_next - ppl_end_prev) / ppl_end_prev × 100.
First B → C run shows broad cross-stage transfer (negative pct on every
band except k); middle-grade -28% confirms the observation that Stage C
sharpens chapter-book reading. Live MRI panel still TODO.

### M6 — Concept-formation timestamps

For a fixed list of ~50 concepts (animals, colors, emotions, simple verbs),
test the model's representation quality per checkpoint via probing
classifiers. The first checkpoint where each concept's probe accuracy
crosses 80% = its "formation timestamp." Visualize as a Gantt chart.
Concepts have birthdays. Watch the model develop a vocabulary in real time.

✓ first analysis tool shipped 2026-04-29; live dashboard panel pending.
`analysis/concept_gantt.py` walks `concepts_step_*.json` per model dir
(or `--combined` across curriculum stages chronologically), computes
formation timestamps at a configurable surprise-bits threshold (default
2.5), and writes `analysis/concept_gantt.{json,png}` plus a console
earliest/latest summary.

### M7 — The architectural mirror

Two models — one transformer, one Mamba-2 — train on the same data with
the same loss target. Side-by-side classroom dashboards. Watch which
architecture forms which capabilities first, where they diverge. Comparative
cognition for neural architectures.

### M8 — Curriculum what-ifs

Branch the model from any checkpoint, train two paths on different
sub-corpora, see which produces better generalization. Render as a
decision tree of training trajectories. The user becomes a parent choosing
which lessons to give the model.

## Wiring plan

**Now (this turn):** ship Tier 1 panels 1-4 (model size meter, neuron
biography, confidence evolution, lens-logit drift). Spawn agent. Probe
dumps already land per Rule 4 starting from Stage D — backwards-compatible
with existing `lens_step_*.npz` from older runs (none exist yet, so the UI
shows "no data" gracefully).

**Sprint 1 (1 week):** Tier 2 (co-activation, learning rate, surprise atlas).
Needs a small per-checkpoint extension to `training/checkpoint_probe.py`.

**Sprint 2 (2-3 weeks):** Tier 3 (skill emergence, capacity saturation,
critical periods). Needs per-capability benchmark suite + HDC fingerprinting.

**Moonshots:** each one is its own multi-week project. M1 (neuron
transcripts) is highest-leverage — it makes the rest of the panels human-
readable. M5 (forgetting curve) is highest-novelty — directly comparable
to human cognitive psych research.

## How this composes with the rest

- **MULTIMODULE_BRAIN**: each new module gets its own classroom panel
  showing how it activates over training. The dashboard scales as the
  architecture grows.
- **CONFIDENCE_MATH**: the four components are already shippable
  visualizations (Tier 1 panel #3).
- **CURRICULUM_PLAN**: each stage transition triggers a forgetting-curve
  measurement. Stage gates become measurable.
- **REALITY MONITOR**: when wired (after Stage F), classroom shows the
  factuality direction emerging, with deception neurons appearing as a
  cluster in the co-activation graph.

The classroom dashboard is the project's core differentiator. Closed-source
labs run interpretability tools offline on models billions of params big.
We get the same visibility live, on an 80M glass model, as a child learns
to read.
