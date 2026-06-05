# Coral Lab — Polyphase Distill-Merge

Standalone, deletable experiment tab. Trains two same-shape byte-level
transformers on disjoint corpora, then blends them into a single same-shape
model via alignment + scalar splice + dual-teacher distill. Compares the
blend against a from-scratch baseline trained on the mixed corpus.

The full algorithm spec lives at
`~/Documents/GitHub/Agent-Documents/Veritate/coral_merge_spec.md`.

## Why "coral"

Codename for the experiment, not a technical acronym. Coral colonies are made
of many small individuals fused into a larger living structure — the same
shape as fusing several small trained models into one effective larger model.

The technical name is **Polyphase Distill-Merge** (PDM). Coral is the
project handle; PDM is the algorithm.

## Where things live

| Path                                                 | What                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `tools/coral/run_coral.py`                           | Vanilla byte-level base-model trainer (no QAT, no adapter) |
| `tools/coral/merge.py`                               | The Coral Merge algorithm: align + splice + distill   |
| `tools/coral/run_b.sh`, `run_baseline.sh`, `run_merge.sh` | One-line launchers                              |
| `tools/coral/README.md`                              | Workflow + removal instructions                       |
| `veritate_mri/web/coral_lab.{css,js}`                | Standalone three-column dashboard                     |
| `veritate_mri/web/index.html`                        | Four `DELETABLE-CORAL` blocks (link, script, tab, body) |
| `veritate_mri/web/index.js`                          | One `DELETABLE-CORAL` entry in the `valid` tab array  |
| `trainers/multimind_m3/plugin.py`                    | One added `"50m"` SIZE_PRESETS entry (harmless if kept) |

## Metrics contract

The trainer writes the canonical `train.csv` per the existing contract at
`veritate_mri/training/save.py:38`:

```
step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed
```

The Coral Lab UI consumes this via the existing `/run/<name>/csv` route
(`veritate_mri/routes/runs_routes.py`). No new Flask routes were added.

## The 30M shape

The constituents use the existing `multimind_m3` preset shape but trained via
the vanilla trainer (no Hebbian adapter):

```
hidden=512, layers=10, ffn=2048, heads=8, seq=256, vocab=256  -->  ~31.7M params
```

The baseline uses a newly-added 50M preset (also added to `multimind_m3` for
consistency):

```
hidden=640, layers=10, ffn=2560, heads=10, seq=256, vocab=256  -->  ~50M params
```

## Workflow

1. Train constituent A on `tinystories` (~95 min on M1 MPS, 6000 steps).
2. Train constituent B on `distill_v1` (~95 min).
3. Train the 50M baseline on `distill_v1_mix_tinystories` (~115 min).
4. Run `merge.py` to produce `coral_blend_30m` (~15 min).
5. In Coral Lab, compare slots A, B, and toggle CMP between baseline and blend.

Acceptance gate: `coral_blend_30m` val loss on the mixed corpus
≤ `coral_baseline_50m` val loss × 1.05 at equal eval steps.

## Algorithm in 30 seconds

1. **Align** — Forward-hook `ff.down` on both models with 2048 mixed-corpus
   samples to collect FFN intermediate activations. Compute the cross-correlation
   matrix per layer. Solve the linear assignment problem (Hungarian via SciPy,
   or greedy fallback). Permute model B's `ff.up` rows and `ff.down` cols —
   structural symmetry, output unchanged but FFN bases now aligned.

2. **Splice** — Replace each weight matrix M with
   `M_blend = α_M * M_A + β_M * M_B_aligned`. The `(α, β)` pairs are scalar
   `nn.Parameter`s initialized to 0.5. ~88 floats of overhead for a 10-layer
   model.

3. **Distill-refine** — Two phases on the mixed corpus:
   - **3a** (5% of budget): freeze weights, train scalars only. Lets the
     splice coefficients find a stable equilibrium.
   - **3b** (95% of budget): unfreeze weights. Both originals serve as
     frozen teachers; loss is `(1-λ)·CE + λ_A·KL(s||T_A) + λ_B·KL(s||T_B)`
     at temperature 2.0, with `λ_A = λ_B = 0.25`.

## Removal

Three steps, fully reversible — see `tools/coral/README.md` for the exact rm
commands. The 50m preset added to `trainers/multimind_m3/plugin.py` is harmless
to leave in place.

## Compatibility notes

The package shim at `veritate/__init__.py` and `veritate/plugin/__init__.py`
is unrelated to Coral but was discovered/fixed in the same session: existing
trainers used a stale `from veritate.X import` path that resolves to nothing
in the current layout. The shim re-exports `veritate_core.model`,
`veritate_core.qat`, `training.save`, and `readers.paths` under the legacy
names. Removing the shim breaks `distill_teacher`, `multimind_m1`,
`multimind_m3`, `multimind_mega`, and `example_plugin`.
