# Coral Lab — Polyphase Distill-Merge experiment

A self-contained experiment. The entire experiment can be deleted by removing
this directory plus three references in [../../veritate_mri/web/index.html](../../veritate_mri/web/index.html)
(marked `DELETABLE-CORAL`) and two web files (`coral_lab.css`, `coral_lab.js`).

## What this is

Coral Merge blends two same-architecture transformer models trained on
disjoint corpora into a single same-shape model whose capability per
training FLOP should exceed a from-scratch baseline. Three steps:

1. **Align** — Hungarian solver on FFN intermediate-activation correlation.
2. **Splice** — Per-matrix learnable scalars (~88 floats for a 10-layer model).
3. **Distill-refine** — Dual-teacher KL plus CE on the mixed corpus.

Full spec: `~/Documents/GitHub/Agent-Documents/Veritate/coral_merge_spec.md`.

## The v1 experimental matrix

| Run                              | Size | Corpus                          | Role                |
| -------------------------------- | ---- | ------------------------------- | ------------------- |
| `coral_a_tinystories_30m`        | 30M  | tinystories                     | Constituent A       |
| `coral_b_distill_v1_30m`         | 30M  | distill_v1                      | Constituent B       |
| `coral_baseline_50m`             | 50M  | distill_v1_mix_tinystories      | Apples-to-apples    |
| `coral_blend_30m`                | 30M  | distill_v1_mix_tinystories      | Output of merge     |

Acceptance gate: `coral_blend_30m` val loss on the mixed corpus ≤ `coral_baseline_50m` val loss × 1.05 at equal eval steps.

## Workflow

### 1. Train Constituent A on tinystories

```bash
cd /Users/mintaka-01/Documents/GitHub/Veritate
.venv/bin/python tools/coral/run_coral.py \
  --name coral_a_tinystories_30m \
  --corpus tinystories \
  --size 30m \
  --total_steps 6000 \
  --batch 16 \
  --seq 256 \
  --description "Coral-A constituent on tinystories"
```

Throughput on Apple M1 16GB: roughly 30–60 minutes for 6000 steps. Monitor live
via the Coral Lab tab.

### 2. Train Constituent B on distill_v1

```bash
.venv/bin/python tools/coral/run_coral.py \
  --name coral_b_distill_v1_30m \
  --corpus distill_v1 \
  --size 30m \
  --total_steps 6000 \
  --batch 16 \
  --seq 256 \
  --description "Coral-B constituent on distill_v1"
```

Note: `distill_v1` is only 3.6 MB. 6000 × 16 × 256 = 25 MB of bytes consumed —
roughly 6.9 epochs over the corpus. Acceptable for this experiment; if
overfitting shows up early, drop `--total_steps` to 3000.

### 3. Train the Baseline 50M on the mixed corpus

```bash
.venv/bin/python tools/coral/run_coral.py \
  --name coral_baseline_50m \
  --corpus distill_v1_mix_tinystories \
  --size 50m \
  --total_steps 6000 \
  --batch 16 \
  --seq 256 \
  --description "Coral baseline 50M on the mixed corpus"
```

### 4. Run the merge

```bash
.venv/bin/python tools/coral/merge.py \
  --name_a   coral_a_tinystories_30m \
  --name_b   coral_b_distill_v1_30m \
  --out_name coral_blend_30m \
  --corpus   distill_v1_mix_tinystories \
  --refine_steps 1500 \
  --batch 16 \
  --seq 256
```

Stage 3a (scalar warmup) is `0.05 × 1500 = 75` steps. Stage 3b (full refine) is
the remaining 1425. Total compute for the merge is roughly 25% of one
constituent's training budget.

### 5. Compare in Coral Lab

Open the dashboard (default `http://localhost:8080`), switch to the
**Coral Lab** tab. Select runs in the three pickers:

- **Slot A** → `coral_a_tinystories_30m`
- **Slot B** → `coral_b_distill_v1_30m`
- **Baseline / Blend** → `coral_baseline_50m` first, then switch to `coral_blend_30m`

The combined chart shows train + val loss across all three slots. Compare the
blend's val loss to the baseline's val loss at the same eval step.

## Removal

To remove the entire experiment without touching anything else:

```bash
# 1. Delete the tools dir
rm -rf tools/coral/

# 2. Delete the standalone web module
rm veritate_mri/web/coral_lab.css veritate_mri/web/coral_lab.js

# 3. Remove the DELETABLE-CORAL marker blocks from index.html
#    (the link + script tags in <head>, the tab in <div class="tabs">,
#     and the tab body. Each block is bracketed by DELETABLE-CORAL comments.)
#    Also remove the "coral" entry from the `valid` array in
#    veritate_mri/web/index.js (marked DELETABLE-CORAL inline).

# 4. Optionally remove the trained runs
rm -rf models/coral_a_tinystories_30m models/coral_b_distill_v1_30m \
       models/coral_baseline_50m       models/coral_blend_30m

# 5. The 50m preset added to trainers/multimind_m3/plugin.py is harmless to
#    leave in place; remove only if a clean revert is required.
```

## Why this on Apple M1 16GB

The hardware constraint is real. A 30M bf16 model uses 60 MB resident; AdamW
state and activations push it to 600–800 MB. Two 30M models plus a 50M blend
fit in unified memory with room to spare. Training throughput on M1 MPS is
roughly 30–60k tokens/s for a 30M model, which puts a 6000-step run at 30–60
minutes. Same-day iteration is feasible; that's the point of starting at this
scale before pushing the algorithm to the 250M+ regime.
