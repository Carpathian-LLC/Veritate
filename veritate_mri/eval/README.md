# veritate_mri/eval — byte-level deep eval (MMLU / HellaSwag / IFEval)

The dashboard-facing copy of the byte-level eval harness. Development sandbox lives
at `experiments/v2/eval_harness/`; this directory is the runtime path the MRI
"deep eval" panel calls into.

## What "deep eval" means here

Standard LLM benchmarks (MMLU, HellaSwag, IFEval) assume a tokenized model and
pick the multiple-choice answer by argmax over per-token log-likelihoods.
Veritate is byte-level (vocab=256), so the *data* is unchanged and only the
scoring unit (token -> byte) adapts. The metric is the same `acc_norm` the
standard `lm-eval-harness` reports.

Core scorer (`score.py`):

```python
score_sequence(model, prompt_bytes, completion_bytes) -> float   # nats/byte
```

Per-byte log-likelihood of `completion_bytes` conditioned on `prompt_bytes`.
Length-normalized so longer candidates aren't penalized. Floor is
`-ln(256) ≈ -5.545` nats/byte (uniform random byte distribution).

## Files

| File | What |
| --- | --- |
| `score.py` | Core scorer: `score_sequence` |
| `mmlu.py` | 4-way multiple choice across 57 academic subjects |
| `hellaswag.py` | 4-way sentence-completion (commonsense) |
| `ifeval.py` | Instruction-following rule-checker scaffold (3 example rules) |
| `run_eval.py` | CLI + programmatic entry. The dashboard imports `run_suites_on_model` |
| `_smoke.py` | <30s CPU smoke test |
| `data/*.json` | ~20-item sample data for the smoke; real benchmarks must be downloaded |

## Two ways to invoke

### 1. From the MRI dashboard (recommended)

The Learning tab's **deep eval** panel exposes:
- A checkpoint picker (defaults to latest).
- Suite checkboxes (MMLU / HellaSwag / IFEval).
- A **run deep eval** button.

Clicking the button POSTs to `/run/<name>/eval_deep` with the chosen suite list.
The server loads the model through the same `Brain` backend the dashboard
already uses for inference, runs the suites synchronously, writes the JSON
report to `models/<name>/eval_deep/<suite>_step_<N>.json`, and returns the
result. Cached results are listed by `GET /run/<name>/eval_deep`.

> **MPS warning.** The 800M training is on MPS. Deep eval will briefly share
> the MPS device with training when run there. The endpoint is **user-triggered
> only** — there is no periodic timer. For runs that need to leave training
> undisturbed, set the dashboard's PyTorch backend to load the brain on CPU.

### 2. CLI

```bash
cd /Users/mirach-00-usc1/Development/Veritate
python -m veritate_mri.eval.run_eval \
  --ckpt models/<name>/checkpoint_step_<N>.pt \
  --suite mmlu,hellaswag,ifeval \
  --device cpu \
  --output report.json
```

The CLI auto-detects the architecture from the state dict (canonical Veritate,
RoPE-only 85M, or 800M with MTP head) using the same dispatcher as
`backends/pytorch.py::Brain`. The MTP head is loaded but unused; single-byte
forward is enough to score.

## Expected wall time per suite

Sample data (the ~20 items shipped here):

| Suite | Items | CPU runtime (M3) |
| --- | --- | --- |
| MMLU sample | 20 | < 1 min |
| HellaSwag sample | 2 | < 10 s |
| IFEval sample | 3 | 30-90 s (generation, not scoring) |

Real benchmarks (download instructions below):

| Suite | Items | Est. CPU at 85M | Est. MPS at 85M | Est. MPS at 800M |
| --- | --- | --- | --- | --- |
| MMLU full | ~14k | several hours | ~20-40 min | ~3-6 hours |
| HellaSwag val | ~10k | several hours | ~30 min | ~5 hours |
| IFEval | ~541 | 1-2 hours (gen-bound) | 20-30 min | 3-4 hours |

For a typical "did the model learn anything?" check, **cap each suite to a few
hundred items** (`--limit 200` on the CLI, or the dashboard's default once
configured) — that's enough to lift accuracy off the ±3% noise floor without
spending hours.

## How to read the numbers

**Random baselines.**

| Suite | Chance accuracy |
| --- | --- |
| MMLU | **0.25** (4-way MCQ) |
| HellaSwag | **0.25** (4-way MCQ) |
| IFEval | depends on rules — the 3-rule sample scaffold lands near 0.0 for random text |

**What "good" looks like for an 800M byte-level model.**

The published GPT-3 175B numbers are: MMLU ~43% (5-shot), HellaSwag ~78%,
IFEval (post-Instruct) ~50-60%. An 800M *byte-level* model trained on a small
mix won't match those. Realistic expectations:

| Range | Reading | Action |
| --- | --- | --- |
| At chance (0.25) | Model has no reliable signal for this suite | Keep training; corpus mix may need richer text |
| 0.25-0.30 | Faint above-chance signal | Encouraging — keep going, eval again at next save |
| 0.30-0.40 | Honest signal | This is where a well-tuned 800M byte model lands on MMLU |
| 0.40-0.50 | Strong | Ahead of typical small models. Look at per-subject for hot spots. |
| 0.50+ | Above the green tile threshold | Real competence on the suite |

For HellaSwag, byte-level models tend to underperform tokenized models because
the model has to predict every byte of the ending verbatim; a 4-byte
difference between candidates already shifts the score budget noticeably.
Cross-reference with MMLU's letter-mode (cheaper, less semantic) to sanity-check.

**The dashboard color-codes accuracy:**

- `> 0.50` — green tint (above chance for 4-way MCQ by a comfortable margin)
- `0.25 - 0.50` — warning (signal exists but isn't reliable)
- `< 0.25` — red (below chance; the harness or the model is broken)

## Downloading the real benchmarks

The sample files are placeholders only. **Do not** report numbers from
`mmlu_sample.json` as MMLU scores.

### MMLU

```python
from datasets import load_dataset
import json
ds = load_dataset("cais/mmlu", "all", split="test")
out = {"questions": [
    {"subject": r["subject"], "question": r["question"],
     "choices": r["choices"], "answer": r["answer"]}
    for r in ds
]}
json.dump(out, open("mmlu_full.json", "w"))
```

Pass `--mmlu-data mmlu_full.json` to the CLI, or drop the file in
`veritate_mri/eval/data/mmlu_full.json` (the dashboard endpoint accepts an
optional `data` field in the POST body to override).

### HellaSwag

```python
ds = load_dataset("Rowan/hellaswag", split="validation")
out = {"items": [
    {"ctx": r["ctx"], "endings": r["endings"],
     "label": int(r["label"]), "activity": r.get("activity_label")}
    for r in ds
]}
json.dump(out, open("hellaswag_val.json", "w"))
```

### IFEval

Source: <https://huggingface.co/datasets/google/IFEval> +
<https://github.com/google-research/google-research/tree/master/instruction_following_eval>

The Google IFEval ruleset is ~25 rule families with deterministic
`str -> bool` checkers. The scaffold here implements 3 (`json`,
`sentence_count`, `forbidden_letter`) and provides a `CHECKERS` registry to
plug the rest into. To port:

1. Read each rule family's reference checker from
   `instruction_following_eval/instructions.py` in the Google repo.
2. Rewrite as a `str -> bool` function and add it to `CHECKERS` in `ifeval.py`.
3. Convert the official IFEval JSONL into this schema:
   ```json
   {"items": [{"prompt": "...",
               "rules": [{"name": "keywords_existence", "keywords": ["..."]}, ...]}]}
   ```

## Smoke test

```bash
cd /Users/mirach-00-usc1/Development/Veritate
python -m veritate_mri.eval._smoke
```

A random-init tiny model produces:
- `score_sequence` near the uniform floor (`-5.55` nats/byte)
- MMLU accuracy near `0.25` (chance for a 4-way MCQ)
- HellaSwag accuracy in `{0, 0.5, 1.0}` for the 2-item smoke

If those land in the expected bands, the framework is unbiased.

## Where this lives relative to the existing "reading level" tile

The Learning tab still has the **reading level** tile (per-band perplexity on
hand-authored grade-labeled corpora). That tile is fast, runs at every
checkpoint via the save hook, and is a *fluency* proxy — it tells you the
model's per-byte cross-entropy on Pre-K through PhD prose, not whether the
model can answer questions. The footnote on that tile now points here.

**Deep eval** runs only on demand, takes minutes, and reports real accuracy.
They are complementary, not redundant.

## Why this matters

Without this, "the model is getting better" reduces to val-NLL going down on
FineWeb-Edu — which tells you the loss is dropping, not that the model can
answer questions. Byte-level MCQ scoring closes that gap with the same metric
on the same data the rest of the field reports.
