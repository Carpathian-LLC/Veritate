# veritate_mri/eval

Byte-level deep-eval harness. Runtime path the MRI "deep eval" panel calls.
Development sandbox lives at `experiments/v2/eval_harness/`.

## scoring

Standard LLM benchmarks (MMLU, HellaSwag, IFEval) assume a tokenized model and
pick the multiple-choice answer by argmax over per-token log-likelihoods.
Veritate is byte-level (vocab=256); the data is unchanged, only the scoring
unit (token -> byte) adapts. Reported metric matches `lm-eval-harness`
`acc_norm`.

Core scorer (`score.py`):

```python
score_sequence(model, prompt_bytes, completion_bytes) -> float   # nats/byte
```

Per-byte log-likelihood of `completion_bytes` conditioned on `prompt_bytes`.
Length-normalized. Floor is `-ln(256) approx -5.545` nats/byte (uniform random
bytes).

## files

| file | what |
| --- | --- |
| `score.py`      | core scorer: `score_sequence` |
| `mmlu.py`       | 4-way MCQ across 57 academic subjects |
| `hellaswag.py`  | 4-way sentence-completion |
| `ifeval.py`     | instruction-following rule-checker scaffold (3 rules) |
| `run_eval.py`   | CLI + programmatic entry. Dashboard imports `run_suites_on_model` |
| `_smoke.py`     | <30s CPU smoke |
| `data/*.json`   | ~20-item sample data for the smoke; real benchmarks downloaded separately |

## invocation

### dashboard

The Learning tab's deep-eval panel exposes a checkpoint picker (defaults to
latest), suite checkboxes (MMLU / HellaSwag / IFEval), and a run button.

POSTs to `/run/<name>/eval_deep` with the suite list. The server loads the
model through the same `Brain` backend the dashboard uses for inference, runs
the suites synchronously, writes the JSON report to
`models/<name>/eval_deep/<suite>_step_<N>.json`, returns the result. Cached
results enumerate at `GET /run/<name>/eval_deep`.

Endpoint is invoked on demand; no periodic timer. On MPS, deep eval shares the
device with an active training run; for runs that must not share the device,
set the dashboard's PyTorch backend to CPU before pressing run.

### cli

```bash
python -m veritate_mri.eval.run_eval \
  --ckpt models/<name>/checkpoint_step_<N>.pt \
  --suite mmlu,hellaswag,ifeval \
  --device cpu \
  --output report.json
```

The CLI loads the model via `veritate_core.load.load_from_state_dict`. MTP
heads load but go unused; single-byte forward is sufficient for scoring.

## wall time

Sample data (~20 items shipped here):

| suite           | items | CPU runtime (M3) |
| --- | --- | --- |
| MMLU sample     | 20  | < 1 min |
| HellaSwag sample| 2   | < 10 s |
| IFEval sample   | 3   | 30-90 s (generation, not scoring) |

Real benchmarks (download instructions below):

| suite           | items | CPU 85M | MPS 85M | MPS 800M |
| --- | --- | --- | --- | --- |
| MMLU full       | ~14k | hours | 20-40 min | 3-6 hours |
| HellaSwag val   | ~10k | hours | ~30 min | ~5 hours |
| IFEval          | ~541 | 1-2 hours (gen-bound) | 20-30 min | 3-4 hours |

`--limit 200` per suite is enough to clear the +/- 3% noise floor on most runs.

## reading the numbers

Random baselines:

| suite       | chance |
| --- | --- |
| MMLU        | 0.25 (4-way MCQ) |
| HellaSwag   | 0.25 (4-way MCQ) |
| IFEval      | near 0.0 for random text on the 3-rule sample scaffold |

Reference for an 800M byte-level model:

| range        | reading | next |
| --- | --- | --- |
| 0.25         | no reliable signal | continue training; consider richer corpus mix |
| 0.25 - 0.30  | faint above-chance | reeval at next checkpoint |
| 0.30 - 0.40  | honest signal | typical band for tuned 800M byte models on MMLU |
| 0.40 - 0.50  | strong | inspect per-subject for hot spots |
| 0.50+        | above the green-tile threshold | real competence on the suite |

HellaSwag underperforms tokenized scoring at byte level: the model must
predict every byte of the ending verbatim, and a 4-byte length delta between
candidates already shifts the score budget. Cross-reference with MMLU
letter-mode (cheaper, less semantic) as a sanity check.

Dashboard color encoding:

- `> 0.50` green
- `0.25 - 0.50` warning
- `< 0.25` red (below chance; harness or model is broken)

## downloading the real benchmarks

Sample files are placeholders. Numbers from `mmlu_sample.json` are not MMLU
scores.

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

Pass `--mmlu-data mmlu_full.json` to the CLI, or drop the file at
`veritate_mri/data/eval/samples/mmlu_full.json` (the dashboard endpoint accepts
an optional `data` field in the POST body to override).

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
`sentence_count`, `forbidden_letter`) and exposes a `CHECKERS` registry. To
port a new rule:

1. Read its reference checker from `instruction_following_eval/instructions.py`.
2. Rewrite as `str -> bool` and add to `CHECKERS` in `ifeval.py`.
3. Convert the official IFEval JSONL into:
   ```json
   {"items": [{"prompt": "...",
               "rules": [{"name": "keywords_existence", "keywords": ["..."]}, ...]}]}
   ```

## smoke test

```bash
python -m veritate_mri.eval._smoke
```

A random-init tiny model produces:

- `score_sequence` near the uniform floor (`-5.55` nats/byte)
- MMLU accuracy near `0.25`
- HellaSwag accuracy in `{0, 0.5, 1.0}` for the 2-item smoke

Those bands indicate an unbiased framework.

## relation to the reading-level tile

The Learning tab's reading-level tile reports per-band perplexity on
grade-labeled corpora. That tile runs at every checkpoint via the save hook
and reports a fluency proxy (per-byte cross-entropy on Pre-K through PhD
prose), not question-answering accuracy. Deep eval runs on demand, takes
minutes, and reports real accuracy. Different signals; both are kept.
