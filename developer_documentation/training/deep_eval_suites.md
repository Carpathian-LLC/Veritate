# deep eval suites

## What it is

The three benchmark suites a checkpoint is scored against on demand: MMLU, HellaSwag, and IFEval. Suite code lives in [veritate_mri/eval/](../../veritate_mri/eval/), the runner is `run_suites_on_model()` in [run_eval.py](../../veritate_mri/eval/run_eval.py), and the routes are `GET`/`POST /run/<name>/eval_deep` plus `/eval_deep/status` in [runs_routes.py](../architecture/backend/routes.md). Distinct from the per-checkpoint grade sets in [eval_sets.md](eval_sets.md), which score every save automatically.

## How it works

- `EVAL_SUITES = ("mmlu", "hellaswag", "ifeval")`. The POST body takes `suite` as one name, a comma list, a list, or `all`.
- MMLU and HellaSwag are multiple choice: scored by comparing per-option likelihood, no generation.
- IFEval is generative. Each item pairs a prompt with verifiable `rules`; the model generates, and a deterministic checker in the `CHECKERS` registry grades pass/fail. Item pass requires every rule to pass.
- `CHECKERS` keys the rule `name` to a function taking `(response, **rule_kwargs) -> bool`: `json`, `sentence_count`, `forbidden_letter`, `word_count`, `item_count`, `contains`, `starts_with`, `forbidden_words`. An unrecognized rule name scores `no_checker` and fails the item, so a typo sinks the score rather than passing silently.
- Items live in `veritate_mri/data/eval/samples/` as `{items: [{prompt, rules: [{name, ...kwargs}]}]}`. `SAMPLE_SETS` names them; the POST body picks one with `ifeval_set`, resolved by `data_path_for()`. An unknown name is a 400.
- The suite reports `pass_rate`, `accuracy` (mirrors it so the dashboard can rank every suite on one key), `by_rule` pass rates, and a truncated per-item transcript.

## Item sets: form versus correctness

Two sets ship, and they measure different things. Report them separately.

| `ifeval_set` | file | what every rule grades |
| --- | --- | --- |
| `default` | `ifeval_sample.json` | mixed: format rules plus rules that also require the right answer |
| `form` | `ifeval_form.json` | obedience only, verifiable without knowing the answer |

A rule that pins the correct answer (`contains` with the expected value, `starts_with` with the
correct yes or no) grades reasoning, not instruction following. In the `default` set 47% of rules
do, so a model whose instruction obedience improves sharply can score flat: it now obeys the shape
and still gets the fact wrong. Use `form` to measure whether training changed obedience, and the
multiple-choice suites to measure whether it changed knowledge.

`starts_with_yes_or_no` exists for this: it passes on either answer, so it grades the leading-token
format the instruction demanded without encoding which answer is true.

## Chat framing

`run_ifeval(..., chat=True)` wraps each prompt in `CHAT_TEMPLATE` (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`) and passes `CHAT_STOP` to the decode loop. The route resolves the flag from the model's own capability block: `chat.status == "trained"` turns it on, and an explicit `ifeval_chat` in the request body overrides. Base models keep the raw prompt.

Both halves are load bearing:

- A chat-trained model fed a bare instruction answers a framing it was not trained on and free-associates, so the score measures framing rather than instruction following.
- `CHAT_STOP` is `<|im_end|` without its closing bracket, matching the serving path's rule that a byte model reproduces a marker approximately. `_generate` trims it off. Without a stop, generation runs past the turn end into a self-conversation, and every length-based checker (`sentence_count`, `word_count`, `item_count`) grades that trailing text too.

## Dependencies

- `readers.paths.EVAL_SAMPLES_ROOT` for the sample sets.
- `readers.capabilities.read()` for the chat-framing default.
- `NON_LANGUAGE_TYPES` from [save.py](../../veritate_mri/training/save.py): a `statistical` or `other` model is refused, since these suites score nonsense on non-text corpora.

## Pitfalls

- The shipped IFEval set is hand written, not the Google benchmark. Scores are comparable across this repo's checkpoints, not against published IFEval numbers.
- Eval items must stay out of training corpora. An instruction genre that authors the same prompts an eval asks turns the benchmark into a memorization check; verify a corpus against the item list before trusting a delta.
- An SFT that installs instruction obedience can leave the `default` score unmoved while form compliance climbs. Read both sets before concluding a run did nothing.
- `item_count` splits on newlines first and falls back to commas, semicolons, and ` and `. A reply that packs items into prose is graded on that split, not on meaning.
- IFEval generates per item, so wall clock scales with `ifeval_max_new` times item count; it dominates a combined run.
