# ------------------------------------------------------------------------------------
# veritate_mri/eval/__init__.py
# ------------------------------------------------------------------------------------
# Byte-level eval harness, in-tree copy that the MRI dashboard drives at runtime.
#
# This is the *production* path (the dashboard's "deep eval" panel imports from here).
# The development sandbox lives at experiments/v2/eval_harness/ and stays untouched —
# changes flow from there into this directory once they're ready to ship.
#
# Standard eval harnesses (lm-eval-harness, Eleuther's) assume a tokenized model and
# pick the multiple-choice answer by argmax over per-token log-likelihoods. Veritate
# is byte-level (vocab=256), so the data is unchanged and only the scoring unit
# (token -> byte) adapts.
#
# Suites:
#   - mmlu.py       : 4-way multiple choice across academic subjects
#   - hellaswag.py  : 4-way sentence-completion
#   - ifeval.py     : instruction-following rule-checker scaffold
#
# Programmatic entry: run_eval.run_eval_suites(brain_or_model, suites=[...], ...)
# CLI entry:          python -m veritate_mri.eval.run_eval --ckpt ... --suite ...
# ------------------------------------------------------------------------------------

from .score import score_sequence  # noqa: F401
