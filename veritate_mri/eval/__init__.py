# ------------------------------------------------------------------------------------
# veritate_mri/eval/__init__.py
# ------------------------------------------------------------------------------------
# Byte-level eval harness the MRI dashboard drives at runtime (the "deep eval" panel).
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

from .score import score_sequence
