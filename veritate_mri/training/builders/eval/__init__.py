# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Owner of the smartness-meter eval sets: the grade bins, comprehension probes,
#   and the grammar / math / reasoning JSONL axes that checkpoint_probe.py scores
#   every checkpoint against.
# - rebuild_all() regenerates all of them in-process. Every builder is seeded, so a
#   rebuild is byte-reproducible and safe to rerun; adding a probe means adding its
#   module name to BUILDERS.
# - Driven by POST /eval_sets in veritate_mri/routes/runs_routes.py.
# veritate_mri/training/builders/eval/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib

# ------------------------------------------------------------------------------------
# Constants

BUILDERS = (
    "build_grade_evals",
    "build_comprehension_probe",
    "build_grammar_eval",
    "build_math_eval",
    "build_reasoning_eval",
)
RC_OK = 0


# ------------------------------------------------------------------------------------
# Functions

def rebuild_all():
    """Run every eval-set builder in order. Returns one row per builder with its
    exit code; a builder that raises propagates to the caller."""
    out = []
    for name in BUILDERS:
        module = importlib.import_module(f"{__name__}.{name}")
        rc = module.main()
        out.append({"builder": name, "rc": int(rc or RC_OK)})
    return out
