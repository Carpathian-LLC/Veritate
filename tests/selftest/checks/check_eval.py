# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - eval harness imports + the run_eval entry callable is present. live eval
#   runs need a brain; this check stays cold.
# tests/selftest/checks/check_eval.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA          = "eval"
SUITE_MODS    = ("mmlu", "hellaswag", "ifeval", "score")
ENTRY_NAMES   = ("run_eval", "run_suites_on_model", "main")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """eval.run_eval imports and exposes a callable entry; every suite module
    imports."""
    failures = []
    for name in SUITE_MODS:
        try:
            __import__("eval." + name)
        except Exception as exc:
            failures.append(f"eval.{name}: {exc}")
    if failures:
        return _status.fail("eval", failures[0], {"errors": failures})

    try:
        import eval.run_eval as run_eval_mod
    except Exception as exc:
        return _status.fail("eval", f"run_eval import: {exc}")
    found = [n for n in ENTRY_NAMES if callable(getattr(run_eval_mod, n, None))]
    if not found:
        return _status.fail("eval", f"no entry found among {ENTRY_NAMES}")
    return _status.ok("eval", f"{len(SUITE_MODS)} suites + entry: {found[0]}")
