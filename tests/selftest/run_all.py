# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - selftest entry point. runs every check_*.py under tests/selftest/checks/,
#   dumps per-check logs and a run summary.
# - works both as `python -m tests.selftest.run_all` and as a direct script.
# tests/selftest/run_all.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import os
import sys

# bootstrap when invoked as a direct script (sys.path lacks repo root).
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.selftest import _runner       # noqa: E402
from tests.selftest import _status       # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

EXIT_OK   = 0
EXIT_FAIL = 1

# ------------------------------------------------------------------------------------
# Functions

def _parse_csv(value):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_parser():
    p = argparse.ArgumentParser(
        prog="selftest",
        description="run every selftest check and dump per-check logs.",
    )
    p.add_argument("--only",       default="", help="comma-separated check names to run (e.g. core_model,mri_endpoints).")
    p.add_argument("--skip",       default="", help="comma-separated check names to skip.")
    p.add_argument("--list",       action="store_true", help="list discovered checks and exit.")
    p.add_argument("--fast",       action="store_true", help="skip checks marked SLOW = True.")
    p.add_argument("--no-network", action="store_true", help="skip checks marked REQUIRES_NETWORK = True.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.list:
        for name in _runner.discover():
            print(name)
        return EXIT_OK

    options = {
        "only":       _parse_csv(args.only),
        "skip":       _parse_csv(args.skip),
        "skip_slow":  bool(args.fast),
        "no_network": bool(args.no_network),
    }

    exit_code, results = _runner.run_all(options)
    if exit_code != EXIT_OK:
        for r in results:
            if r.status == _status.STATUS_FAIL:
                print(f"FAIL  {r.name}: see {r.log_path}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
