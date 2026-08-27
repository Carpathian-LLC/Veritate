# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - CLI entry point for training/fuse.py, the "merge fuse" step of the IDEA 20 T3 m1
#   spec: theta <- alpha*theta_ft + (1-alpha)*theta_prev. The module owns the logic;
#   this only parses arguments and reports.
# - usage: .veritate_venv/bin/python -m tools.fuse_checkpoints <model> <base_step>
#          <tuned_step> --alpha 0.7 [--out-step N]
# veritate_mri/tools/fuse_checkpoints.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from training import fuse as fusemod  # noqa: E402

# ------------------------------------------------------------------------------------
# Functions


def main():
    ap = argparse.ArgumentParser(description="Fuse a consolidated checkpoint back toward its base.")
    ap.add_argument("model")
    ap.add_argument("base_step", type=int)
    ap.add_argument("tuned_step", type=int)
    ap.add_argument("--alpha", type=float, required=True, help="weight on the tuned side")
    ap.add_argument("--out-step", type=int, default=None)
    a = ap.parse_args()
    step, stats = fusemod.fuse(a.model, a.base_step, a.tuned_step, a.alpha, out_step=a.out_step)
    print(f"fused {a.model}@{a.tuned_step} toward @{a.base_step} at alpha={a.alpha} "
          f"-> step_{step}.pt ({stats['fused']} fused, {stats['passed_through']} passed through)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
