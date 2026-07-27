# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Top-level entry for the byte-level eval harness in its dashboard-facing copy.
#
# Two ways to invoke this:
#
# 1. CLI (one-shot):
#      python -m veritate_mri.eval.run_eval \
#          --ckpt models/<name>/checkpoint_step_<N>.pt \
#          --suite mmlu,hellaswag,ifeval \
#          --output report.json
#
# 2. Programmatic (used by the dashboard's POST /run/<name>/eval_deep): import
#    run_suites_on_model from this module and pass the model, suites as a list of
#    suite names, limit (None for every item), and verbose. It returns the report
#    dict the CLI would have written.
#
# Optional CLI flags:
#   --device {cpu,mps,cuda}    default: cpu (CPU smoke; MPS is busy with training)
#   --limit N                  cap items per suite (smoke-friendly)
#   --mmlu-mode {letter,text,both}    default: text
#   --mmlu-data PATH           override default sample data
#   --hellaswag-data PATH      "
#   --ifeval-data PATH         "
#   --verbose
#
# Checkpoint loading uses the same logic as veritate_mri/backends/pytorch.py::Brain so
# any Veritate-family ckpt (canonical, RoPE 85M, 800M MTP) is supported. Inference is
# pure single-byte forward; the MTP head (if present) is loaded but unused.
# veritate_mri/eval/run_eval.py
# ------------------------------------------------------------------------------------
# Imports:

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

# veritate_core lives at the repo root; the harness runs both as a module and as
# a script, so make the import resolvable either way.
if paths.REPO_ROOT not in sys.path:
    sys.path.insert(0, paths.REPO_ROOT)

# ------------------------------------------------------------------------------------
# Functions


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load any Veritate-family ckpt onto `device` and return a model in eval() mode."""
    from veritate_core.load import load_from_state_dict, shape_from_state_dict
    s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = dict(s.get("args", {}))
    sd = s["model"]
    del s
    if "tok_emb.weight" not in sd:
        raise RuntimeError(
            "Checkpoint has no tok_emb.weight; non-vanilla architectures (MoE etc.) "
            "are not supported by this eval harness."
        )
    shape = shape_from_state_dict(sd, cfg)
    model = load_from_state_dict(sd, cfg)
    del sd
    model.to(device).eval()
    return model, cfg, shape


# ------------------------------------------------------------------------------------
# Public programmatic entry (used by veritate_mri/app.py's eval_deep endpoint).

def run_suites_on_model(model,
                        suites: list[str],
                        limit: int | None = None,
                        mmlu_mode: str = "text",
                        mmlu_data: str | None = None,
                        hellaswag_data: str | None = None,
                        ifeval_data: str | None = None,
                        ifeval_max_new: int = 256,
                        verbose: bool = False,
                        progress_cb=None) -> dict:
    """Run a subset of the eval suites on an already-loaded model.

    `model` is anything with the same forward contract as `veritate_core.model.Veritate`
    (or the 800M plugin). The dashboard passes `Brain.model` directly.

    `suites` is a list drawn from {"mmlu", "hellaswag", "ifeval"}.

    `progress_cb`, if supplied, is invoked as `progress_cb(suite, i, n)` so callers
    can surface live progress (currently used by the dashboard's spinner).
    """
    from .hellaswag import DEFAULT_DATA as HS_DEFAULT
    from .hellaswag import run_hellaswag
    from .ifeval import DEFAULT_DATA as IF_DEFAULT
    from .ifeval import run_ifeval
    from .mmlu import DEFAULT_DATA as MMLU_DEFAULT
    from .mmlu import run_mmlu

    out: dict = {"suites": {}}
    suites = [s.strip().lower() for s in suites if s and s.strip()]

    if "mmlu" in suites:
        def _cb(i, n, _item):
            if progress_cb: progress_cb("mmlu", i, n)
        out["suites"]["mmlu"] = run_mmlu(
            model,
            data_path=mmlu_data or MMLU_DEFAULT,
            mode=mmlu_mode, limit=limit, verbose=verbose, progress_cb=_cb,
        )
    if "hellaswag" in suites:
        def _cb(i, n, _item):
            if progress_cb: progress_cb("hellaswag", i, n)
        out["suites"]["hellaswag"] = run_hellaswag(
            model,
            data_path=hellaswag_data or HS_DEFAULT,
            limit=limit, verbose=verbose, progress_cb=_cb,
        )
    if "ifeval" in suites:
        def _cb(i, n, _item):
            if progress_cb: progress_cb("ifeval", i, n)
        out["suites"]["ifeval"] = run_ifeval(
            model,
            data_path=ifeval_data or IF_DEFAULT,
            max_new=ifeval_max_new, limit=limit, verbose=verbose, progress_cb=_cb,
        )
    return out


# ------------------------------------------------------------------------------------
# CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a Veritate-family checkpoint")
    p.add_argument("--suite", default="mmlu",
                   help="Comma-sep: mmlu,hellaswag,ifeval")
    p.add_argument("--output", default=None,
                   help="Write JSON report here (default: stdout only)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--limit", type=int, default=None,
                   help="Cap items per suite")
    p.add_argument("--mmlu-mode", default="text",
                   choices=["letter", "text", "both"])
    p.add_argument("--mmlu-data",      default=None)
    p.add_argument("--hellaswag-data", default=None)
    p.add_argument("--ifeval-data",    default=None)
    p.add_argument("--ifeval-max-new", type=int, default=256)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    suites = [s.strip().lower() for s in args.suite.split(",") if s.strip()]
    print(f"[eval] loading {args.ckpt} -> {args.device}")
    t0 = time.perf_counter()
    model, _cfg, shape = load_checkpoint(args.ckpt, device=args.device)
    print(f"[eval] loaded in {time.perf_counter()-t0:.1f}s; shape={shape}")

    sub = run_suites_on_model(
        model,
        suites=suites,
        limit=args.limit,
        mmlu_mode=args.mmlu_mode,
        mmlu_data=args.mmlu_data,
        hellaswag_data=args.hellaswag_data,
        ifeval_data=args.ifeval_data,
        ifeval_max_new=args.ifeval_max_new,
        verbose=args.verbose,
    )
    report = {
        "ckpt": os.path.abspath(args.ckpt),
        "device": args.device,
        "shape": shape,
        **sub,
    }
    if "mmlu" in report["suites"]:
        print(f"[eval] mmlu accuracy = {report['suites']['mmlu']['accuracy']:.3f}")
    if "hellaswag" in report["suites"]:
        print(f"[eval] hellaswag accuracy = {report['suites']['hellaswag']['accuracy']:.3f}")
    if "ifeval" in report["suites"]:
        print(f"[eval] ifeval pass_rate = {report['suites']['ifeval']['pass_rate']:.3f}")

    print(json.dumps(report, indent=2, default=str))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
