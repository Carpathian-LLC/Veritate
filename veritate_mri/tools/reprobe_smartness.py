"""Re-run the smartness-meter axes (reading + math + grammar + reasoning)
across all checkpoints of a model.

Use this after rebuilding the eval bins or adding new probes so historical
hook directories carry the new artifacts and the dashboard trajectory plots
can render uniformly across the run.

What it does, per checkpoint <models/<name>/checkpoints/step_<N>.pt>:
    1. Load the model.
    2. Run dump_grades, dump_math, dump_grammar, dump_reasoning into the
       matching <models/<name>/hooks/step_<N>/> directory.
    3. Rename the per-step output files to canonical names
       (grades.json, math.json, grammar.json, reasoning.json).

Skips a checkpoint when its hook step dir is missing — that means the
checkpoint was never finalized through save.py and there is nowhere
canonical to put the artifacts.

Usage:
    python veritate_mri/tools/reprobe_smartness.py <model_name>
    python veritate_mri/tools/reprobe_smartness.py <model_name> --only math grammar
    python veritate_mri/tools/reprobe_smartness.py <model_name> --steps 1400 1600
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MRI_ROOT  = REPO_ROOT / "veritate_mri"

sys.path.insert(0, str(MRI_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from readers import paths  # noqa: E402

CKPT_RE = re.compile(r"^step_(\d+)\.pt$")

AXIS_TO_FN = {
    "grades":    ("dump_grades",    "grades_step_{step}.json",    "grades.json"),
    "math":      ("dump_math",      "math_step_{step}.json",      "math.json"),
    "grammar":   ("dump_grammar",   "grammar_step_{step}.json",   "grammar.json"),
    "reasoning": ("dump_reasoning", "reasoning_step_{step}.json", "reasoning.json"),
}


def discover_checkpoints(name):
    ckpt_dir = paths.checkpoints_dir(name)
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for fn in os.listdir(ckpt_dir):
        m = CKPT_RE.match(fn)
        if not m:
            continue
        out.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def hook_step_dir(name, step):
    return paths.hook_step_dir(name, step)


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run smartness-meter axes on historical checkpoints.")
    ap.add_argument("name", help="model name, e.g. tinystories_1b_bf16_v1_mega")
    ap.add_argument("--only", nargs="+", choices=list(AXIS_TO_FN.keys()),
                    help="run only the listed axes (default: all four)")
    ap.add_argument("--steps", nargs="+", type=int,
                    help="run only the listed step numbers (default: all checkpoints found)")
    args = ap.parse_args()

    axes = args.only or list(AXIS_TO_FN.keys())

    from training import checkpoint_probe as cp

    cps = discover_checkpoints(args.name)
    if not cps:
        print(f"no checkpoints found for {args.name}", file=sys.stderr)
        return 1

    if args.steps:
        wanted = set(args.steps)
        cps = [c for c in cps if c[0] in wanted]
        if not cps:
            print(f"no checkpoints match --steps {args.steps}", file=sys.stderr)
            return 1

    print(f"reprobing {len(cps)} checkpoint(s) on axes: {axes}")
    for step, ckpt_path in cps:
        step_dir = hook_step_dir(args.name, step)
        if not os.path.isdir(step_dir):
            print(f"  step {step}: no hook dir, skipping (checkpoint was never finalized through save.py)")
            continue
        try:
            model, _ = cp._load_checkpoint(ckpt_path)
        except Exception as e:
            print(f"  step {step}: load failed: {e}")
            continue
        print(f"  step {step}: loaded {ckpt_path}")
        for axis in axes:
            fn_name, src_tmpl, canonical = AXIS_TO_FN[axis]
            fn = getattr(cp, fn_name)
            try:
                fn(model, step_dir, step)
            except Exception as e:
                print(f"    {axis}: failed: {e}")
                continue
            src = os.path.join(step_dir, src_tmpl.format(step=step))
            dst = os.path.join(step_dir, canonical)
            if os.path.isfile(src):
                os.replace(src, dst)
                print(f"    {axis}: wrote {canonical}")
            else:
                print(f"    {axis}: produced no file at {src}")
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
