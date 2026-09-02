# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - general-capability probe: next-byte val loss for arbitrary checkpoints against a
#   NAMED corpus, so "did consolidating knowledge cost the model its general ability"
#   is a number rather than an opinion.
# - a training run's own val rows cannot answer that question. Left without an explicit
#   --val_bin the trainer validates on the heaviest member of the training mix, so a
#   study run's val measures fit to the study material; general ability is never
#   sampled at all. This tool pins the corpus, so every checkpoint of every model is
#   scored on the same bytes.
# - the draw is seeded exactly as the trainer seeds it (seed + 1) and the loader is the
#   trainer's own, so the Nth iteration reads byte-identical windows across models.
#   Comparing two checkpoints at the same iters is exact: same windows, only the
#   weights differ.
# - the role mask is NOT applied, matching the trainer: apply_role_mask touches only the
#   training batch (veritate_trainer.py train loop), while evaluate() consumes raw
#   draws. Masking here would produce numbers that no train.csv row can be compared to.
# - usage: .veritate_venv/bin/python -m tools.val_eval <model> <step> [<step>...]
#          [--val-bin mixed_chat] [--iters 8] [--batch 4] [--out path.json]
#          [--baseline <model>:<step>]
# veritate_mri/tools/val_eval.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.dirname(os.path.normpath(os.path.join(HERE, ".."))))

from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

# tools/ -> veritate_mri/ -> repo root. Two dirnames stop at veritate_mri, where no
# models/ exists, so every lookup fails with a confusing FileNotFoundError.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VAL = "mixed_chat"
DEFAULT_ITERS = 8
DEFAULT_BATCH = 4

# ------------------------------------------------------------------------------------
# Functions


def resolve_val_bin(name):
    """Accept a stem ('mixed_chat'), a bare filename, or a path; return the val bin path."""
    if os.path.sep in name or name.endswith(".bin"):
        path = name if os.path.isabs(name) else os.path.join(CORPUS_ROOT, name)
    else:
        path = os.path.join(CORPUS_ROOT, name + "_val.bin")
    if not os.path.exists(path):
        raise FileNotFoundError("no val bin at " + path)
    return path


def load_config(model_name):
    with open(os.path.join(REPO, "models", model_name, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg, seq):
    from veritate_core.model_patched import VeritatePatched
    sh, ta = cfg["shape"], cfg["training_args"]
    return VeritatePatched(vocab=256, hidden=sh["hidden"], layers=sh["layers"], ffn=sh["ffn"],
                           heads=sh["heads"], seq=seq, activation=ta.get("activation", "gelu"),
                           capture_l1=False, global_mixer="recurrent")


def score(model_name, step, val_path, iters, batch, device="cpu"):
    """Val loss for one checkpoint. Returns None when every iteration was skipped."""
    from training import veritate_trainer as vt
    cfg = load_config(model_name)
    ta = cfg["training_args"]
    seq, n_chunks = int(ta["seq"]), int(ta["n_chunks"])
    model = build_model(cfg, seq)
    vt.load_resume_state(model, model_name, step, device)
    model.eval()
    draw, _n = vt.make_data_loader(val_path, seq * n_chunks, batch, int(ta.get("seed", 0)) + 1)
    v = vt.evaluate(model, draw, iters, seq, None, int(ta.get("bptt_window", 1)),
                    device_type=device, state_carry=ta.get("state_carry", "off"))
    del model
    return v


def run(model_name, steps, val_bin=DEFAULT_VAL, iters=DEFAULT_ITERS, batch=DEFAULT_BATCH,
        device="cpu", baseline=None, out_path=None):
    os.environ["VERITATE_EXPERIENCE_LOG"] = "0"
    val_path = resolve_val_bin(val_bin)
    report = {"val_bin": os.path.basename(val_path), "iters": iters, "batch": batch,
              "baseline": None, "rows": []}
    print(f"val {os.path.basename(val_path)}  iters {iters}  batch {batch}", flush=True)

    ref = None
    if baseline:
        bm, _, bs = baseline.partition(":")
        ref = score(bm, int(bs), val_path, iters, batch, device)
        report["baseline"] = {"model": bm, "step": int(bs), "val": ref}
        print(f"  baseline {bm}@{bs}: val {ref:.6f}", flush=True)

    for step in steps:
        v = score(model_name, step, val_path, iters, batch, device)
        row = {"model": model_name, "step": step, "val": v}
        if ref and v is not None:
            row["delta_pct"] = round((v - ref) / ref * 100.0, 3)
            print(f"  {model_name}@{step}: val {v:.6f}   {row['delta_pct']:+.2f}% vs baseline",
                  flush=True)
        else:
            print(f"  {model_name}@{step}: val {v}", flush=True)
        report["rows"].append(row)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description="Next-byte val loss for checkpoints on a pinned corpus")
    ap.add_argument("model")
    ap.add_argument("steps", nargs="+", type=int)
    ap.add_argument("--val-bin", default=DEFAULT_VAL)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", default=None,
                    help="<model>:<step> scored first; every row reports its delta against it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.model, a.steps, val_bin=a.val_bin, iters=a.iters, batch=a.batch, device=a.device,
        baseline=a.baseline, out_path=a.out)


if __name__ == "__main__":
    main()
