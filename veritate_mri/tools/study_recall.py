# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - closed-book recall over study chunks, measured by LIKELIHOOD rather than by
#   free-running generation. "Does the model know this chunk?" is a likelihood
#   question: teacher-forced NLL over the chunk's bytes, given only its label, is
#   exactly how strongly the weights expect that content.
# - this replaces sampling for the primary metric on two grounds. It is one forward
#   pass instead of N sequential decode steps (~240x cheaper: study_exam needed ~90 s
#   a chunk on cardinal, four checkpoints would have cost four hours), and it is a
#   cleaner measurement, because free-running similarity conflates what the weights
#   know with how the decoder behaves. tools/study_exam.py remains for qualitative
#   inspection of what the model actually says.
# - the assistant role mask does the span selection: loss is computed only on the
#   chunk bytes, never on the prompt that names them, so the score cannot be inflated
#   by the model predicting its own question.
# - the load-bearing number is the GAP between studied and held-out chunks drawn from
#   the same files. Held-out NLL falls too as the model gets better at the domain;
#   only the gap isolates memorization.
# - usage: .veritate_venv/bin/python -m tools.study_recall <model> <step> [<step>...]
#          [--stem study] [--out path.json]
# veritate_mri/tools/study_recall.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.dirname(os.path.normpath(os.path.join(HERE, ".."))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
# matches build_study_corpus's recite form exactly, so the default measures the model on a
# phrasing it was trained on
PROMPT = "Show me {label}."
# a phrasing build_study_corpus never generates. Recall on a TRAINED form can be surface fit
# to that wording; recall on a held-out form is the honest test that the label is bound to
# the content rather than to the sentence that asked for it.
HELDOUT_PROMPT = "I need the code for {label} — write it out."

# ------------------------------------------------------------------------------------
# Functions


def exchange_bytes(label, text, prompt=PROMPT):
    """The chunk as the trainer saw it: the label in the user turn, the chunk itself as
    the assistant's answer."""
    return (f"{IM_S}user\n{prompt.format(label=label)}{IM_E}\n"
            f"{IM_S}assistant\n{text}{IM_E}\n").encode()


def chunk_nll(model, vt, label, text, seq, device="cpu", prompt=PROMPT):
    """Mean NLL over the chunk's bytes only. None when the exchange does not fit."""
    raw = exchange_bytes(label, text, prompt)
    if len(raw) > seq:
        raw = raw[:seq]
    buf = raw + b"\n" * (seq + 1 - len(raw))
    arr = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    toks = torch.from_numpy(arr[:-1]).unsqueeze(0).to(device)
    tgts = torch.from_numpy(arr[1:].copy()).unsqueeze(0).to(device)
    tgts = vt.apply_role_mask(toks, tgts)
    if int((tgts >= 0).sum()) == 0:
        return None
    with torch.no_grad():
        _logits, loss = model(toks, targets=tgts)
    return None if loss is None else float(loss)


def summarize(rows):
    vals = [r["nll"] for r in rows if r["nll"] is not None]
    if not vals:
        return {"n": 0}
    mean = float(np.mean(vals))
    return {"n": len(vals), "nll": round(mean, 4), "bpb": round(mean / math.log(2), 4),
            "median": round(float(np.median(vals)), 4)}


def run(model_name, steps, stem="study", device="cpu", out_path=None, exam_path=None,
        prompt=PROMPT):
    os.environ["VERITATE_EXPERIENCE_LOG"] = "0"
    from training import veritate_trainer as vt

    from veritate_core.model_patched import VeritatePatched
    exam_path = exam_path or os.path.join(CORPUS_ROOT, f"{stem}_exam.json")
    with open(exam_path, encoding="utf-8") as f:
        exam = json.load(f)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(repo)
    with open(os.path.join(repo, "models", model_name, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sh, ta = cfg["shape"], cfg["training_args"]
    seq = int(ta["seq"])
    report = {"model": model_name, "stem": stem, "prompt": prompt, "steps": {}}
    base_gap = None
    for step in steps:
        model = VeritatePatched(vocab=256, hidden=sh["hidden"], layers=sh["layers"], ffn=sh["ffn"],
                                heads=sh["heads"], seq=seq, activation=ta.get("activation", "gelu"),
                                capture_l1=False, global_mixer="recurrent")
        vt.load_resume_state(model, model_name, step, device)
        model.eval()
        entry = {}
        for split in ("studied", "holdout"):
            rows = [{"label": c["label"],
                     "nll": chunk_nll(model, vt, c["label"], c["text"], seq, device, prompt)}
                    for c in exam.get(split, [])]
            entry[split] = summarize(rows)
            entry[f"{split}_rows"] = rows
        st, ho = entry["studied"], entry["holdout"]
        entry["gap_nll"] = round(ho.get("nll", 0) - st.get("nll", 0), 4)
        # the two splits are a random partition, not a difficulty-matched pair: on
        # wren1_9 the held-out functions were EASIER at step 0 (gap -0.1413), so the raw
        # gap carries that imbalance. The signal is how far the gap MOVED from the
        # untrained baseline, which cancels it.
        if base_gap is None:
            base_gap = entry["gap_nll"]
        entry["gap_shift"] = round(entry["gap_nll"] - base_gap, 4)
        report["steps"][str(step)] = entry
        print(f"  step {step:<5} studied nll {st.get('nll')}  holdout nll {ho.get('nll')}  "
              f"gap {entry['gap_nll']:+.4f}  shift vs baseline {entry['gap_shift']:+.4f}",
              flush=True)
        del model
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description="Likelihood-based closed-book recall over study chunks")
    ap.add_argument("model")
    ap.add_argument("steps", nargs="+", type=int)
    ap.add_argument("--stem", default="study")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--exam", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt", default=None,
                    help="prompt template; use \"{label}\" when the label IS the question "
                         "(code-QA corpora built with build_study_corpus --mode qa)")
    ap.add_argument("--heldout-form", action="store_true",
                    help="query with a phrasing build_study_corpus never trained, the honest "
                         "test that the label is bound to the content and not to the wording")
    a = ap.parse_args()
    form = ("custom prompt" if a.prompt else
            "HELD-OUT form (never trained)" if a.heldout_form else "trained form")
    print(f"recall: {a.model} steps {a.steps}  [{form}]  "
          f"(lower nll = better known; GAP is the signal)", flush=True)
    run(a.model, a.steps, stem=a.stem, device=a.device, out_path=a.out, exam_path=a.exam,
        prompt=a.prompt or (HELDOUT_PROMPT if a.heldout_form else PROMPT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
