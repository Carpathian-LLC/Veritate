# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IDEA 20 E4 retention quiz: the closed-book exam over the 50 injected facts
#   (data/eval/e4_facts.json), both directions, no context and no retrieval —
#   every answer must come from the weights. Grading is bare-greedy per the
#   grading rule (top_k_sample=1, no repetition penalty beyond the serving ban).
# - Purpose: re-examine a slept model with ZERO further training days or weeks
#   after consolidation, so retention (not acquisition) is what is measured.
#   E4 closeout checkpoint and quiz dates live in handoff.md.
# - usage: .veritate_venv/bin/python -m tools.e4_retention_quiz <model> <step>
#          [--threads 4] [--device cpu|auto] [--out path.json]
#   Reference numbers (2026-08-20, acquisition-time): wren1_5@700 fwd 45/50
#   rev 49/50. A retention quiz scoring near these means the facts persisted.
# veritate_mri/tools/e4_retention_quiz.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import sys

# ------------------------------------------------------------------------------------
# Constants

MRI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(MRI_ROOT)
FACTS_PATH = os.path.join(MRI_ROOT, "data", "eval", "e4_facts.json")
IM_S, IM_E = "<|im_start|>", "<|im_end|>"

# ------------------------------------------------------------------------------------
# Functions


def ask(brain, q):
    prompt = f"{IM_S}user\n{q}{IM_E}\n{IM_S}assistant\n"
    out = []
    for ev in brain.stream_fast(prompt, mode="stream", max_new=48, temperature=0.7,
                                top_k_sample=1, rep_window=256, rep_penalty=0.0,
                                no_repeat_ngram=16):
        if ev.get("kind") == "fast_byte":
            out.append(ev["byte"])
    text = bytes(out).decode("utf-8", "replace")
    for m in (IM_E[:-1], IM_S[:-1]):
        i = text.find(m)
        if i >= 0:
            text = text[:i]
    return text.strip()


def run(model, step, threads=4, device="auto", out_path=None):
    if device == "cpu":
        os.environ["VERITATE_INFER_DEVICE"] = "cpu"
    os.environ["VERITATE_EXPERIENCE_LOG"] = "0"  # a quiz is not experience
    from inference.backends.pytorch import Brain
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts = json.load(f)
    ck = os.path.join(REPO, "models", model, "checkpoints", f"step_{step}.pt")
    brain = Brain(ck, threads=threads)
    fwd = rev = 0
    rows = []
    for fa in facts:
        r1, r2 = ask(brain, fa["q_fwd"]), ask(brain, fa["q_rev"])
        h1 = fa["a_fwd"].lower() in r1.lower()
        h2 = fa["a_rev"].lower() in r2.lower()
        fwd += h1
        rev += h2
        rows.append({"id": fa["id"], "fwd": h1, "rev": h2,
                     "r_fwd": r1[:60], "r_rev": r2[:60]})
        if fa["id"] % 10 == 0:
            print(f"[e4quiz] {fa['id']}: fwd={h1} rev={h2} r={r1[:50]!r}", flush=True)
    n = len(facts)
    report = {"model": f"{model}@{step}", "n": n, "fwd": fwd, "rev": rev,
              "fwd_acc": round(fwd / n, 3), "rev_acc": round(rev / n, 3),
              "rows": rows}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print(f"RESULT fwd {fwd}/{n} rev {rev}/{n}")
    return report


def main():
    ap = argparse.ArgumentParser(description="E4 closed-book retention quiz")
    ap.add_argument("model")
    ap.add_argument("step", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.model, a.step, threads=a.threads, device=a.device, out_path=a.out)


if __name__ == "__main__":
    sys.path.insert(0, MRI_ROOT)
    main()
