# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - closed-book exam over study chunks, the generalization of e4_retention_quiz from
#   atomic facts to arbitrary documents. No context, no retrieval: every answer comes
#   from the weights.
# - the load-bearing element is the HELD-OUT control. A model that is merely good at
#   code scores well on recite without having memorized anything, so the studied score
#   alone proves nothing. The signal is the GAP between studied and held-out chunks
#   drawn from the same source distribution; build_study_corpus --holdout-frac reserves
#   the held-out set from training entirely.
# - a byte-level model will not reproduce a 1 KB body verbatim, so scoring is graded,
#   not exact: sequence similarity, common-prefix share, and (reverse direction) whether
#   the label is recovered from an excerpt.
# - usage: .veritate_venv/bin/python -m tools.study_exam <model> <step>
#          [--stem study] [--threads 4] [--max-new 400] [--limit N] [--out path.json]
# veritate_mri/tools/study_exam.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import difflib
import json
import os
import sys

# ------------------------------------------------------------------------------------
# Constants

MRI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(MRI_ROOT)
IM_S, IM_E = "<|im_start|>", "<|im_end|>"
EXCERPT_B = 200
MAX_NEW_DEFAULT = 192

# ------------------------------------------------------------------------------------
# Functions


def ask(brain, q, max_new):
    """Bare-greedy decode, per the grading rule: no sampling, no repetition penalty
    beyond the serving ban, so the score reflects the weights and not a search."""
    prompt = f"{IM_S}user\n{q}{IM_E}\n{IM_S}assistant\n"
    out = []
    for ev in brain.stream_fast(prompt, mode="stream", max_new=max_new, temperature=0.7,
                                top_k_sample=1, rep_window=256, rep_penalty=0.0,
                                no_repeat_ngram=0):
        if ev.get("kind") == "fast_byte":
            out.append(ev["byte"])
    text = bytes(out).decode("utf-8", "replace")
    for m in (IM_E[:-1], IM_S[:-1]):
        i = text.find(m)
        if i >= 0:
            text = text[:i]
    return text.strip()


def prefix_share(reply, target):
    """Share of the target's opening the reply reproduces exactly. For code this is
    the signature, which is the part a partial memory should get first."""
    n = 0
    for a, b in zip(reply, target, strict=False):
        if a != b:
            break
        n += 1
    return round(n / max(1, len(target)), 4)


def score_chunk(brain, chunk, max_new):
    """Both directions for one chunk: recite (label -> body) and identify (body -> label).

    Similarity is measured against the first max_new bytes of the chunk, not the whole
    chunk: a decode capped at max_new can never match a longer target, and scoring
    against the full length would report a budget limit as a memory failure. Both
    splits are capped identically, so the studied-vs-holdout gap stays the signal."""
    label, text = chunk["label"], chunk["text"]
    target = text[:max_new]
    recite = ask(brain, f"Show me {label}.", max_new)
    ident = ask(brain, f"Where is this from?\n{text[:EXCERPT_B]}", 48)
    sim = difflib.SequenceMatcher(None, recite, target).ratio()
    return {"label": label, "bytes": len(text), "scored_bytes": len(target),
            "sim": round(sim, 4),
            "prefix": prefix_share(recite, target),
            "identify": label.lower() in ident.lower(),
            "reply": recite[:120], "ident_reply": ident[:60]}


def summarize(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {"n": n,
            "sim": round(sum(r["sim"] for r in rows) / n, 4),
            "prefix": round(sum(r["prefix"] for r in rows) / n, 4),
            "identify": sum(r["identify"] for r in rows),
            "identify_acc": round(sum(r["identify"] for r in rows) / n, 3)}


def run(model, step, stem="study", threads=4, max_new=MAX_NEW_DEFAULT, limit=0,
        device="auto", out_path=None, exam_path=None):
    if device == "cpu":
        os.environ["VERITATE_INFER_DEVICE"] = "cpu"
    os.environ["VERITATE_EXPERIENCE_LOG"] = "0"      # an exam is not experience
    from inference.backends.pytorch import Brain
    from readers.paths import CORPUS_ROOT
    exam_path = exam_path or os.path.join(CORPUS_ROOT, f"{stem}_exam.json")
    with open(exam_path, encoding="utf-8") as f:
        exam = json.load(f)
    ck = os.path.join(REPO, "models", model, "checkpoints", f"step_{step}.pt")
    brain = Brain(ck, threads=threads)
    report = {"model": f"{model}@{step}", "stem": stem}
    for split in ("studied", "holdout"):
        chunks = exam.get(split, [])
        if limit:
            chunks = chunks[:limit]
        rows = []
        for i, ch in enumerate(chunks):
            rows.append(score_chunk(brain, ch, max_new))
            if i % 10 == 0:
                print(f"[exam] {split} {i}/{len(chunks)} sim={rows[-1]['sim']:.3f} "
                      f"id={rows[-1]['identify']}", flush=True)
        report[split] = summarize(rows)
        report[f"{split}_rows"] = rows
    st, ho = report["studied"], report["holdout"]
    report["gap_sim"] = round(st.get("sim", 0) - ho.get("sim", 0), 4)
    report["gap_identify"] = round(st.get("identify_acc", 0) - ho.get("identify_acc", 0), 3)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print(f"RESULT {model}@{step}  studied sim {st.get('sim')} prefix {st.get('prefix')} "
          f"id {st.get('identify')}/{st.get('n')}  |  holdout sim {ho.get('sim')} "
          f"prefix {ho.get('prefix')} id {ho.get('identify')}/{ho.get('n')}  |  "
          f"GAP sim {report['gap_sim']} id {report['gap_identify']}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Closed-book exam over study chunks")
    ap.add_argument("model")
    ap.add_argument("step", type=int)
    ap.add_argument("--stem", default="study")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=MAX_NEW_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N per split")
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--exam", default=None, help="path to {stem}_exam.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.model, a.step, stem=a.stem, threads=a.threads, max_new=a.max_new, limit=a.limit,
        device=a.device, out_path=a.out, exam_path=a.exam)


if __name__ == "__main__":
    sys.path.insert(0, MRI_ROOT)
    main()
