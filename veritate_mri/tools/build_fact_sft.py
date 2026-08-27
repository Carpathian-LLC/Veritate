# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep m1, the "study" step (IDEA 20 E4): renders atomic facts into the many
#   varied exposures weight-learning needs (reversal curse + extraction results:
#   a fact stated once, one way, is memorized-but-unextractable). ~20 surface
#   forms per fact — statement variants BOTH directions, QA pairs BOTH
#   directions, dialogue form — templated fully in-house (self-contained rule:
#   no external models). Output: ChatML train/val bins for the consolidation
#   run; mix with the model's own base corpus at launch for rehearsal.
# - fact schema (JSON list): {"stmt", "subj", "obj", "q_fwd", "a_fwd",
#   "q_rev", "a_rev", "kind": "lives"|"job"} — kind picks phrasing pools.
# - usage: python -m tools.build_fact_sft <facts.json> [--stem fact_sft]
#          [--per-fact 20] [--seed 0]
# veritate_mri/tools/build_fact_sft.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
VAL_EVERY = 20

STMT_TEMPLATES = {
    "lives": ["{subj} lives in {obj}.",
              "{subj} makes their home in {obj}.",
              "{subj} is from {obj}.",
              "{obj} is where {subj} lives.",
              "You can find {subj} in {obj}.",
              "{subj} has settled in {obj}."],
    "job":   ["{subj} works as {obj_art}.",
              "{subj} is {obj_art} by trade.",
              "{subj} earns a living as {obj_art}.",
              "The {obj} around here is {subj}.",
              "{subj}'s trade is that of {obj_art}.",
              "Being {obj_art} is {subj}'s work."],
    # self-facts carry the possessive in subj ("your sister's name"), so the
    # sentence-initial forms take the capitalized variant
    "self":  ["{subj_cap} is {obj}.",
              "{obj} is {subj}.",
              "You told me {subj} is {obj}.",
              "Noted: {subj} is {obj}.",
              "I remember that {subj} is {obj}.",
              "{subj_cap}: {obj}."],
}

# ------------------------------------------------------------------------------------
# Functions


def _an(word):
    return f"an {word}" if word[0].lower() in "aeiou" else f"a {word}"


def _cap(text):
    return text[:1].upper() + text[1:]


def render_fact(fact, rng, per_fact):
    """Yield ChatML exchange strings for one fact: varied statements (both
    directions), QA both directions, and dialogue forms."""
    outs = []
    kind = fact.get("kind", "lives")
    pool = STMT_TEMPLATES.get(kind, STMT_TEMPLATES["lives"])
    obj = fact["obj"]
    for t in pool:
        s = t.format(subj=fact["subj"], obj=obj, obj_art=_an(obj),
                     subj_cap=_cap(fact["subj"]))
        outs.append(f"{IM_S}user\nTell me something.{IM_E}\n{IM_S}assistant\n{s}{IM_E}\n")
    outs.append(f"{IM_S}user\n{fact['q_fwd']}{IM_E}\n{IM_S}assistant\n"
                f"{fact['a_fwd'].capitalize() if fact['a_fwd'][0].islower() else fact['a_fwd']}"
                f" — {fact['stmt']}{IM_E}\n")
    outs.append(f"{IM_S}user\n{fact['q_rev']}{IM_E}\n{IM_S}assistant\n"
                f"{fact['a_rev']} — {fact['stmt']}{IM_E}\n")
    outs.append(f"{IM_S}user\n{fact['stmt']} Did you get that?{IM_E}\n{IM_S}assistant\n"
                f"Got it: {fact['stmt']}{IM_E}\n")
    outs.append(f"{IM_S}user\nWhat do you know about {fact['subj']}?{IM_E}\n"
                f"{IM_S}assistant\n{fact['stmt']}{IM_E}\n")
    while len(outs) < per_fact:
        outs.append(rng.choice(outs[:len(pool) + 4]))
    rng.shuffle(outs)
    return outs[:per_fact]


def build(facts_path, stem="fact_sft", per_fact=20, seed=0, out_dir=None):
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    with open(facts_path) as f:
        facts = json.load(f)
    rng = random.Random(seed)
    exchanges = []
    for fact in facts:
        exchanges.extend(render_fact(fact, rng, per_fact))
    rng.shuffle(exchanges)
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    tb = vb = 0
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for i, ex in enumerate(exchanges):
            b = ex.encode()
            if i % VAL_EVERY == 0:
                fv.write(b)
                vb += len(b)
            else:
                ft.write(b)
                tb += len(b)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    return len(facts), len(exchanges), tb, vb


def main():
    ap = argparse.ArgumentParser(description="Render atomic facts into varied SFT exposures.")
    ap.add_argument("facts_json")
    ap.add_argument("--stem", default="fact_sft")
    ap.add_argument("--per-fact", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    nf, ne, tb, vb = build(args.facts_json, stem=args.stem,
                           per_fact=args.per_fact, seed=args.seed)
    print(f"{nf} facts -> {ne} exchanges: {args.stem}_train.bin {tb}B / val {vb}B")


if __name__ == "__main__":
    main()
