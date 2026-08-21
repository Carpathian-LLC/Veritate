# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep m2, the "raw transcript" arm (IDEA 20 E4): renders the same atomic
#   facts as NATURAL MULTI-TURN CONVERSATIONS instead of m1's drilled study
#   pairs — the form the experience log actually records. The matched-arm
#   question: does consolidating what a chat looks like teach what a fact-SFT
#   teaches? Differences from build_fact_sft.py, all deliberate:
#     * facts arrive embedded in small talk, told BY the user;
#     * the assistant echoes the fact in its own turn (assistant loss mask
#       only trains on assistant bytes — a fact the model never restates is
#       invisible to consolidation; real chat echoes, so the corpus does);
#     * reverse framing occurs at conversational frequency (~1 in 3 mentions),
#       not m1's drilled 50% — if the reversal curse reappears here, that is
#       an m2 finding, not a corpus bug;
#     * distractor turns surround the fact so exposure density matches a real
#       transcript, not a flashcard deck.
# - same fact schema and exchange-stream bin format as m1; templated fully
#   in-house (self-contained rule: no external models).
# - usage: python -m tools.build_fact_chats <facts.json> [--stem fact_chat]
#          [--per-fact 20] [--seed 0]
# veritate_mri/tools/build_fact_chats.py
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

# how the user drops the fact into conversation, by fact kind
TELL_TEMPLATES = {
    "lives": ["My friend {subj} lives in {obj} these days.",
              "I was visiting {subj} over in {obj} last week.",
              "You remember {subj}? They've moved to {obj}.",
              "{subj} has a place in {obj} now.",
              "I got a letter from {subj} — still in {obj}."],
    "job":   ["My friend {subj} works as {obj_art} now.",
              "{subj} took up work as {obj_art}.",
              "Did I mention {subj} is {obj_art} these days?",
              "{subj} makes a living as {obj_art}.",
              "I ran into {subj} — busy as ever, being {obj_art}."],
}
# how the assistant echoes it back (the bytes consolidation actually trains on)
ECHO_TEMPLATES = {
    "lives": ["{obj} — so {subj} lives in {obj} now. How do they find it?",
              "I'll remember that: {subj} lives in {obj}.",
              "So {subj} is in {obj} these days. Noted.",
              "{subj} in {obj} — got it."],
    "job":   ["{obj_art_cap} — so {subj} works as {obj_art} now. Do they enjoy it?",
              "I'll remember that: {subj} works as {obj_art}.",
              "So {subj} earns a living as {obj_art} these days. Noted.",
              "{subj}, {obj_art} — got it."],
}
# reverse-framed mention: the user asks from the object side, mid-conversation
REV_TEMPLATES = {
    "lives": ("Who did you say lives in {obj} again?", "That's {subj} — {subj} lives in {obj}."),
    "job":   ("Who was it that works as {obj_art}?", "That's {subj} — {subj} works as {obj_art}."),
}
# distractor turns: unrelated small talk surrounding the fact
DISTRACTORS = [
    ("Rotten weather today.", "It does sound grim out there. Staying in?"),
    ("I finally fixed the gate hinge.", "Good — no more creaking, then."),
    ("Thinking about soup for dinner.", "Soup sounds right for the season."),
    ("The market was heaving this morning.", "Busy days there lately. Find what you needed?"),
    ("My knee's been acting up again.", "Sorry to hear it. Rest it if you can."),
    ("The bus was late twice this week.", "That route has been unreliable lately."),
]

# ------------------------------------------------------------------------------------
# Functions


def _an(word):
    return f"an {word}" if word[0].lower() in "aeiou" else f"a {word}"


def _fmt(t, fact):
    obj_art = _an(fact["obj"])
    return t.format(subj=fact["subj"], obj=fact["obj"], obj_art=obj_art,
                    obj_art_cap=obj_art.capitalize())


def render_conversation(fact, rng, reverse=False):
    """One small conversation: distractor turn, the fact told and echoed (or
    reverse-asked and answered), distractor turn. Returns one exchange string."""
    kind = fact.get("kind", "lives")
    turns = []
    d1 = rng.choice(DISTRACTORS)
    turns.append((d1[0], d1[1]))
    if reverse:
        q, a = REV_TEMPLATES.get(kind, REV_TEMPLATES["lives"])
        turns.append((_fmt(q, fact), _fmt(a, fact)))
    else:
        tell = _fmt(rng.choice(TELL_TEMPLATES.get(kind, TELL_TEMPLATES["lives"])), fact)
        echo = _fmt(rng.choice(ECHO_TEMPLATES.get(kind, ECHO_TEMPLATES["lives"])), fact)
        turns.append((tell, echo))
    d2 = rng.choice(DISTRACTORS)
    if d2 != d1:
        turns.append((d2[0], d2[1]))
    out = []
    for u, a in turns:
        out.append(f"{IM_S}user\n{u}{IM_E}\n{IM_S}assistant\n{a}{IM_E}\n")
    return "".join(out)


def build(facts_path, stem="fact_chat", per_fact=20, seed=0, out_dir=None):
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    with open(facts_path) as f:
        facts = json.load(f)
    rng = random.Random(seed)
    exchanges = []
    for fact in facts:
        for i in range(per_fact):
            exchanges.append(render_conversation(fact, rng, reverse=(i % 3 == 2)))
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
    ap = argparse.ArgumentParser(description="Render atomic facts as natural chat transcripts (m2 arm).")
    ap.add_argument("facts_json")
    ap.add_argument("--stem", default="fact_chat")
    ap.add_argument("--per-fact", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    nf, ne, tb, vb = build(args.facts_json, stem=args.stem,
                           per_fact=args.per_fact, seed=args.seed)
    print(f"{nf} facts -> {ne} conversations: {args.stem}_train.bin {tb}B / val {vb}B")


if __name__ == "__main__":
    main()
