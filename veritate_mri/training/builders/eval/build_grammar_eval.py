# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Generate grammar eval pairs for the smartness-meter grammar axis.
#   Each pair: (correct_sentence, mutated_sentence). Probe scores both under
#   the model and counts a preference for the correct one (lower NLL).
# - Four pair-types, 50 each, hand-templated with slot fills. Output:
#   veritate_mri/data/eval/grade/grammar/<type>.jsonl
#   {"correct", "incorrect", "type"}.
# - Types: sv_agreement, articles, tense, word_order.
# - Usage:
#     python veritate_mri/training/builders/eval/build_grammar_eval.py
# veritate_mri/training/builders/eval/build_grammar_eval.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import random
from pathlib import Path


# ------------------------------------------------------------------------------------
# Constants

SEED = 4242
N_PER_TYPE = 50

SUBJ_SING = ["The cat", "The boy", "The teacher", "The dog", "The river", "The clock", "The girl", "The horse", "The man", "The bird"]
SUBJ_PLUR = ["The cats", "The boys", "The teachers", "The dogs", "The rivers", "The clocks", "The girls", "The horses", "The men", "The birds"]
VERB_S    = ["sleeps", "runs", "watches", "barks", "flows", "ticks", "laughs", "gallops", "walks", "sings"]
VERB_BASE = ["sleep", "run", "watch", "bark", "flow", "tick", "laugh", "gallop", "walk", "sing"]


# ------------------------------------------------------------------------------------
# Functions

def sv_agreement(rng):
    out = []
    for i in range(N_PER_TYPE):
        if rng.random() < 0.5:
            j = rng.randint(0, len(SUBJ_SING) - 1)
            correct   = f"{SUBJ_SING[j]} {VERB_S[j]}."
            incorrect = f"{SUBJ_SING[j]} {VERB_BASE[j]}."
        else:
            j = rng.randint(0, len(SUBJ_PLUR) - 1)
            correct   = f"{SUBJ_PLUR[j]} {VERB_BASE[j]}."
            incorrect = f"{SUBJ_PLUR[j]} {VERB_S[j]}."
        out.append({"correct": correct, "incorrect": incorrect, "type": "sv_agreement"})
    return out


VOWEL_NOUNS = ["apple", "orange", "umbrella", "elephant", "ice cube", "owl", "egg", "ant", "octopus", "envelope"]
CONS_NOUNS  = ["banana", "cat", "tree", "dog", "house", "book", "river", "stone", "table", "mountain"]


def articles(rng):
    out = []
    for _ in range(N_PER_TYPE):
        if rng.random() < 0.5:
            n = rng.choice(VOWEL_NOUNS)
            out.append({"correct": f"She saw an {n}.", "incorrect": f"She saw a {n}.", "type": "articles"})
        else:
            n = rng.choice(CONS_NOUNS)
            out.append({"correct": f"She saw a {n}.", "incorrect": f"She saw an {n}.", "type": "articles"})
    return out


PAST_VERBS    = ["walked", "ran", "ate", "saw", "drove", "wrote", "called", "bought", "found", "took"]
PRESENT_VERBS = ["walks",  "runs", "eats", "sees", "drives", "writes", "calls", "buys", "finds", "takes"]


def tense(rng):
    out = []
    for _ in range(N_PER_TYPE):
        j = rng.randint(0, len(PAST_VERBS) - 1)
        if rng.random() < 0.5:
            correct   = f"Yesterday he {PAST_VERBS[j]} to the store."
            incorrect = f"Yesterday he {PRESENT_VERBS[j]} to the store."
        else:
            correct   = f"Every morning she {PRESENT_VERBS[j]} to school."
            incorrect = f"Every morning she {PAST_VERBS[j]} to school."
        out.append({"correct": correct, "incorrect": incorrect, "type": "tense"})
    return out


WORD_ORDER_PAIRS = [
    ("She gave the book to him.",      "She gave to him the book."),
    ("They quickly finished the work.", "They finished quickly the work."),
    ("He has been waiting all day.",   "He been has waiting all day."),
    ("I will be there at noon.",       "I be will there at noon."),
    ("We have not seen them yet.",     "We not have seen them yet."),
    ("The man with the hat left.",     "The man the hat with left."),
    ("She is older than her brother.", "She is older her brother than."),
    ("They sat down on the bench.",    "They down sat on the bench."),
    ("He picked up the keys.",         "He picked the keys up.")[::1],  # both fine; replace
    ("Why are you late again?",        "Why you are late again?"),
]
# replace ambiguous one with clearer pair
WORD_ORDER_PAIRS[8] = ("She has always loved music.", "She always has loved music.")


def word_order(rng):
    out = []
    for _ in range(N_PER_TYPE):
        c, i = rng.choice(WORD_ORDER_PAIRS)
        out.append({"correct": c, "incorrect": i, "type": "word_order"})
    return out


TYPES = [
    ("sv_agreement", sv_agreement),
    ("articles",     articles),
    ("tense",        tense),
    ("word_order",   word_order),
]


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "grade_eval" / "grammar"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name, fn in TYPES:
        pairs = fn(rng)
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  wrote: {path.name} ({len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
