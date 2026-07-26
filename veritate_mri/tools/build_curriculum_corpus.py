# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds a developmental "child concepts" byte corpus for tiny models (10M):
#   objects/categories -> properties -> spatial relations -> actions with ROLE
#   diversity (active/passive/who/what) -> mini narratives. The action stage
#   states each event four ways so the model must track subject-vs-object roles
#   instead of emitting the salient noun (the F4 role-binding seed).
# - Deterministic: fixed seed, combinatorial expansion. Output is raw bytes
#   (vocab=256). No meta sidecar (preflight rule 37); provenance lives in
#   developer_documentation/corpus/curriculum_corpus.md.
# veritate_mri/tools/build_curriculum_corpus.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import os
import random

# ------------------------------------------------------------------------------------
# Constants

SEED = 0
VAL_FRACTION = 0.03
BLOCK_SEP = "\n\n"

OBJECTS = [
    ("ball", "toy", "a"), ("doll", "toy", "a"), ("block", "toy", "a"),
    ("apple", "food", "an"), ("cake", "food", "a"), ("egg", "food", "an"),
    ("cup", "thing", "a"), ("book", "thing", "a"), ("box", "thing", "a"),
    ("key", "thing", "a"), ("hat", "thing", "a"), ("shoe", "thing", "a"),
    ("car", "thing", "a"), ("spoon", "thing", "a"), ("chair", "thing", "a"),
    ("dog", "animal", "a"), ("cat", "animal", "a"), ("bird", "animal", "a"),
    ("fish", "animal", "a"), ("cow", "animal", "a"), ("frog", "animal", "a"),
]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown"]
SIZES = ["big", "small", "tall", "short", "little"]
TEXTURES = ["soft", "hard", "hot", "cold", "wet", "dry", "old", "new"]
SURFACES = ["table", "chair", "box", "floor", "bed", "shelf", "mat"]
SPATIAL = ["on", "under", "in", "near", "behind"]
ANIMATE = ["the dog", "the cat", "the girl", "the boy", "the bird", "the man", "the woman"]
# (base, present, past)
ACTIONS = [
    ("chase", "chases", "chased"), ("see", "sees", "saw"), ("hold", "holds", "held"),
    ("push", "pushes", "pushed"), ("follow", "follows", "followed"),
    ("find", "finds", "found"), ("carry", "carries", "carried"), ("kick", "kicks", "kicked"),
]

# Stage 2: harder material with real headroom. Stage 1 saturates a 16M model
# (val 0.0925 by step 1250, then flat) because its inventories are tiny, so a
# grown model has nothing left to learn. Stage 2 widens every inventory and adds
# structures stage 1 lacks: three-role ditransitives, pronoun coreference,
# negation, comparatives, counting, and subordinate clauses.

S2_OBJECTS = [
    ("ball", "toy", "a"), ("doll", "toy", "a"), ("block", "toy", "a"), ("kite", "toy", "a"),
    ("drum", "toy", "a"), ("puzzle", "toy", "a"), ("robot", "toy", "a"), ("marble", "toy", "a"),
    ("apple", "food", "an"), ("cake", "food", "a"), ("egg", "food", "an"), ("pear", "food", "a"),
    ("bread", "food", "a"), ("plum", "food", "a"), ("carrot", "food", "a"), ("berry", "food", "a"),
    ("cup", "thing", "a"), ("book", "thing", "a"), ("box", "thing", "a"), ("key", "thing", "a"),
    ("hat", "thing", "a"), ("shoe", "thing", "a"), ("car", "thing", "a"), ("spoon", "thing", "a"),
    ("chair", "thing", "a"), ("lamp", "thing", "a"), ("clock", "thing", "a"), ("brush", "thing", "a"),
    ("rope", "thing", "a"), ("coin", "thing", "a"), ("towel", "thing", "a"), ("basket", "thing", "a"),
    ("dog", "animal", "a"), ("cat", "animal", "a"), ("bird", "animal", "a"), ("fish", "animal", "a"),
    ("cow", "animal", "a"), ("frog", "animal", "a"), ("horse", "animal", "a"), ("mouse", "animal", "a"),
    ("sheep", "animal", "a"), ("duck", "animal", "a"), ("goat", "animal", "a"), ("bee", "animal", "a"),
]
S2_COLORS = COLORS + ["orange", "purple", "grey", "pink", "silver", "golden"]
S2_SIZES = SIZES + ["huge", "tiny", "wide", "narrow", "heavy", "light"]
S2_TEXTURES = TEXTURES + ["smooth", "rough", "sharp", "clean", "dirty", "shiny", "sticky", "empty"]
S2_PLACES = [
    "kitchen", "garden", "attic", "porch", "barn", "hallway", "meadow", "cellar",
    "library", "beach", "bridge", "market", "yard", "pond", "tower", "path",
]
S2_SURFACES = SURFACES + ["bench", "cart", "ledge", "crate", "stool", "windowsill", "step", "rug"]
S2_SPATIAL = SPATIAL + ["beside", "above", "below", "between", "against", "beyond"]
# (phrase, pronoun, possessive)
S2_ANIMATE = [
    ("the dog", "it", "its"), ("the cat", "it", "its"), ("the girl", "she", "her"),
    ("the boy", "he", "his"), ("the bird", "it", "its"), ("the man", "he", "his"),
    ("the woman", "she", "her"), ("the farmer", "they", "their"), ("the baker", "they", "their"),
    ("the sailor", "they", "their"), ("the painter", "they", "their"), ("the driver", "they", "their"),
    ("the teacher", "they", "their"), ("the gardener", "they", "their"), ("the singer", "they", "their"),
    ("the tailor", "they", "their"), ("the miller", "they", "their"), ("the hunter", "they", "their"),
]
S2_ACTIONS = ACTIONS + [
    ("watch", "watches", "watched"), ("lift", "lifts", "lifted"), ("drop", "drops", "dropped"),
    ("wash", "washes", "washed"), ("paint", "paints", "painted"), ("catch", "catches", "caught"),
    ("pull", "pulls", "pulled"), ("greet", "greets", "greeted"), ("feed", "feeds", "fed"),
    ("draw", "draws", "drew"), ("count", "counts", "counted"), ("mend", "mends", "mended"),
]
# (base, present, past) — three-role verbs: subject, theme, recipient
S2_DITRANS = [
    ("give", "gives", "gave"), ("show", "shows", "showed"), ("send", "sends", "sent"),
    ("hand", "hands", "handed"), ("pass", "passes", "passed"), ("bring", "brings", "brought"),
    ("offer", "offers", "offered"), ("lend", "lends", "lent"), ("sell", "sells", "sold"),
]
# Passive voice needs the PAST PARTICIPLE, not the simple past. Only irregulars
# where the two differ are listed; everything else falls back to the past form.
PARTICIPLES = {
    "see": "seen", "draw": "drawn", "give": "given", "show": "shown",
    "take": "taken", "throw": "thrown", "fall": "fallen", "eat": "eaten",
    "hide": "hidden", "write": "written", "break": "broken", "choose": "chosen",
}
# Relations that take a single landmark ("between" needs two, so it is excluded
# anywhere only one surface is named).
S2_SPATIAL_ONE = [r for r in S2_SPATIAL if r != "between"]
# Blocks that emit a "Who <verb> the <object>?" question must NOT draw that
# object from the animals, because animals are also ANIMATE entities — an
# accidental match then leaks a held-out (subject, object) pair's answer into
# training. Measured before this guard: 7/91 held-out WHO answers leaked.
S2_THINGS = [o for o in S2_OBJECTS if o[1] != "animal"]
S2_MANNER = ["quickly", "slowly", "quietly", "gently", "carefully", "eagerly", "loudly", "neatly"]
S2_TIMES = ["yesterday", "this morning", "last night", "at noon", "after lunch", "before dawn"]
S2_NUMBERS = [(2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six"), (7, "seven")]
S2_CAUSAL = ["because", "after", "before", "while", "since", "although"]

# ------------------------------------------------------------------------------------
# Functions

def _cap(s):
    return s[0].upper() + s[1:]


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def _pp(verb):
    return PARTICIPLES.get(verb[0], verb[2])


def _agree(pronoun, verb):
    # singular "they" (and plural pronouns) take the base form: "they pull",
    # not "they pulls". he/she/it take the 3rd-person-singular form.
    return verb[0] if pronoun in ("they", "we", "you", "i") else verb[1]


def naming_block(rng, obj):
    name, cat, art = obj
    cart = _art(cat)
    lines = [
        f"{_cap(art)} {name} is {cart} {cat}.",
        f"The {name} is {cart} {cat}.",
        f"Is {art} {name} {cart} {cat}? Yes, {art} {name} is {cart} {cat}.",
        f"What is {art} {name}? {_cap(art)} {name} is {cart} {cat}.",
    ]
    rng.shuffle(lines)
    return " ".join(lines)


def property_block(rng, obj):
    name, _, art = obj
    color = rng.choice(COLORS)
    size = rng.choice(SIZES)
    tex = rng.choice(TEXTURES)
    lines = [
        f"The {name} is {color}.",
        f"The {name} is {size}.",
        f"The {name} is {tex}.",
        f"It is a {size} {color} {name}.",
        f"What color is the {name}? The {name} is {color}.",
        f"The {color} {name} is {tex}.",
    ]
    rng.shuffle(lines)
    return " ".join(lines)


def spatial_block(rng, obj):
    name, _, _ = obj
    surf = rng.choice(SURFACES)
    rel = rng.choice(SPATIAL)
    lines = [
        f"The {name} is {rel} the {surf}.",
        f"Where is the {name}? The {name} is {rel} the {surf}.",
        f"The {surf} has the {name} {rel} it.",
        f"Look, the {name} is {rel} the {surf}.",
    ]
    rng.shuffle(lines)
    return " ".join(lines)


def action_block(rng, subj, obj_a, verb, questions=True):
    base, pres, past = verb
    # Four surface forms of ONE event so subject/object roles cannot be guessed
    # from position: active, passive, subject-question, object-question. When
    # questions=False (held-out pair) only the two declarative forms are shown,
    # so the who/what mapping must generalize from OTHER pairs to be answered.
    s = subj
    o = obj_a
    decl = f"{_cap(s)} {pres} {o}. {_cap(o)} is {_pp(verb)} by {s}. "
    if not questions:
        return decl.strip()
    return decl + f"Who {pres} {o}? {_cap(s)} does. What does {s} {base}? {_cap(o)}."


def holdout_verb_map(frac):
    # Deterministic: which (subj,obj) pairs are held out, and the single verb
    # each is shown with declaratively (so the test question's verb matches).
    rng = random.Random(SEED + 99)
    pairs = [(s, o) for s in ANIMATE for o in ANIMATE if s != o]
    rng.shuffle(pairs)
    n = int(len(pairs) * frac)
    return {p: rng.choice(ACTIONS) for p in pairs[:n]}


def test_items(verbmap):
    items = []
    for (s, o) in sorted(verbmap):
        base, pres, past = verbmap[(s, o)]
        items.append({
            "decl": f"{_cap(s)} {pres} {o}.",
            "who_q": f"Who {pres} {o}?",
            "gold_subj": s,
            "what_q": f"What does {s} {base}?",
            "gold_obj": o,
        })
    return items


def narrative_block(rng):
    subj = rng.choice(ANIMATE)
    obj = rng.choice(OBJECTS)
    name, _, art = obj
    color = rng.choice(COLORS)
    surf = rng.choice(SURFACES)
    rel = rng.choice(SPATIAL)
    return (
        f"{_cap(subj)} has {art} {color} {name}. "
        f"The {name} is {rel} the {surf}. "
        f"{_cap(subj)} sees the {name}. "
        f"{_cap(subj)} is happy."
    )


def s2_naming_block(rng, obj):
    name, cat, art = obj
    cart = _art(cat)
    color = rng.choice(S2_COLORS)
    place = rng.choice(S2_PLACES)
    return (
        f"{_cap(art)} {name} is {cart} {cat}. The {color} {name} is in the {place}. "
        f"What is {art} {name}? {_cap(art)} {name} is {cart} {cat}. "
        f"Where is the {color} {name}? It is in the {place}."
    )


def s2_property_block(rng, obj):
    name, _, art = obj
    c1, c2 = rng.sample(S2_COLORS, 2)
    size = rng.choice(S2_SIZES)
    tex = rng.choice(S2_TEXTURES)
    return (
        f"The {size} {name} is {c1}. The other {name} is {c2}. "
        f"The {c1} {name} is {tex}. Which {name} is {tex}? The {c1} one. "
        f"The {c2} {name} is not {tex}."
    )


def s2_comparative_block(rng):
    a, b = rng.sample(S2_OBJECTS, 2)
    size = rng.choice(["bigger", "smaller", "heavier", "lighter", "taller", "shorter"])
    opp = {"bigger": "smaller", "smaller": "bigger", "heavier": "lighter",
           "lighter": "heavier", "taller": "shorter", "shorter": "taller"}[size]
    return (
        f"The {a[0]} is {size} than the {b[0]}. So the {b[0]} is {opp} than the {a[0]}. "
        f"Which is {size}, the {a[0]} or the {b[0]}? The {a[0]}."
    )


def s2_negation_block(rng, obj):
    name, _, _ = obj
    s1, s2 = rng.sample(S2_SURFACES, 2)
    rel = rng.choice(S2_SPATIAL_ONE)
    return (
        f"The {name} is not {rel} the {s1}. The {name} is {rel} the {s2}. "
        f"Is the {name} {rel} the {s1}? No, it is {rel} the {s2}."
    )


def s2_count_block(rng, obj):
    name, cat, _ = obj
    n, word = rng.choice(S2_NUMBERS)
    place = rng.choice(S2_PLACES)
    m, mword = rng.choice([x for x in S2_NUMBERS if x[0] != n])
    return (
        f"There are {word} {name}s in the {place}. {_cap(mword)} more {name}s are outside. "
        f"How many {name}s are in the {place}? {_cap(word)}."
    )


def s2_ditransitive_block(rng, subj, theme, recip, verb, questions=True):
    base, pres, past = verb
    decl = (f"{_cap(subj)} {pres} the {theme} to {recip}. "
            f"The {theme} is {_pp(verb)} to {recip} by {subj}. ")
    if not questions:
        return decl.strip()
    return decl + (f"Who {pres} the {theme}? {_cap(subj)} does. "
                   f"What does {subj} {base}? The {theme}. "
                   f"Who receives the {theme}? {_cap(recip)}.")


def s2_coref_block(rng):
    (s, pro, poss) = rng.choice(S2_ANIMATE)
    obj = rng.choice(S2_THINGS)
    place = rng.choice(S2_PLACES)
    verb = rng.choice(S2_ACTIONS)
    return (
        f"{_cap(s)} found {obj[2]} {obj[0]} in the {place}. "
        f"{_cap(pro)} {_agree(pro, verb)} {poss} {obj[0]} every day. "
        f"Who {verb[1]} the {obj[0]}? {_cap(s)} does."
    )


def s2_multiclause_block(rng):
    (s1, _, _) = rng.choice(S2_ANIMATE)
    (s2, _, _) = rng.choice(S2_ANIMATE)
    conn = rng.choice(S2_CAUSAL)
    v1 = rng.choice(S2_ACTIONS)
    v2 = rng.choice(S2_ACTIONS)
    obj = rng.choice(S2_THINGS)
    manner = rng.choice(S2_MANNER)
    when = rng.choice(S2_TIMES)
    return (
        f"{_cap(conn)} {s1} {v1[1]} the {obj[0]}, {s2} {v2[1]} it {manner}. "
        f"{_cap(when)} {s2} {v2[2]} the {obj[0]} again. "
        f"Who {v2[1]} the {obj[0]} {manner}? {_cap(s2)} does."
    )


def s2_narrative_block(rng):
    (s, pro, poss) = rng.choice(S2_ANIMATE)
    obj = rng.choice(S2_OBJECTS)
    place = rng.choice(S2_PLACES)
    color = rng.choice(S2_COLORS)
    rel = rng.choice(S2_SPATIAL_ONE)
    surf = rng.choice(S2_SURFACES)
    when = rng.choice(S2_TIMES)
    manner = rng.choice(S2_MANNER)
    return (
        f"{_cap(when)} {s} walked to the {place}. "
        f"{_cap(pro)} carried {obj[2]} {color} {obj[0]} in {poss} hands. "
        f"{_cap(pro)} put it {rel} the {surf} {manner}. "
        f"The {color} {obj[0]} stayed {rel} the {surf} all day."
    )


def s2_holdout_verb_map(frac):
    rng = random.Random(SEED + 199)
    ents = [e[0] for e in S2_ANIMATE]
    pairs = [(s, o) for s in ents for o in ents if s != o]
    rng.shuffle(pairs)
    n = int(len(pairs) * frac)
    return {p: rng.choice(S2_ACTIONS) for p in pairs[:n]}


def build_stream_s2(target_bytes, verbmap=None):
    rng = random.Random(SEED + 2)
    held = verbmap or {}
    ents = [e[0] for e in S2_ANIMATE]
    blocks = []
    total = 0
    obj_builders = [s2_naming_block, s2_property_block, s2_negation_block, s2_count_block]
    free_builders = [s2_comparative_block, s2_coref_block, s2_multiclause_block, s2_narrative_block]
    while total < target_bytes:
        b = rng.choice(obj_builders)(rng, rng.choice(S2_OBJECTS))
        blocks.append(b)
        total += len(b)
        b = rng.choice(free_builders)(rng)
        blocks.append(b)
        total += len(b)
        subj, o = rng.sample(ents, 2)
        if (subj, o) in held:
            b = action_block(rng, subj, o, held[(subj, o)], questions=False)
        else:
            b = action_block(rng, subj, o, rng.choice(S2_ACTIONS))
        blocks.append(b)
        total += len(b)
        s, r = rng.sample(ents, 2)
        b = s2_ditransitive_block(rng, s, rng.choice(S2_THINGS)[0], r,
                                  rng.choice(S2_DITRANS))
        blocks.append(b)
        total += len(b)
    rng.shuffle(blocks)
    return BLOCK_SEP.join(blocks).encode("utf-8")


def build_stream(target_bytes, verbmap=None):
    # verbmap: {(subj,obj): verb} of held-out pairs shown declaratively only
    # (no who/what questions), with that fixed verb, so the test verb matches.
    rng = random.Random(SEED)
    held = verbmap or {}
    blocks = []
    total = 0
    builders_static = [naming_block, property_block, spatial_block]
    while total < target_bytes:
        obj = rng.choice(OBJECTS)
        fn = rng.choice(builders_static)
        b = fn(rng, obj)
        blocks.append(b)
        subj, o = rng.choice(ANIMATE), rng.choice(ANIMATE)
        while o == subj:
            o = rng.choice(ANIMATE)
        if (subj, o) in held:
            # held-out pair: declarative-only, with its designated verb
            blocks.append(action_block(rng, subj, o, held[(subj, o)], questions=False))
            blocks.append(action_block(rng, subj, o, held[(subj, o)], questions=False))
        else:
            blocks.append(action_block(rng, subj, o, rng.choice(ACTIONS)))
            blocks.append(action_block(rng, subj, o, rng.choice(ACTIONS)))
        if rng.random() < 0.4:
            blocks.append(narrative_block(rng))
        total += len(b) + 200
    rng.shuffle(blocks)
    return BLOCK_SEP.join(blocks).encode("utf-8")


def write_split(data, out_train, out_val):
    n_val = int(len(data) * VAL_FRACTION)
    cut = len(data) - n_val
    with open(out_train, "wb") as f:
        f.write(data[:cut])
    with open(out_val, "wb") as f:
        f.write(data[cut:])
    return cut, n_val


def main():
    import json
    ap = argparse.ArgumentParser(description="developmental child-concept corpus")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--target-mb", type=int, default=30)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2),
                    help="1 = child concepts; 2 = harder material with headroom (a 16M saturates stage 1)")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="fraction of (subj,obj) pairs shown declarative-only; their who/what tests go to --test-out")
    ap.add_argument("--test-out", default="")
    args = ap.parse_args()
    hmap = s2_holdout_verb_map if args.stage == 2 else holdout_verb_map
    builder = build_stream_s2 if args.stage == 2 else build_stream
    verbmap = hmap(args.holdout_frac) if args.holdout_frac > 0 else None
    data = builder(args.target_mb * 1024 * 1024, verbmap=verbmap)
    tr, va = write_split(data, args.out_train, args.out_val)
    print(f"wrote {args.out_train} ({tr} bytes) + {args.out_val} ({va} bytes)")
    if verbmap and args.test_out:
        items = test_items(verbmap)
        with open(args.test_out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"wrote {args.test_out} ({len(items)} held-out role tests)")


if __name__ == "__main__":
    main()
