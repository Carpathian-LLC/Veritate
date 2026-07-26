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

# ------------------------------------------------------------------------------------
# Functions

def _cap(s):
    return s[0].upper() + s[1:]


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


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
    decl = f"{_cap(s)} {pres} {o}. {_cap(o)} is {past} by {s}. "
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
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="fraction of (subj,obj) pairs shown declarative-only; their who/what tests go to --test-out")
    ap.add_argument("--test-out", default="")
    args = ap.parse_args()
    verbmap = holdout_verb_map(args.holdout_frac) if args.holdout_frac > 0 else None
    data = build_stream(args.target_mb * 1024 * 1024, verbmap=verbmap)
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
