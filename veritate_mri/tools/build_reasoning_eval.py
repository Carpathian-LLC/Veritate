"""Generate reasoning eval problems for the smartness-meter reasoning axis.

Four type-tiers, 50 each, templated where possible. Output written to
veritate_mri/grade_eval/reasoning/<tier>.jsonl as {"prompt", "answer"}.
The probe argmax-decodes the model's continuation and string-matches.

Tier categories (NOT difficulty levels — they probe different cognitive moves):
    recall          fact completion (capitals, basic relations)
    pattern         analogy completion (cat:kitten :: dog:?)
    deduction1      one-step syllogism (All A are B; X is A. Therefore X is ?)
    deduction_n     multi-step ordering / transitive (A>B, B>C => A>C)

Usage:
    python veritate_mri/tools/build_reasoning_eval.py
"""

import json
import random
from pathlib import Path

SEED = 8675
N_PER_TIER = 50

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
    ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("India", "Delhi"),
    ("Greece", "Athens"), ("Turkey", "Ankara"), ("Mexico", "Mexico"), ("Portugal", "Lisbon"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Poland", "Warsaw"),
]
COLORS_OF = [
    ("the sky on a clear day", "blue"), ("fresh grass", "green"), ("a ripe banana", "yellow"),
    ("snow", "white"), ("coal", "black"), ("the sun at noon", "yellow"),
    ("a stop sign", "red"), ("a ripe tomato", "red"), ("the ocean", "blue"),
]
ANIMAL_FACTS = [
    ("Dogs are mammals. So a dog is a", "mammal"),
    ("Sparrows are birds. So a sparrow is a", "bird"),
    ("Salmon are fish. So a salmon is a", "fish"),
    ("Bees are insects. So a bee is an", "insect"),
    ("Cows are mammals. So a cow is a", "mammal"),
]


def recall(rng):
    out = []
    pool = []
    for c, cap in CAPITALS:
        pool.append({"prompt": f"The capital of {c} is ", "answer": cap})
    for thing, color in COLORS_OF:
        pool.append({"prompt": f"The color of {thing} is ", "answer": color})
    for fact, ans in ANIMAL_FACTS:
        pool.append({"prompt": f"{fact} ", "answer": ans})
    # numeric-recall fillers to reach N
    for i in range(50):
        n = rng.randint(2, 9)
        pool.append({"prompt": f"There are {n} days in {n} day{'s' if n > 1 else ''}, so the answer is ", "answer": str(n)})
    rng.shuffle(pool)
    out.extend(pool[:N_PER_TIER])
    for i, p in enumerate(out):
        p["type"] = "recall"
    return out


ANALOGIES = [
    ("cat", "kitten", "dog", "puppy"),
    ("cow", "calf",   "horse", "foal"),
    ("hen", "chick",  "duck", "duckling"),
    ("man", "boy",    "woman", "girl"),
    ("king", "queen", "prince", "princess"),
    ("hot", "cold",   "up", "down"),
    ("big", "small",  "tall", "short"),
    ("happy", "sad",  "fast", "slow"),
    ("day", "night",  "summer", "winter"),
    ("teacher", "student", "doctor", "patient"),
]


def pattern(rng):
    out = []
    for _ in range(N_PER_TIER):
        a, b, c, d = rng.choice(ANALOGIES)
        out.append({"prompt": f"{a} : {b} :: {c} : ", "answer": d, "type": "pattern"})
    return out


SYLL_TEMPLATES = [
    ("All {A} are {B}. {X} is a {A}. Therefore {X} is a", "{B}"),
    ("Every {A} is also a {B}. {X} is a {A}. So {X} is a", "{B}"),
]
SYLL_FILLS = [
    ("dogs", "mammal", "Rex"), ("birds", "animal", "Sparrow"), ("squares", "shape", "This"),
    ("apples", "fruit", "It"), ("triangles", "shape", "This"), ("roses", "flower", "It"),
    ("cars", "vehicle", "It"), ("oaks", "tree", "It"), ("salmon", "fish", "It"),
    ("violins", "instrument", "It"),
]


def deduction1(rng):
    out = []
    for _ in range(N_PER_TIER):
        tpl_p, tpl_a = rng.choice(SYLL_TEMPLATES)
        A_plur, B, X = rng.choice(SYLL_FILLS)
        # "All dogs are mammal" works; keep B singular for clean answer.
        prompt = tpl_p.format(A=A_plur, B=B, X=X) + " "
        ans = tpl_a.format(B=B)
        out.append({"prompt": prompt, "answer": ans, "type": "deduction1"})
    return out


NAMES = ["Alex", "Ben", "Carl", "Dan", "Eve", "Finn", "Gail", "Hugo", "Ivy", "Jay"]


def deduction_n(rng):
    out = []
    for _ in range(N_PER_TIER):
        kind = rng.choice(["taller", "older", "faster"])
        a, b, c = rng.sample(NAMES, 3)
        rel = {"taller": "taller than", "older": "older than", "faster": "faster than"}[kind]
        ask = {"taller": "shortest", "older": "youngest", "faster": "slowest"}[kind]
        # a > b > c, so c is the answer
        prompt = f"{a} is {rel} {b}. {b} is {rel} {c}. The {ask} is "
        out.append({"prompt": prompt, "answer": c, "type": "deduction_n"})
    return out


TIERS = [
    ("recall",      recall),
    ("pattern",     pattern),
    ("deduction1",  deduction1),
    ("deduction_n", deduction_n),
]


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "grade_eval" / "reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name, fn in TIERS:
        items = fn(rng)
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in items:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  wrote: {path.name} ({len(items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
