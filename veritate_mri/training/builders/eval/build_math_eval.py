# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Generate math eval problems for the smartness-meter math axis.
#   Five tiers of 50 problems each as JSONL at
#   veritate_mri/data/eval/grade/math/<tier>.jsonl ({"prompt", "answer"}).
# - Tiers: t1_arith1 (single-digit), t2_arith2 (two-digit), t3_algebra
#   (one-step linear), t4_word (single-op word problem), t5_multi (multi-step).
# - Seeded for reproducibility.
# - Usage:
#     python veritate_mri/training/builders/eval/build_math_eval.py
# veritate_mri/training/builders/eval/build_math_eval.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import random
from pathlib import Path


# ------------------------------------------------------------------------------------
# Constants

SEED = 1729
N_PER_TIER = 50

NAMES_M = ["Tom", "Ben", "Alex", "Sam", "Jack", "Leo", "Max", "Eli"]
NAMES_F = ["Sara", "Mia", "Lily", "Anna", "Zoe", "Eva", "Ivy", "Ada"]
ITEMS   = ["apples", "books", "marbles", "stickers", "cookies", "pencils", "stones", "coins"]


# ------------------------------------------------------------------------------------
# Functions

def t1_arith1(rng):
    out = []
    for _ in range(N_PER_TIER):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.5:
            out.append({"prompt": f"{a} + {b} = ", "answer": str(a + b)})
        else:
            if a < b: a, b = b, a
            out.append({"prompt": f"{a} - {b} = ", "answer": str(a - b)})
    return out


def t2_arith2(rng):
    out = []
    for _ in range(N_PER_TIER):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            out.append({"prompt": f"{a} + {b} = ", "answer": str(a + b)})
        elif op == "-":
            if a < b: a, b = b, a
            out.append({"prompt": f"{a} - {b} = ", "answer": str(a - b)})
        else:
            x, y = rng.randint(2, 12), rng.randint(2, 12)
            out.append({"prompt": f"{x} * {y} = ", "answer": str(x * y)})
    return out


def t3_algebra(rng):
    out = []
    for _ in range(N_PER_TIER):
        x = rng.randint(1, 20)
        b = rng.randint(1, 20)
        op = rng.choice(["+", "-"])
        if op == "+":
            out.append({"prompt": f"x + {b} = {x + b}, x = ", "answer": str(x)})
        else:
            out.append({"prompt": f"x - {b} = {x - b}, x = ", "answer": str(x)})
    return out


def t4_word(rng):
    out = []
    for _ in range(N_PER_TIER):
        name = rng.choice(NAMES_M + NAMES_F)
        item = rng.choice(ITEMS)
        a = rng.randint(3, 20)
        b = rng.randint(1, a)
        if rng.random() < 0.5:
            out.append({
                "prompt": f"{name} had {a} {item}. {name} gave away {b}. How many {item} does {name} have now? ",
                "answer": str(a - b),
            })
        else:
            extra = rng.randint(1, 10)
            out.append({
                "prompt": f"{name} had {a} {item}. {name} got {extra} more. How many {item} does {name} have now? ",
                "answer": str(a + extra),
            })
    return out


def t5_multi(rng):
    out = []
    for _ in range(N_PER_TIER):
        a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(2, 9)
        kind = rng.choice(["addmul", "submul", "addadd"])
        if kind == "addmul":
            ans = (a + b) * c
            out.append({"prompt": f"({a} + {b}) * {c} = ", "answer": str(ans)})
        elif kind == "submul":
            if a < b: a, b = b, a
            ans = (a - b) * c
            out.append({"prompt": f"({a} - {b}) * {c} = ", "answer": str(ans)})
        else:
            d = rng.randint(2, 15)
            ans = a + b + d
            out.append({"prompt": f"{a} + {b} + {d} = ", "answer": str(ans)})
    return out


TIERS = [
    ("t1_arith1", t1_arith1),
    ("t2_arith2", t2_arith2),
    ("t3_algebra", t3_algebra),
    ("t4_word", t4_word),
    ("t5_multi", t5_multi),
]


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "grade_eval" / "math"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name, fn in TIERS:
        problems = fn(rng)
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  wrote: {path.name} ({len(problems)} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
