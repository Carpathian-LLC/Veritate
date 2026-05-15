# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the agent_json starter corpus shipped at carpathian-llc/Veritate-Corpus.
# - Output: schema-strict JSON turns per documentation/corpus/framing.md.
#   Each example is one or more single-line JSON objects (thought / tool_call /
#   observation / answer), terminated by <|endoftext|>.
# - Three example kinds:
#     direct      , one {thought, answer} object
#     two-turn    , {thought} then {answer}
#     calculator  , {thought, tool_call} then {observation} then {answer}
# - Deterministic via fixed PRNG seed so the produced .bin has a stable
#   sha256 across rebuilds.
# veritate_mri/tools/build_agent_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import random
import sys

# ------------------------------------------------------------------------------------
# Constants

EOT = "<|endoftext|>"

DEFAULT_SEED        = 20260514
DEFAULT_VAL_RATIO   = 0.02
DEFAULT_TARGET_MB   = 15   # train size in megabytes


# ------------------------------------------------------------------------------------
# Source: factual prompts (no tool needed)

FACT_PROMPTS = [
    ("Capital of France?",           "France is in western Europe.",     "Paris."),
    ("Capital of Spain?",            "Spain is in southwestern Europe.", "Madrid."),
    ("Capital of Germany?",          "Germany is in central Europe.",    "Berlin."),
    ("Capital of Italy?",            "Italy is in southern Europe.",     "Rome."),
    ("Capital of Japan?",            "Japan is an island nation in East Asia.", "Tokyo."),
    ("Capital of Canada?",           "Canada is in North America.",      "Ottawa."),
    ("Capital of Brazil?",           "Brazil is in South America.",      "Brasilia."),
    ("Capital of Australia?",        "Australia is a continent and country in Oceania.", "Canberra."),
    ("Largest planet?",              "The largest planet orbits the sun beyond the asteroid belt.", "Jupiter."),
    ("Smallest planet?",             "The smallest planet is closest to the sun.", "Mercury."),
    ("Sun's closest star neighbor?", "Proxima Centauri is the nearest stellar neighbor to the sun.", "Proxima Centauri."),
    ("Author of Hamlet?",            "Hamlet is an English play from the early 1600s.", "William Shakespeare."),
    ("Author of 1984?",              "1984 is a dystopian novel from 1949.", "George Orwell."),
    ("Painter of the Mona Lisa?",    "The Mona Lisa is an Italian Renaissance portrait.", "Leonardo da Vinci."),
    ("Inventor of the telephone?",   "The telephone was patented in 1876.", "Alexander Graham Bell."),
    ("Discoverer of penicillin?",    "Penicillin was discovered in 1928.", "Alexander Fleming."),
    ("First US president?",          "The first US president took office in 1789.", "George Washington."),
    ("Year WW2 ended?",              "WW2 ended in the mid 1940s.", "1945."),
    ("Year the Berlin Wall fell?",   "The Berlin Wall fell in the late 1980s.", "1989."),
    ("Longest river?",               "The longest river runs through northeast Africa.", "The Nile."),
    ("Tallest mountain?",            "The tallest mountain is in the Himalayas.", "Mount Everest."),
    ("Largest ocean?",               "The largest ocean covers more than 60 million square miles.", "The Pacific Ocean."),
    ("Continents on Earth?",         "Earth has a fixed standard set of continents.", "Seven."),
    ("Bones in adult human body?",   "An adult skeleton has slightly more than two hundred bones.", "About 206."),
    ("Speed of light in vacuum?",    "Light has a fixed speed in vacuum.", "About 299,792 km/s."),
    ("Chemical symbol for gold?",    "Gold's symbol comes from its Latin name aurum.", "Au."),
    ("Chemical symbol for silver?",  "Silver's symbol comes from its Latin name argentum.", "Ag."),
    ("Chemical symbol for iron?",    "Iron's symbol comes from its Latin name ferrum.", "Fe."),
    ("Water formula?",               "Water has two hydrogen atoms and one oxygen atom.", "H2O."),
    ("DNA stands for?",              "DNA is a biomolecule that carries genetic information.", "Deoxyribonucleic acid."),
    ("CPU stands for?",              "A CPU is the chip in a computer that runs instructions.", "Central processing unit."),
    ("RAM stands for?",              "RAM is the volatile memory a program runs in.", "Random access memory."),
    ("HTTP stands for?",             "HTTP is the protocol web browsers use to fetch pages.", "Hypertext transfer protocol."),
    ("URL stands for?",              "A URL is the address that locates a web resource.", "Uniform resource locator."),
    ("HTML stands for?",             "HTML is the markup language for web pages.", "Hypertext markup language."),
    ("Boiling point of water at sea level in Celsius?", "Water boils at a standard temperature at standard pressure.", "100 degrees Celsius."),
    ("Freezing point of water in Celsius?", "Water freezes at zero degrees on the Celsius scale.", "0 degrees Celsius."),
    ("How many sides does a hexagon have?", "A hexagon is a six-sided polygon.", "Six."),
    ("How many sides does an octagon have?", "An octagon is an eight-sided polygon.", "Eight."),
    ("How many sides does a pentagon have?", "A pentagon is a five-sided polygon.", "Five."),
    ("How many sides does a triangle have?", "A triangle is a three-sided polygon.", "Three."),
    ("How many degrees in a circle?", "A full revolution covers a fixed number of degrees.", "360 degrees."),
    ("How many degrees in a right angle?", "A right angle is a quarter of a full revolution.", "90 degrees."),
    ("How many degrees in a straight angle?", "A straight angle is half a revolution.", "180 degrees."),
]

# ------------------------------------------------------------------------------------
# Functions

# Source: math prompts (use the calculator tool)
def _calc_examples(rng, count):
    out = []
    for _ in range(count):
        kind = rng.choice(["add","sub","mul","div"])
        if kind == "add":
            a = rng.randint(10, 999); b = rng.randint(10, 999)
            expr = f"{a}+{b}"; ans = a + b
            human_q = rng.choice([f"What is {a} plus {b}?", f"Compute {a} + {b}.", f"{a} + {b} = ?"])
        elif kind == "sub":
            a = rng.randint(100, 9999); b = rng.randint(10, a-1)
            expr = f"{a}-{b}"; ans = a - b
            human_q = rng.choice([f"What is {a} minus {b}?", f"Compute {a} - {b}.", f"{a} - {b} = ?"])
        elif kind == "mul":
            a = rng.randint(11, 99); b = rng.randint(11, 99)
            expr = f"{a}*{b}"; ans = a * b
            human_q = rng.choice([f"What is {a} times {b}?", f"Compute {a} * {b}.", f"{a} times {b}?"])
        else:
            b = rng.randint(2, 30); q = rng.randint(2, 50); a = b * q
            expr = f"{a}/{b}"; ans = q
            human_q = rng.choice([f"What is {a} divided by {b}?", f"{a} / {b} = ?"])
        thought = rng.choice([
            f"I need to compute {expr}.",
            f"Let me use the calculator for {expr}.",
            f"I should evaluate {expr} with the calculator tool.",
            f"This is arithmetic; I'll call the calculator.",
        ])
        out.append((human_q, thought, expr, str(ans)))
    return out

# ------------------------------------------------------------------------------------
# Sample writers

def _line(obj):
    """One JSON object on one line. ensure_ascii=False keeps non-ASCII
    intact, but for byte-level training it doesn't matter much, most of
    the corpus is ASCII anyway."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ": ")) + "\n"

def _direct_example(prompt, thought, answer):
    """Single object {thought, answer}."""
    return _line({"thought": thought, "answer": answer}) + EOT + "\n"

def _two_turn_example(prompt, thought, answer):
    """Two objects: {thought} then {answer}."""
    return _line({"thought": thought}) + _line({"answer": answer}) + EOT + "\n"

def _calc_example(human_q, thought, expression, answer_str):
    """Three objects: {thought, tool_call} → {observation} → {answer}."""
    t1 = _line({"thought": thought,
                "tool_call": {"name": "calculator", "args": {"expression": expression}}})
    t2 = _line({"observation": answer_str})
    t3 = _line({"answer": answer_str})
    return t1 + t2 + t3 + EOT + "\n"

# ------------------------------------------------------------------------------------
# Assembly

def _assemble(rng, target_bytes):
    fact_pool = list(FACT_PROMPTS)
    out = []
    written = 0
    while written < target_bytes:
        r = rng.random()
        if r < 0.45:
            # direct
            prompt, thought, answer = rng.choice(fact_pool)
            chunk = _direct_example(prompt, thought, answer).encode("utf-8")
        elif r < 0.75:
            # two-turn
            prompt, thought, answer = rng.choice(fact_pool)
            chunk = _two_turn_example(prompt, thought, answer).encode("utf-8")
        else:
            # calculator
            human_q, thought, expression, answer_str = rng.choice(_calc_examples(rng, 1))
            chunk = _calc_example(human_q, thought, expression, answer_str).encode("utf-8")
        out.append(chunk)
        written += len(chunk)
    return b"".join(out)


# ------------------------------------------------------------------------------------
# Main

def build(out_train, out_val, target_mb, val_ratio, seed):
    rng = random.Random(seed)
    target_bytes = int(target_mb * 1024 * 1024)
    val_bytes_target   = int(target_bytes * val_ratio)
    train_bytes_target = target_bytes - val_bytes_target

    train_data = _assemble(rng, train_bytes_target)
    val_data   = _assemble(rng, val_bytes_target)

    os.makedirs(os.path.dirname(os.path.abspath(out_train)) or ".", exist_ok=True)
    with open(out_train, "wb") as f:
        f.write(train_data)
    with open(out_val, "wb") as f:
        f.write(val_data)

    return {
        "train_path":   out_train,
        "val_path":     out_val,
        "train_bytes":  len(train_data),
        "val_bytes":    len(val_data),
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the agent_json byte-level corpus.")
    ap.add_argument("--out-train",  required=True)
    ap.add_argument("--out-val",    required=True)
    ap.add_argument("--target-mb",  type=int,   default=DEFAULT_TARGET_MB)
    ap.add_argument("--val-ratio",  type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    stats = build(args.out_train, args.out_val, args.target_mb, args.val_ratio, args.seed)
    print(f"agent_json built:")
    print(f"  train: {stats['train_path']}  ({stats['train_bytes']/1e6:.2f} MB)")
    print(f"  val:   {stats['val_path']}    ({stats['val_bytes']/1e6:.2f} MB)")

if __name__ == "__main__":
    sys.exit(main())
