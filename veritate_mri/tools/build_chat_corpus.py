# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the chat byte-level corpus shipped at carpathian-llc/Veritate-Corpus.
# - Format: ChatML. Each turn is <|im_start|>{role}\n...<|im_end|>. Examples
#   are separated by <|endoftext|>. Industry-standard frame, byte-level emitted.
# - Source 1: public-domain Project Gutenberg text already cached at
#     trainers/corpus/_pg_cache/*.txt
# - Source 2: hand-written templates (facts, definitions, simple math).
# - Output: framed U/A turns per documentation/corpus/framing.md.
# - Deterministic: a fixed PRNG seed means every run produces the same bytes.
#   This matters because corpus content is shipped as a binary asset on GitHub
#   and we want the sha256 stable across rebuilds.
# - Run:
#     python veritate_mri/tools/build_chat_corpus.py \
#       --out-train  C:/GitHub/Veritate-Corpus/chat_50mb_train.bin \
#       --out-val    C:/GitHub/Veritate-Corpus/chat_50mb_val.bin \
#       --target-mb  50
# veritate_mri/tools/build_chat_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import random
import re
import sys

# ------------------------------------------------------------------------------------
# Constants

PG_CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "trainers", "corpus", "_pg_cache"))

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|endoftext|>"

DEFAULT_SEED        = 20260514
DEFAULT_VAL_RATIO   = 0.02
DEFAULT_TARGET_MB   = 30   # train size in megabytes

# ------------------------------------------------------------------------------------
# Source: hand-written Q/A templates

FACT_QA = [
    ("What is the capital of France?",  "Paris."),
    ("What is the capital of Spain?",   "Madrid."),
    ("What is the capital of Italy?",   "Rome."),
    ("What is the capital of Germany?", "Berlin."),
    ("What is the capital of Japan?",   "Tokyo."),
    ("What is the capital of China?",   "Beijing."),
    ("What is the capital of Brazil?",  "Brasilia."),
    ("What is the capital of Canada?",  "Ottawa."),
    ("What is the capital of Australia?", "Canberra."),
    ("What is the capital of Russia?",  "Moscow."),
    ("What is the capital of India?",   "New Delhi."),
    ("What is the capital of Egypt?",   "Cairo."),
    ("What is the capital of Mexico?",  "Mexico City."),
    ("What is the capital of Argentina?","Buenos Aires."),
    ("What is the capital of South Korea?", "Seoul."),
    ("What is the capital of Thailand?","Bangkok."),
    ("What is the capital of Sweden?",  "Stockholm."),
    ("What is the capital of Norway?",  "Oslo."),
    ("What is the capital of Greece?",  "Athens."),
    ("What is the capital of Portugal?","Lisbon."),
    ("How many continents are there?",  "Seven."),
    ("How many planets are in our solar system?", "Eight."),
    ("What is the largest planet in our solar system?", "Jupiter."),
    ("What is the smallest planet in our solar system?", "Mercury."),
    ("What is the closest star to Earth?", "The Sun."),
    ("How many days are in a year?", "Three hundred sixty-five."),
    ("How many days are in a leap year?", "Three hundred sixty-six."),
    ("How many hours are in a day?", "Twenty-four."),
    ("How many minutes are in an hour?", "Sixty."),
    ("How many seconds are in a minute?", "Sixty."),
    ("What is water made of?", "Two hydrogen atoms and one oxygen atom."),
    ("What is the chemical symbol for water?", "H2O."),
    ("What is the chemical symbol for gold?", "Au."),
    ("What is the chemical symbol for silver?", "Ag."),
    ("What is the chemical symbol for iron?", "Fe."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare."),
    ("Who wrote Hamlet?", "William Shakespeare."),
    ("Who wrote Pride and Prejudice?", "Jane Austen."),
    ("Who wrote Moby Dick?", "Herman Melville."),
    ("Who wrote 1984?", "George Orwell."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo."),
    ("Who composed the Ninth Symphony?", "Ludwig van Beethoven."),
    ("Who invented the telephone?", "Alexander Graham Bell."),
    ("Who discovered penicillin?", "Alexander Fleming."),
    ("Who developed the theory of general relativity?", "Albert Einstein."),
    ("Who was the first president of the United States?", "George Washington."),
    ("In which year did World War II end?", "Nineteen forty-five."),
    ("In which year did the Berlin Wall fall?", "Nineteen eighty-nine."),
    ("What language is spoken in Brazil?", "Portuguese."),
    ("What language is spoken in Egypt?", "Arabic."),
    ("What language is spoken in Mexico?", "Spanish."),
    ("What is the longest river in the world?", "The Nile."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean."),
    ("What is the tallest mountain in the world?", "Mount Everest."),
    ("What is the largest desert in the world?", "The Sahara, if you count hot deserts. Antarctica, if you count cold deserts."),
    ("How many bones are in the adult human body?", "About two hundred and six."),
    ("What organ pumps blood through the body?", "The heart."),
    ("What organ is responsible for filtering blood?", "The kidneys."),
    ("How many chambers does the human heart have?", "Four."),
]

DEFINITION_QA = [
    ("What is a noun?",       "A word that names a person, place, thing, or idea."),
    ("What is a verb?",       "A word that describes an action or state of being."),
    ("What is an adjective?", "A word that describes a noun."),
    ("What is an adverb?",    "A word that modifies a verb, an adjective, or another adverb."),
    ("What is a synonym?",    "A word that has the same or similar meaning as another word."),
    ("What is an antonym?",   "A word that has the opposite meaning of another word."),
    ("What is a metaphor?",   "A figure of speech that describes one thing as if it were another."),
    ("What is a simile?",     "A figure of speech that compares two things using 'like' or 'as'."),
    ("What is photosynthesis?", "The process by which plants use sunlight to make food from carbon dioxide and water."),
    ("What is gravity?",      "A force that pulls objects with mass toward each other."),
    ("What is evaporation?",  "The process by which a liquid turns into a gas."),
    ("What is an atom?",      "The smallest unit of matter that retains the properties of an element."),
    ("What is a molecule?",   "Two or more atoms bonded together."),
    ("What is energy?",       "The capacity to do work or cause change."),
    ("What is a cell?",       "The smallest unit of life that can function independently."),
    ("What is DNA?",          "A molecule that carries the genetic instructions for living organisms."),
    ("What is a democracy?",  "A system of government where citizens choose their leaders by voting."),
    ("What is inflation?",    "A general rise in prices over time."),
    ("What is an ecosystem?", "A community of living organisms interacting with their environment."),
    ("What is climate?",      "The average weather pattern in a place over a long time."),
    ("What is weather?",      "The condition of the atmosphere at a particular time and place."),
    ("What is recycling?",    "The process of converting waste into reusable material."),
    ("What is a fraction?",   "A number that represents part of a whole."),
    ("What is a decimal?",    "A number that uses a point to separate the whole part from the fractional part."),
    ("What is a percentage?", "A number expressed as a fraction of one hundred."),
    ("What is an integer?",   "A whole number, positive or negative, including zero."),
    ("What is a prime number?", "A number greater than one that has no divisors other than one and itself."),
    ("What is the difference between speed and velocity?", "Speed is how fast something moves. Velocity is speed in a specific direction."),
    ("What is the difference between mass and weight?", "Mass is the amount of matter in an object. Weight is the force of gravity on that mass."),
]

WRITING_NUMS = ["one","two","three","four","five","six","seven","eight","nine","ten",
                "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
                "eighteen","nineteen","twenty"]


# ------------------------------------------------------------------------------------
# Functions

def _spell(n):
    if 0 <= n < len(WRITING_NUMS): return WRITING_NUMS[n]
    if n == 100: return "one hundred"
    return str(n)

def _math_qa(rng, count):
    out = []
    for _ in range(count):
        kind = rng.choice(["add","sub","mul","div"])
        if kind == "add":
            a = rng.randint(1, 20); b = rng.randint(1, 20)
            out.append((f"What is {a} plus {b}?", f"{a+b}."))
            out.append((f"{a} + {b} = ?", f"{a+b}."))
        elif kind == "sub":
            a = rng.randint(2, 30); b = rng.randint(1, a-1)
            out.append((f"What is {a} minus {b}?", f"{a-b}."))
            out.append((f"{a} - {b} = ?", f"{a-b}."))
        elif kind == "mul":
            a = rng.randint(2, 12); b = rng.randint(2, 12)
            out.append((f"What is {a} times {b}?", f"{a*b}."))
            out.append((f"{a} * {b} = ?", f"{a*b}."))
        else:
            b = rng.randint(2, 10); q = rng.randint(2, 10); a = b * q
            out.append((f"What is {a} divided by {b}?", f"{q}."))
            out.append((f"{a} / {b} = ?", f"{q}."))
    return out

# Greetings & conversational scaffolding to give the model a sense of dialog
# turn-taking, not just question-answer pairs.

CONVO_OPENERS = [
    ("Hello.", "Hello. How can I help you?"),
    ("Hi there.", "Hi. What's on your mind?"),
    ("Good morning.", "Good morning. How can I assist?"),
    ("Good evening.", "Good evening. What would you like to know?"),
    ("Hey.", "Hey. What's up?"),
    ("Hi, can you help me?", "Of course. What do you need?"),
    ("Thanks!", "You're welcome."),
    ("Thank you.", "Glad I could help."),
    ("Goodbye.", "Goodbye. Take care."),
    ("Bye.", "Bye. Have a good day."),
    ("Are you a real person?", "No. I am a language model."),
    ("Who are you?", "I am a small language model. I answer questions when I can."),
    ("How are you?", "I'm functioning normally. How are you?"),
    ("Tell me a joke.", "Why did the byte cross the road? To get to the other side."),
    ("Are you sure?", "I do my best to be accurate, but I can be wrong."),
]

# ------------------------------------------------------------------------------------
# Source: extract Q/A from Project Gutenberg classics

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

def _strip_gutenberg_header(text):
    """Trim the long Project Gutenberg header and footer if present."""
    start_markers = ["*** START OF THIS PROJECT GUTENBERG", "*** START OF THE PROJECT GUTENBERG"]
    end_markers   = ["*** END OF THIS PROJECT GUTENBERG", "*** END OF THE PROJECT GUTENBERG"]
    for m in start_markers:
        i = text.find(m)
        if i >= 0:
            # advance past the line
            j = text.find("\n", i)
            if j >= 0:
                text = text[j+1:]
            break
    for m in end_markers:
        i = text.find(m)
        if i >= 0:
            text = text[:i]
            break
    return text

def _passages_from_text(text, min_len=200, max_len=700):
    """Split text into passages of a few sentences each. Keeps things short
    so the synthesized Q/A pairs aren't dominated by one author's prose."""
    text = re.sub(r"\s+", " ", text)
    sents = _SENT_RE.split(text)
    out = []
    buf = []
    cur = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if cur + len(s) + 1 > max_len and cur >= min_len:
            out.append(" ".join(buf))
            buf = []
            cur = 0
        buf.append(s)
        cur += len(s) + 1
    if buf and cur >= min_len:
        out.append(" ".join(buf))
    return out

PASSAGE_PROMPTS = [
    "Tell me a passage from a classic book.",
    "Share something from a public-domain story.",
    "Quote me a passage from an old novel.",
    "Give me an excerpt from a classic.",
    "Read me something from an old book.",
]

def _pg_passage_qa(rng, count, max_pg_files=10):
    out = []
    if not os.path.isdir(PG_CACHE_DIR):
        return out
    files = sorted([f for f in os.listdir(PG_CACHE_DIR) if f.endswith(".txt")])
    if not files:
        return out
    rng.shuffle(files)
    files = files[:max_pg_files]
    passages_pool = []
    for fname in files:
        path = os.path.join(PG_CACHE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        text = _strip_gutenberg_header(text)
        passages_pool.extend(_passages_from_text(text))
        if len(passages_pool) >= count * 4:
            break
    rng.shuffle(passages_pool)
    for p in passages_pool[:count]:
        prompt = rng.choice(PASSAGE_PROMPTS)
        out.append((prompt, p))
    return out

# ------------------------------------------------------------------------------------
# Conversation assembly

def _wrap_turn(user_text, asst_text):
    return (f"{IM_START}user\n{user_text}{IM_END}\n"
            f"{IM_START}assistant\n{asst_text}{IM_END}")

def _wrap_conversation(pairs):
    """One conversation = N turn-pairs joined by newlines, ending with EOT."""
    body = "\n".join(_wrap_turn(u, a) for u, a in pairs)
    return body + "\n" + EOT + "\n"

def _assemble(rng, qa_pool, opener_pool, target_bytes):
    """Repeatedly pick 1-3 Q/A pairs plus an optional opener and emit a framed
    conversation, until target_bytes is reached. Returns the concatenated
    UTF-8 bytes."""
    out = []
    written = 0
    while written < target_bytes:
        pairs = []
        # 30% of conversations start with a friendly opener
        if rng.random() < 0.3 and opener_pool:
            pairs.append(rng.choice(opener_pool))
        n_qa = rng.choices([1, 2, 3, 4], weights=[40, 35, 15, 10])[0]
        for _ in range(n_qa):
            pairs.append(rng.choice(qa_pool))
        # 15% close with a thanks/goodbye
        if rng.random() < 0.15:
            pairs.append(rng.choice([
                ("Thanks!", "You're welcome."),
                ("Thank you.", "Glad to help."),
                ("Got it, thanks.", "Anytime."),
            ]))
        chunk = _wrap_conversation(pairs).encode("utf-8")
        out.append(chunk)
        written += len(chunk)
    return b"".join(out)


# ------------------------------------------------------------------------------------
# Main

def build(out_train, out_val, target_mb, val_ratio, seed):
    rng = random.Random(seed)
    qa_pool = list(FACT_QA) + list(DEFINITION_QA)
    qa_pool += _math_qa(rng, 400)
    qa_pool += _pg_passage_qa(rng, 600, max_pg_files=20)

    target_bytes = int(target_mb * 1024 * 1024)
    val_bytes_target   = int(target_bytes * val_ratio)
    train_bytes_target = target_bytes - val_bytes_target

    train_data = _assemble(rng, qa_pool, CONVO_OPENERS, train_bytes_target)
    val_data   = _assemble(rng, qa_pool, CONVO_OPENERS, val_bytes_target)

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
        "qa_pool_size": len(qa_pool),
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the chat byte-level corpus.")
    ap.add_argument("--out-train",  required=True)
    ap.add_argument("--out-val",    required=True)
    ap.add_argument("--target-mb",  type=int,   default=DEFAULT_TARGET_MB)
    ap.add_argument("--val-ratio",  type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    stats = build(args.out_train, args.out_val, args.target_mb, args.val_ratio, args.seed)
    print(f"chat built:")
    print(f"  train: {stats['train_path']}  ({stats['train_bytes']/1e6:.2f} MB)")
    print(f"  val:   {stats['val_path']}    ({stats['val_bytes']/1e6:.2f} MB)")
    print(f"  qa_pool size: {stats['qa_pool_size']}")

if __name__ == "__main__":
    sys.exit(main())
