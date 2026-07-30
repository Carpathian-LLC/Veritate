# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the in-pretrain "missing skills" slice: grounded extraction from a
#   context block, novel-string copying, honest misses, and multi-passage
#   selection. These are the four skills the ledger measured as absent
#   (failures.md 2026-07-06, 2026-07-08, 2026-07-11, 2026-07-13): alien-fact
#   extraction scored 1/4 at every checkpoint, so it is a training-dose gap.
# - DOSE: mix a small fraction into PRETRAIN. Concentrated late-phase SFT on a
#   narrow corpus destroys conversation-copy (failures.md 2026-07-06/08).
# - ANTI-TEMPLATE-COLLAPSE (the design constraint): passages, entities, and
#   answer strings come from the REAL source corpus, never a closed hand-written
#   vocabulary. Only QUESTION phrasing is templated. A purely combinatorial
#   generator measured val 0.092 while teaching nothing (held-out role binding
#   17%/0%, failures.md, role-binding entry, 2026-07-25): the
#   model learns surface statistics when the answer set is closed. Answers here
#   are open-ended because they are literal spans of real text.
# - Frame: serving prompt template, veritate_mri/routes/hybrid_routes.py:60, so
#   the training surface equals the inference surface: a "context: " block, then
#   the canonical ChatML turn pair, then the record separator.
# - Deterministic: fixed seed, streaming read, so the output sha256 is stable.
# - Run:
#     python veritate_mri/tools/build_skills_corpus.py \
#       --source trainers/corpus/fineweb_edu_train.bin \
#       --out-train trainers/corpus/skills_v1_train.bin \
#       --out-val   trainers/corpus/skills_v1_val.bin --target-mb 40
# veritate_mri/tools/build_skills_corpus.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import hashlib
import os
import random
import re
import string
from collections import Counter, deque

# ------------------------------------------------------------------------------------
# Constants

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"
EOT_BYTES = EOT.encode("utf-8")
CONTEXT_PREFIX = "context: "

DEFAULT_SEED = 20260725
DEFAULT_VAL_RATIO = 0.03
DEFAULT_TARGET_MB = 40
BYTES_PER_MB = 1024 * 1024
READ_CHUNK = 1 << 20
MAX_RECORD_BYTES = 12000
TXT_SUFFIX = ".txt"
DECODE_ERRORS = "ignore"

FAMILY_SHARES = {
    "grounded_extraction": 0.40,
    "novel_string_recall": 0.25,
    "honest_miss": 0.20,
    "distractor_select": 0.15,
}
MAX_TEMPLATE_SHARE = 0.05
TEMPLATE_MIN_ALLOWANCE = 8

PASSAGE_MIN_CHARS = 260
PASSAGE_MAX_CHARS = 760
PASSAGE_MIN_SENTENCES = 2
SENTENCE_MIN_CHARS = 24
PASSAGES_PER_RECORD = 2
PRINTABLE_MIN_RATIO = 0.97
POOL_SIZE = 64

SPAN_MIN_CHARS = 3
SPAN_MAX_CHARS = 64
NEIGHBOR_WORDS = 4
TOPIC_WORDS = 7
ANCHOR_WORDS = 5
CLOZE_BLANK = "____"
QUOTE_CHAR = "\""
POSITION_BUCKETS = ("start", "middle", "end")
BUCKET_EDGES = (0.34, 0.67)

DISTRACTOR_COUNTS = (2, 2, 3)
DISTRACTOR_JOIN = " "

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")
CHATML_MARKER = "<|"

SPAN_PATTERNS = (
    ("year", re.compile(r"\b(?:1[89]\d{2}|20\d{2})\b")),
    ("number", re.compile(r"\b\d{3,}\b")),
    ("amount", re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")),
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b")),
    ("name", re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")),
    ("quoted phrase", re.compile(r"\"[^\"]{4,60}\"")),
    ("acronym", re.compile(r"\b[A-Z]{2,6}\b")),
)

GROUNDED_QUESTIONS = {
    "g_which": "Which {kind} is stated in the context?",
    "g_direct": "Read the context. What {kind} does it give?",
    "g_copy": "Copy the {kind} exactly as it appears in the text.",
    "g_only": "Using only the context, answer: what {kind} is given?",
    "g_mentions": "The passage mentions a {kind}. What is it?",
    "g_locate": "Find the {kind} in the passage and write it back.",
    "g_quote": "Quote the exact {kind} from the passage above.",
    "g_cloze": "Complete the missing part from the context: \"{cloze}\"",
    "g_fill": "Fill the blank using the context: {cloze}",
    "g_after": "In the context, what words follow \"{before}\"?",
    "g_before": "In the context, what comes just before \"{after}\"?",
    "g_verbatim": "Answer from the passage verbatim: what {kind} appears there?",
}

CARRIER_TEMPLATES = (
    "The {noun} is {surface}.",
    "Records list the {noun} as {surface}.",
    "For this entry the {noun} is {surface}.",
    "The associated {noun} is {surface}.",
    "Filed under {noun} {surface}.",
    "Its {noun} reads {surface}.",
    "Note that the {noun} is {surface}.",
    "The {noun} on file is {surface}.",
)

NOVEL_QUESTIONS = {
    "n_what": "What {noun} does the context give?",
    "n_copy": "Copy the {noun} from the passage exactly.",
    "n_lists": "The passage lists a {noun}. Write it out.",
    "n_report": "Read the context and report the {noun}.",
    "n_written": "Give the {noun} exactly as written above.",
    "n_which": "Which {noun} appears in the text?",
    "n_char": "Repeat the {noun} character for character.",
    "n_only": "From the context only: what is the {noun}?",
    "n_state": "State the {noun} shown in the passage.",
    "n_contains": "The context contains a {noun}. What is it?",
    "n_cloze": "Fill the blank from the context: {cloze}",
}

MISS_PROBES = (
    ("email", re.compile(r"@"), (
        "What email address does the text give?",
        "Which email is listed in the passage?",
        "Read the context: what address should someone write to?",
    )),
    ("year", re.compile(r"\b(?:1[89]\d{2}|20\d{2})\b"), (
        "In what year did this happen, according to the passage?",
        "What year does the context state?",
        "Which year is given above?",
    )),
    ("price", re.compile(r"[$£€]|\bdollars?\b|\bprice[sd]?\b|\bcosts?\b"), (
        "What does it cost, according to the text?",
        "What price does the passage give?",
        "How much money is mentioned in the context?",
    )),
    ("percentage", re.compile(r"%|\bpercent"), (
        "What percentage does the context report?",
        "Which percentage is stated in the passage?",
    )),
    ("phone", re.compile(r"\b\d{3}[-./\s]\d{3,4}\b|\bphone\b|\bcall\b"), (
        "What phone number is listed above?",
        "Which number should someone call, per the context?",
    )),
    ("url", re.compile(r"https?://|www\."), (
        "What website does the passage link to?",
        "Which URL is given in the context?",
    )),
    ("date", re.compile(r"\b(?:January|February|March|April|May|June|July|August|"
                        r"September|October|November|December)\b"), (
        "On what date did this take place, according to the text?",
        "What calendar date does the context give?",
    )),
    ("place", re.compile(r"\b(?:Street|Avenue|Road|Suite|Boulevard)\b"), (
        "What street address does the passage give?",
        "Which street address is listed in the context?",
    )),
    ("author", re.compile(r"\bauthors?\b|\bwritten by\b|\bwrote\b"), (
        "Who wrote this, according to the passage?",
        "Which author does the context credit?",
    )),
    ("entity", re.compile(r"\bzzzz_never_matches\b"), (
        "What does the context say about {entity}?",
        "According to the passage, what is {entity} responsible for?",
        "Where does the text place {entity}?",
    )),
)

MISS_REPLIES = {
    "r_absent": "The passage does not give that.",
    "r_notin": "That is not in the text.",
    "r_silent": "The context says nothing about it.",
    "r_cannot": "I cannot answer from this passage: it never states that.",
    "r_topic": "Not stated here. The text is about {topic}.",
    "r_noguess": "The text does not mention it, so I will not guess.",
    "r_missing": "No, that detail is missing from the context above.",
    "r_nothing": "There is nothing in the passage that answers this.",
    "r_covers": "The passage covers {topic}, but it does not give that.",
    "r_omitted": "That information is absent from the text.",
    "r_find": "I do not find it in the context.",
    "r_leaves": "The text leaves that out.",
    "r_invent": "Nothing in the passage states it, so I will not invent one.",
    "r_guessing": "Not in the context. I would be guessing.",
    "r_never": "The passage never says.",
    "r_provide": "That is not something the text provides.",
}

DISTRACTOR_QUESTIONS = {
    "d_begins": "The context holds more than one passage. In the one that begins \"{anchor}\", what {kind} is given?",
    "d_near": "Only one passage above mentions \"{anchor}\". What {kind} does it state?",
    "d_using": "Using the passage about \"{anchor}\", answer: what {kind} does it give?",
    "d_ignore": "Ignore the passages that do not mention \"{anchor}\". What {kind} appears there?",
    "d_belongs": "Which {kind} belongs to the passage starting \"{anchor}\"?",
    "d_find": "Find \"{anchor}\" in the context and report the {kind} stated with it.",
    "d_several": "Several passages are given. From the one containing \"{anchor}\", copy the {kind}.",
    "d_cloze": "One passage in the context reads \"{cloze}\". What belongs in the blank?",
}

ANSWER_WRAPPERS = (
    "{span}",
    "{span}",
    "{span}",
    "{span}",
    "The context gives {span}.",
    "It says {span}.",
    "From the passage: {span}.",
    "{span}, as stated in the text.",
    "The text says {span}.",
)

DIGITS = string.digits
UPPER = string.ascii_uppercase
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
GIVEN_NAMES = ("Delia", "Marcus", "Ingrid", "Tobias", "Priya", "Nolan", "Yusuf",
               "Beatrix", "Harlan", "Sylvie", "Emeka", "Rosalind", "Caspar", "Noor")
SURNAMES = ("Fontaine", "Achterberg", "Okonjo", "Vasquez", "Lindqvist", "Marsh",
            "Petrossian", "Bramwell", "Nakashima", "Ferreira", "Hollis", "Zeidan")
SURNAME_HYPHEN_CHANCE = 0.25
YEAR_RANGE = (1994, 2031)
DAY_RANGE = (1, 28)

# ------------------------------------------------------------------------------------
# Functions

def _rand_digits(rng, n):
    return "".join(rng.choice(DIGITS) for _ in range(n))


def _rand_upper(rng, n):
    return "".join(rng.choice(UPPER) for _ in range(n))


def _surface_code(rng):
    return f"{_rand_upper(rng, 2)}-{_rand_digits(rng, 4)}-{_rand_upper(rng, 3)}"


def _surface_order(rng):
    return f"ORD-{_rand_digits(rng, 5)}-{_rand_upper(rng, 1)}"


def _surface_amount(rng):
    return f"${_rand_digits(rng, 2)},{_rand_digits(rng, 3)}"


def _surface_units(rng):
    return _rand_digits(rng, 5)


def _surface_date(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(*DAY_RANGE)}, {rng.randint(*YEAR_RANGE)}"


def _surface_name(rng):
    last = rng.choice(SURNAMES)
    if rng.random() < SURNAME_HYPHEN_CHANCE:
        last = f"{last}-{rng.choice(SURNAMES)}"
    return f"{rng.choice(GIVEN_NAMES)} {last}"


def _surface_serial(rng):
    return f"SN{_rand_digits(rng, 6)}"


def _surface_case(rng):
    return f"{_rand_digits(rng, 2)}-CV-{_rand_digits(rng, 5)}"


def _surface_plate(rng):
    return f"{_rand_upper(rng, 3)} {_rand_digits(rng, 4)}"


def _surface_version(rng):
    return f"v{rng.randint(1, 9)}.{_rand_digits(rng, 2)}.{rng.randint(0, 9)}"


def _surface_ref(rng):
    return f"REF/{_rand_digits(rng, 4)}/{_rand_upper(rng, 2)}"


def _surface_batch(rng):
    return f"{_rand_upper(rng, 1)}{_rand_digits(rng, 3)}-{_rand_digits(rng, 3)}"


def _surface_policy(rng):
    return f"{_rand_upper(rng, 4)}{_rand_digits(rng, 5)}"


SURFACE_BUILDERS = (
    ("reference code", _surface_code),
    ("order id", _surface_order),
    ("invoice amount", _surface_amount),
    ("unit count", _surface_units),
    ("effective date", _surface_date),
    ("contact name", _surface_name),
    ("serial number", _surface_serial),
    ("case number", _surface_case),
    ("license plate", _surface_plate),
    ("version tag", _surface_version),
    ("filing reference", _surface_ref),
    ("batch number", _surface_batch),
    ("policy number", _surface_policy),
)

MISS_QUESTIONS = {f"m_{probe}_{i}": text
                  for probe, _, texts in MISS_PROBES
                  for i, text in enumerate(texts)}
MISS_PROBE_KEYS = {probe: tuple(f"m_{probe}_{i}" for i in range(len(texts)))
                   for probe, _, texts in MISS_PROBES}
ALL_TEMPLATES = {}
for _bank in (GROUNDED_QUESTIONS, NOVEL_QUESTIONS, MISS_QUESTIONS,
              MISS_REPLIES, DISTRACTOR_QUESTIONS):
    ALL_TEMPLATES.update(_bank)
TEMPLATE_FIELDS = {k: set(re.findall(r"\{(\w+)\}", v)) for k, v in ALL_TEMPLATES.items()}


def _normalize(text):
    return WHITESPACE_RE.sub(" ", text).strip()


def _is_clean(text):
    if CHATML_MARKER in text:
        return False
    printable = sum(1 for c in text if c in string.printable)
    return printable >= PRINTABLE_MIN_RATIO * len(text)


def _iter_records(path):
    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(READ_CHUNK)
            if not chunk:
                break
            parts = (buf + chunk).split(EOT_BYTES)
            buf = parts.pop()
            for part in parts:
                yield part.decode("utf-8", errors=DECODE_ERRORS)
            while len(buf) > MAX_RECORD_BYTES:
                yield buf[:MAX_RECORD_BYTES].decode("utf-8", errors=DECODE_ERRORS)
                buf = buf[MAX_RECORD_BYTES:]
        if buf:
            yield buf.decode("utf-8", errors=DECODE_ERRORS)


def _source_files(source):
    if os.path.isdir(source):
        return [os.path.join(source, n) for n in sorted(os.listdir(source))
                if n.endswith(TXT_SUFFIX)]
    return [source]


def _passages_from_record(rng, record):
    text = _normalize(record)
    if len(text) < PASSAGE_MIN_CHARS or not _is_clean(text):
        return
    # uppercase start drops mid-word fragments left by chunking a long record
    sents = [s for s in SENTENCE_SPLIT_RE.split(text)
             if len(s) >= SENTENCE_MIN_CHARS and s[0].isupper()]
    if len(sents) < PASSAGE_MIN_SENTENCES:
        return
    i = rng.randrange(len(sents))
    made = 0
    while made < PASSAGES_PER_RECORD and i < len(sents):
        window, size = [], 0
        while i < len(sents) and size + len(sents[i]) <= PASSAGE_MAX_CHARS:
            window.append(sents[i])
            size += len(sents[i]) + 1
            i += 1
            if size >= PASSAGE_MIN_CHARS:
                break
        if size >= PASSAGE_MIN_CHARS and len(window) >= PASSAGE_MIN_SENTENCES:
            yield " ".join(window)
            made += 1
        elif not window:
            return


def _iter_passages(rng, source):
    for path in _source_files(source):
        for record in _iter_records(path):
            yield from _passages_from_record(rng, record)


def _position_bucket(idx, total):
    ratio = idx / total
    if ratio < BUCKET_EDGES[0]:
        return POSITION_BUCKETS[0]
    if ratio < BUCKET_EDGES[1]:
        return POSITION_BUCKETS[1]
    return POSITION_BUCKETS[2]


def _unique_spans(passage):
    spans = []
    for kind, pattern in SPAN_PATTERNS:
        found = pattern.findall(passage)
        if len(found) != 1:
            continue
        span = found[0].strip()
        idx = passage.find(span)
        if idx > 0 and SPAN_MIN_CHARS <= len(span) <= SPAN_MAX_CHARS:
            spans.append((span, kind, idx))
    return spans


def _pick_span(rng, spans, total):
    wanted = rng.choice(POSITION_BUCKETS)
    matching = [s for s in spans if _position_bucket(s[2], total) == wanted]
    return rng.choice(matching or spans)


def _cloze_of(passage, span):
    for sentence in SENTENCE_SPLIT_RE.split(passage):
        if span in sentence:
            return sentence.replace(span, CLOZE_BLANK, 1)
    return ""


def _span_fields(passage, span, kind, idx):
    before = " ".join(passage[:idx].split()[-NEIGHBOR_WORDS:])
    after = " ".join(passage[idx + len(span):].split()[:NEIGHBOR_WORDS])
    return {"kind": kind, "cloze": _cloze_of(passage, span),
            "before": before, "after": after}


def _pick_template(rng, bank, fields, counts, emitted):
    have = {k for k, v in fields.items() if v}
    eligible = [k for k in bank if TEMPLATE_FIELDS[k] <= have]
    if not eligible:
        return None
    allowance = max(TEMPLATE_MIN_ALLOWANCE, int(emitted * MAX_TEMPLATE_SHARE))
    under = [k for k in eligible if counts[k] < allowance]
    return rng.choice(under or eligible)


def _answer(rng, span):
    return rng.choice(ANSWER_WRAPPERS).format(span=span)


def grounded_example(rng, passage, _pool, counts, emitted):
    spans = _unique_spans(passage)
    if not spans:
        return None
    span, kind, idx = _pick_span(rng, spans, len(passage))
    fields = _span_fields(passage, span, kind, idx)
    key = _pick_template(rng, GROUNDED_QUESTIONS, fields, counts, emitted)
    if key is None:
        return None
    return (GROUNDED_QUESTIONS[key].format(**fields), _answer(rng, span),
            passage, key, _position_bucket(idx, len(passage)))


def novel_string_example(rng, passage, _pool, counts, emitted):
    noun, builder = rng.choice(SURFACE_BUILDERS)
    surface = builder(rng)
    if surface in passage:
        return None
    carrier = rng.choice(CARRIER_TEMPLATES).format(noun=noun, surface=surface)
    sents = SENTENCE_SPLIT_RE.split(passage)
    bucket = rng.choice(POSITION_BUCKETS)
    at = {POSITION_BUCKETS[0]: 0,
          POSITION_BUCKETS[1]: len(sents) // 2,
          POSITION_BUCKETS[2]: len(sents)}[bucket]
    context = " ".join([*sents[:at], carrier, *sents[at:]])
    fields = {"noun": noun, "cloze": carrier.replace(surface, CLOZE_BLANK, 1)}
    key = _pick_template(rng, NOVEL_QUESTIONS, fields, counts, emitted)
    if key is None:
        return None
    return (NOVEL_QUESTIONS[key].format(**fields), _answer(rng, surface),
            context, key, bucket)


def honest_miss_example(rng, passage, _pool, counts, emitted):
    open_probes = [p for p in MISS_PROBES if not p[1].search(passage)]
    if not open_probes:
        return None
    probe = rng.choice(open_probes)[0]
    entity = rng.choice((_surface_name, _surface_code))(rng)
    if entity in passage:
        return None
    fields = {"entity": entity, "topic": " ".join(passage.split()[:TOPIC_WORDS])}
    bank = {k: MISS_QUESTIONS[k] for k in MISS_PROBE_KEYS[probe]}
    key = _pick_template(rng, bank, fields, counts, emitted)
    reply_key = _pick_template(rng, MISS_REPLIES, fields, counts, emitted)
    if key is None or reply_key is None:
        return None
    counts[reply_key] += 1
    return (MISS_QUESTIONS[key].format(**fields), MISS_REPLIES[reply_key].format(**fields),
            passage, key, None)


def distractor_example(rng, passage, pool, counts, emitted):
    n_other = rng.choice(DISTRACTOR_COUNTS) - 1
    others = [p for p in rng.sample(list(pool), min(len(pool), n_other)) if p != passage]
    if len(others) < n_other:
        return None
    spans = _unique_spans(passage)
    if not spans:
        return None
    span, kind, idx = _pick_span(rng, spans, len(passage))
    anchor = " ".join(passage.split()[:ANCHOR_WORDS])
    if QUOTE_CHAR in anchor or any(span in o or anchor in o for o in others):
        return None
    fields = {"kind": kind, "anchor": anchor, "cloze": _cloze_of(passage, span)}
    key = _pick_template(rng, DISTRACTOR_QUESTIONS, fields, counts, emitted)
    if key is None:
        return None
    slot = rng.randrange(len(others) + 1)
    parts = [*others[:slot], passage, *others[slot:]]
    return (DISTRACTOR_QUESTIONS[key].format(**fields), _answer(rng, span),
            DISTRACTOR_JOIN.join(parts), key, _position_bucket(idx, len(passage)))


BUILDERS = {
    "grounded_extraction": grounded_example,
    "novel_string_recall": novel_string_example,
    "honest_miss": honest_miss_example,
    "distractor_select": distractor_example,
}


def _neediest_family(families, emitted):
    # one attempt per passage, always the family furthest behind quota: a
    # rejected passage is dropped, not substituted, so realized shares track
    # FAMILY_SHARES instead of the per-family rejection rate
    return min(FAMILY_SHARES, key=lambda n: families[n] - FAMILY_SHARES[n] * emitted)


def _frame(context, user, assistant):
    return (f"{CONTEXT_PREFIX}{context}\n"
            f"{IM_START}user\n{user}{IM_END}\n"
            f"{IM_START}assistant\n{assistant}{IM_END}\n"
            f"{EOT}\n")


def build(source, out_train, out_val, target_mb, val_ratio, seed):
    rng = random.Random(seed)
    pool = deque(maxlen=POOL_SIZE)
    templates, families, positions = Counter(), Counter(), Counter()
    digests = {out_train: hashlib.sha256(), out_val: hashlib.sha256()}
    sizes = {out_train: 0, out_val: 0}
    target = target_mb * BYTES_PER_MB
    with open(out_train, "wb") as ft, open(out_val, "wb") as fv:
        handles = {out_train: ft, out_val: fv}
        for passage in _iter_passages(rng, source):
            pool.append(passage)
            emitted = sum(families.values())
            family = _neediest_family(families, emitted)
            made = BUILDERS[family](rng, passage, pool, templates, emitted)
            if made is None:
                continue
            user, assistant, context, key, position = made
            payload = _frame(context, user, assistant).encode("utf-8")
            path = out_val if rng.random() < val_ratio else out_train
            handles[path].write(payload)
            digests[path].update(payload)
            sizes[path] += len(payload)
            templates[key] += 1
            families[family] += 1
            if position is not None:
                positions[position] += 1
            if sizes[out_train] + sizes[out_val] >= target:
                break
    return {"families": families, "templates": templates, "positions": positions,
            "sizes": sizes, "target": target,
            "sha256": {p: d.hexdigest() for p, d in digests.items()}}


def _report(stats, out_train, out_val):
    total = sum(stats["families"].values())
    written = sum(stats["sizes"].values())
    print(f"examples {total}  train {stats['sizes'][out_train]} B  val {stats['sizes'][out_val]} B")
    if written < stats["target"]:
        print(f"SOURCE EXHAUSTED: {written} B of {stats['target']} B requested; supply more source")
    for path in (out_train, out_val):
        print(f"sha256 {os.path.basename(path)} {stats['sha256'][path]}")
    for label in ("families", "positions"):
        counted = sum(stats[label].values())
        share = "  ".join(f"{k} {v} ({v / counted:.1%})"
                          for k, v in stats[label].most_common())
        print(f"{label}: {share}")
    print("template distribution:")
    for key, n in stats["templates"].most_common():
        print(f"  {key:<16} {n:>7}  {n / total:.2%}")


def main():
    ap = argparse.ArgumentParser(description="missing-skills byte corpus from real passages")
    ap.add_argument("--source", required=True, help=".bin corpus or directory of .txt")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--target-mb", type=int, default=DEFAULT_TARGET_MB)
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    stats = build(args.source, args.out_train, args.out_val,
                  args.target_mb, args.val_ratio, args.seed)
    _report(stats, args.out_train, args.out_val)


if __name__ == "__main__":
    main()
