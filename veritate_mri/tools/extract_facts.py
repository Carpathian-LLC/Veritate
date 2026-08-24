# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the extraction pre-pass of the tell-it-once loop (IDEA 20 E4): raw-transcript
#   sleep is falsified (failures.md 2026-08-21 m2, 0/50) while drilled fact
#   flashcards consolidate (successes.md E4, 45/50), so conversation facts must be
#   mined out of the experience log and rendered through build_fact_sft before a
#   sleep run can bind them. Pipeline: data/experience/*.jsonl -> facts JSON in the
#   build_fact_sft schema {stmt, subj, obj, q_fwd, a_fwd, q_rev, a_rev, kind}.
# - rule-based only (self-contained rule: no external models in the extraction
#   path) and precision-first: a missed fact is a lost memory, a false fact is a
#   lie planted into weights. Every ambiguous decision rejects. Covered relations:
#   residence ("lives"), occupation ("job"), and a guarded copula ("X is a Y")
#   that maps to "job" only when the object noun passes the occupation lexicon.
#   Negations, questions, hedges, and hypotheticals are rejected and reported.
# - facts extract from user turns AND assistant restatements (the m2 echo lesson:
#   assistant bytes are what consolidation trains on). Repeats dedupe; when one
#   subject+relation later appears with a different object the newest wins and
#   the fact carries {"revised": true} (the E2 revised-fact concern).
# - the record "model" field may be a bin filename ("veritate.bin") rather than a
#   model dir name; --model matches case-insensitive on the extension-stripped
#   basename, either side as substring of the other.
# - usage: python -m tools.extract_facts [--days N] [--model NAME]
#          [--out facts.json] [--report]
# veritate_mri/tools/extract_facts.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import base64
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import EXPERIENCE_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IM_END = "<|im_end|>"
TURN_RX = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)

NAME  = r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)"    # personal name, one or two tokens
PLACE = r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)"    # place name, one or two tokens
OBJ   = r"([a-z]+(?: [a-z]+)?)"              # lowercase noun phrase, one or two tokens
FILL  = r"(?:[a-z]+ ){0,2}"                  # up to two lowercase filler words (cues re-checked)
DASH  = r"[—–-]+"

LIVES_RX = [re.compile(p) for p in (
    rf"{NAME} {FILL}lives in {PLACE}",
    rf"{NAME} {FILL}(?:has |had )?moved to {PLACE}",
    rf"{NAME} {FILL}is from {PLACE}",
    rf"{NAME} has a place in {PLACE}",
    rf"{NAME} has settled in {PLACE}",
    rf"{NAME} makes (?:her|his|their) home in {PLACE}",
    rf"{NAME} is in {PLACE} (?:these days|now)",
    rf"{NAME} in {PLACE} {DASH} got it",
    rf"visiting {NAME} over in {PLACE}",
    rf"from {NAME} {DASH} still in {PLACE}",
)]
# (regex, guarded): guarded patterns pass the object through the occupation lexicon
JOB_RX = [(re.compile(p), g) for p, g in (
    (rf"{NAME} {FILL}works as an? {OBJ}",                 False),
    (rf"{NAME} {FILL}took up work as an? {OBJ}",          False),
    (rf"{NAME} {FILL}(?:earns|makes) a living as an? {OBJ}", False),
    (rf"{NAME} is {FILL}an? {OBJ} by trade",              False),
    (rf"{NAME}'s trade is that of an? {OBJ}",             False),
    (rf"ran into {NAME}[^.?!]*?being an? {OBJ}",          False),
    (rf"{NAME}, an? {OBJ} {DASH} got it",                 False),
    (rf"{NAME} is {FILL}an? {OBJ}",                       True),
)]
# whole-utterance patterns that survive the question filter (rhetorical tells)
SPECIAL_RX = [(k, re.compile(p), g) for k, p, g in (
    ("job",   rf"Did I mention {NAME} is {FILL}an? {OBJ}",                 True),
    ("lives", rf"You remember {NAME}\? They(?:'ve| have) moved to {PLACE}", False),
)]

CUES = [(re.compile(p), r) for p, r in (
    (r"n't\b|\bnot\b|\bnever\b|\bno longer\b|\bnobody\b",                          "negated"),
    (r"\bused to\b|\bformerly\b|\bretired\b|\bmoved (?:out|away)\b",               "past/former"),
    (r"\bmight\b|\bmay\b|\bmaybe\b|\bperhaps\b|\bpossibly\b|\bprobably\b"
     r"|\bcould\b|\bwould\b|\bshould\b",                                           "uncertain"),
    (r"\bi think\b|\bi guess\b|\bi bet\b|\bi hear(?:d)?\b|\bapparently\b"
     r"|\bsupposedly\b|\ballegedly\b|\brumou?rs?\b|\bnot sure\b|\bsomeone said\b", "hearsay"),
    (r"\bif\b|\bwhether\b|\bunless\b|\bimagine\b|\bsuppose\b|\bwish(?:es|ed|ing)?\b"
     r"|\bhop(?:e|es|ed|ing)\b|\bwants? to\b|\bplan(?:s|ning)? to\b"
     r"|\bsomeday\b|\bone day\b",                                                  "hypothetical/future"),
)]

NAME_STOPWORDS = frozenset([
    "The", "A", "An", "I", "It", "He", "She", "We", "They", "You", "My", "Your", "His", "Her", "Their", "Our",
    "Its", "This", "That", "These", "Those", "There", "Here", "Who", "What", "Where", "When", "Why", "How",
    "Which", "Did", "Does", "Do", "Is", "Are", "Was", "Were", "Will", "Would", "Can", "Could", "Should",
    "Not", "No", "Yes", "And", "But", "Or", "So", "Well", "Now", "Then", "If", "Oh", "Ah", "Hey", "Hi",
    "Hello", "Ok", "Okay", "Thanks", "Please", "Sorry", "Someone", "Anyone", "Everyone", "Nobody", "Somebody",
    "God", "Sir", "Madam", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December"])
INTERROGATIVE_LEADS = frozenset([
    "who", "what", "where", "when", "why", "how", "which", "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "will", "would", "have", "has", "had", "should", "shall", "am"])

OBJ_TAILS = frozenset(["now", "today", "lately", "currently", "too", "again", "here", "there", "these", "days"])
OCC_SUFFIXES = ("er", "or", "ist", "ian", "eer", "ier", "ess", "smith", "wright",
                "monger", "mason", "herd", "hand", "man", "woman")
OCCUPATION_WORDS = frozenset([
    "chef", "cook", "nurse", "judge", "clerk", "guard", "guide", "pilot", "coach", "medic", "vet", "mason",
    "priest", "maid", "scout", "surgeon", "poet", "merchant", "servant", "accountant"])
NON_OCCUPATIONS = frozenset([
    "man", "woman", "human", "member", "stranger", "newcomer", "teenager", "toddler", "brother", "father",
    "mother", "sister", "daughter", "neighbor", "neighbour", "loner", "owner", "winner", "loser", "partner",
    "believer", "beginner", "foreigner", "outsider", "insider", "villager", "elder", "youngster",
    "vegetarian", "civilian", "wonder", "answer", "matter", "corner", "summer", "winter", "dinner", "letter",
    "number", "order", "offer", "water", "paper", "weather", "monster", "bother", "charter", "chapter",
    "quarter", "shoulder", "disaster"])

# ------------------------------------------------------------------------------------
# Functions


def model_match(rec_model, want):
    """True when the record's model field names the wanted model. The field may be
    a bin filename ("veritate.bin") or a dir name; compare extension-stripped
    lowercase basenames, either side as substring of the other."""
    a = os.path.splitext(os.path.basename(str(rec_model).strip().lower()))[0]
    b = os.path.splitext(os.path.basename(str(want).strip().lower()))[0]
    return bool(a) and bool(b) and (a in b or b in a)


def parse_exchange(rec):
    """One experience-log record -> [(role, text)]. ChatML prompts split into their
    turns (system turns dropped); a markerless prompt is one user turn; the output
    is an assistant turn."""
    try:
        prompt = base64.b64decode(rec["prompt_b64"]).decode("utf-8", "replace")
        output = base64.b64decode(rec["output_b64"]).decode("utf-8", "replace")
    except (KeyError, TypeError, ValueError):
        return []
    turns = [(m.group(1), m.group(2)) for m in TURN_RX.finditer(prompt)]
    if not turns and prompt.strip():
        turns = [("user", prompt)]
    reply = output.split(IM_END)[0]
    if reply.strip():
        turns.append(("assistant", reply))
    return [(r, t) for r, t in turns if r in ("user", "assistant")]


def occupation_noun(phrase):
    """Occupation lexicon guard for the bare copula: last noun must look agentive
    (suffix or whitelist) and not name a plain person category."""
    last = phrase.split()[-1]
    if last in NON_OCCUPATIONS:
        return False
    if last in OCCUPATION_WORDS:
        return True
    return len(last) >= 4 and last.endswith(OCC_SUFFIXES)


def _strip_tails(phrase):
    words = phrase.split()
    while words and words[-1] in OBJ_TAILS:
        words.pop()
    return " ".join(words)


def _cue(text):
    low = text.lower()
    for rx, reason in CUES:
        if rx.search(low):
            return reason
    return None


def _is_question(sent):
    return sent.endswith("?") or sent.split(None, 1)[0].rstrip(",.!").lower() in INTERROGATIVE_LEADS


def _validate(kind, subj, obj, sent, guarded, cands, rejs):
    toks = subj.split()
    while toks and toks[0] in NAME_STOPWORDS:                  # "So Talia ..." -> "Talia"
        toks.pop(0)
    if not toks or any(t in NAME_STOPWORDS for t in toks):
        rejs.append((sent, f"subject guard: {subj}"))
        return
    subj = " ".join(toks)
    if kind == "lives" and any(t in NAME_STOPWORDS for t in obj.split()):
        rejs.append((sent, f"place guard: {obj}"))
        return
    if kind == "job" and guarded and not occupation_noun(obj):
        rejs.append((sent, f"copula object not occupational: {obj}"))
        return
    cands.append({"kind": kind, "subj": subj, "obj": obj, "src": sent})


def scan_text(text):
    """Extract fact candidates from one turn's text. Returns (candidates,
    rejections): candidates are {kind, subj, obj, src}; rejections are
    (sentence, reason) for pattern hits blocked by a filter."""
    cands, rejs = [], []
    work = text
    for kind, rx, guarded in SPECIAL_RX:
        for m in rx.finditer(work):
            span = m.group(0).strip()
            obj = _strip_tails(m.group(2)) if kind == "job" else m.group(2)
            reason = _cue(span)
            if reason:
                rejs.append((span, reason))
            elif obj:
                _validate(kind, m.group(1), obj, span, guarded, cands, rejs)
        work = rx.sub(" ", work)
    for sent in re.split(r"(?<=[.!?])\s+", work):
        sent = sent.strip()
        if not sent:
            continue
        matches = {}
        for rx in LIVES_RX:
            for m in rx.finditer(sent):
                matches[("lives", m.group(1), m.group(2).lower())] = ("lives", m.group(1), m.group(2), False)
        for rx, guarded in JOB_RX:
            for m in rx.finditer(sent):
                obj = _strip_tails(m.group(2))
                if not obj:
                    continue
                key = ("job", m.group(1), obj.lower())
                if key not in matches or matches[key][3]:      # an unguarded hit outranks the copula
                    matches[key] = ("job", m.group(1), obj, guarded)
        if not matches:
            continue
        if _is_question(sent):
            rejs.append((sent, "question"))
            continue
        reason = _cue(sent)
        if reason:
            rejs.append((sent, reason))
            continue
        for kind, subj, obj, guarded in matches.values():
            _validate(kind, subj, obj, sent, guarded, cands, rejs)
    return cands, rejs


def make_fact(kind, subj, obj, revised=False, seen=1):
    """Build one build_fact_sft-schema fact. Extra keys (seen, revised) ride along;
    the renderer ignores them."""
    if kind == "job":
        art = "an" if obj[0] in "aeiou" else "a"
        fact = {"stmt": f"{subj} works as {art} {obj}.", "subj": subj, "obj": obj,
                "q_fwd": f"What does {subj} do for work?", "a_fwd": obj,
                "q_rev": f"Who works as {art} {obj}?", "a_rev": subj, "kind": "job"}
    else:
        fact = {"stmt": f"{subj} lives in {obj}.", "subj": subj, "obj": obj,
                "q_fwd": f"Where does {subj} live?", "a_fwd": obj,
                "q_rev": f"Who lives in {obj}?", "a_rev": subj, "kind": "lives"}
    fact["seen"] = seen
    if revised:
        fact["revised"] = True
    return fact


def extract(records, model=None):
    """Run extraction over experience-log records. Returns (facts, rejections):
    facts in build_fact_sft schema, deduped, newest object per subject+relation
    (older objects mark the fact revised); rejections as {(sentence, reason): n}."""
    seen, rejections = {}, {}
    for rec in records:
        if model and not model_match(rec.get("model", ""), model):
            continue
        ts = rec.get("ts") or 0
        for _role, text in parse_exchange(rec):
            cands, rejs = scan_text(text)
            for sent, reason in rejs:
                rejections[(sent, reason)] = rejections.get((sent, reason), 0) + 1
            for c in cands:
                objs = seen.setdefault((c["subj"].lower(), c["kind"]), {})
                slot = objs.setdefault(c["obj"].lower(), {"subj": c["subj"], "obj": c["obj"], "count": 0, "ts": ts})
                slot["count"] += 1
                slot["ts"] = max(slot["ts"], ts)
    facts = []
    for (_subj_l, kind), objs in sorted(seen.items()):
        best = max(objs.values(), key=lambda s: (s["ts"], s["count"]))
        facts.append(make_fact(kind, best["subj"], best["obj"], revised=len(objs) > 1, seen=best["count"]))
    return facts, rejections


def load_records(days=None):
    """Yield experience-log records, oldest file first; torn lines skip."""
    files = sorted(glob.glob(os.path.join(EXPERIENCE_ROOT, "*.jsonl")))
    if days:
        files = files[-days:]
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def report(facts, rejections):
    """Human-readable extraction report: what was kept, what was blocked and why."""
    lines = [f"extracted {len(facts)} facts:"]
    for f in facts:
        tag = " [revised]" if f.get("revised") else ""
        lines.append(f"  {f['kind']:<5} x{f['seen']:<3} {f['stmt']}{tag}")
    lines.append(f"rejected {sum(rejections.values())} pattern hits ({len(rejections)} distinct):")
    for (sent, reason), n in sorted(rejections.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  [{reason}] x{n}: {sent}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Extract declarative facts from the experience log.")
    ap.add_argument("--days", type=int, default=None, help="only the N most recent log days")
    ap.add_argument("--model", default=None, help="only records served by this model (substring match)")
    ap.add_argument("--out", default="facts.json")
    ap.add_argument("--report", action="store_true", help="print the extraction report")
    args = ap.parse_args()
    facts, rejections = extract(load_records(days=args.days), model=args.model)
    with open(args.out, "w") as f:
        json.dump(facts, f, indent=1)
    print(f"{len(facts)} facts -> {args.out}")
    if args.report:
        print(report(facts, rejections))


if __name__ == "__main__":
    main()
