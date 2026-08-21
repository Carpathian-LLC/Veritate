# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Two-pass conversation generation. The authoring pipeline asks the teacher to
#   WRITE A DIALOGUE -- invent both sides as a script -- and models write scripts
#   like screenplays: measured 2026-08-20 at a 120 B median assistant turn, with
#   nothing past 274 B across two model sizes, two output formats and four batch
#   sizes. Asked the SAME question as itself, the same model answers in 2,380 B.
#   A 20x gap, and the cause is the task, not the model (failures.md 2026-08-20).
# - So this module never asks for a dialogue. Pass 1 writes only the USER turn.
#   Pass 2 sends that turn to the teacher as a real chat request and keeps the
#   genuine reply. Follow-ups repeat both passes with the exchange in context,
#   which also breaks the 7-turn depth ceiling every curated source has
#   (successes.md 2026-08-10).
# - Length is a BLEND by explicit user decision: some replies short and
#   conversational, some full and thorough. It is shaped at ASK time by register
#   (a genuinely short answer beats a truncated long one, which reads as a
#   lecture cut off mid-thought), with sentence-boundary trimming as a ceiling
#   only. Trimming never cuts mid-sentence: that would teach the model to stop
#   mid-thought, which is the exact defect the 2026-08-10 rebuild removed.
# veritate_mri/teacher/interview.py
# ------------------------------------------------------------------------------------
# Imports:

import random
import re
import time

from .client import TeacherError

# ------------------------------------------------------------------------------------
# Constants

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Every teacher call in this mode goes through ask(), which reports both edges of
# the call to an optional watcher: the dashboard's live call feed is built from
# these and nothing else has to be instrumented.
PHASE_START = "start"
PHASE_FIRST = "first"
PHASE_DONE = "done"
PHASE_FAIL = "fail"
PHASE_SALVAGE = "salvage"
CALL_ANSWER = "answer"
CALL_FOLLOWUP = "follow-up"

# A conversation is worth 2*depth-1 teacher calls and every one of them is paid
# for the moment it returns. A failure on the last call used to discard all of
# them; the exchanges already complete are kept instead, which is also what makes
# Stop keep the work that was in flight when it was pressed.
MIN_SALVAGE_TURNS = 2

# The blend. Weights are the share of assistant turns drawn from each register.
# `max_bytes` is a ceiling applied by whole-sentence trimming, not a target --
# a register that comes back under its ceiling is left alone.
REGISTERS = (
    {
        "id": "brief",
        "weight": 0.20,
        "max_bytes": 180,
        "instruction": "Answer in one or two sentences. Be direct and conversational. "
                       "No lists, no headings, no preamble.",
    },
    {
        "id": "normal",
        "weight": 0.55,
        "max_bytes": 480,
        "instruction": "Answer in two to four sentences, the way you would say it out loud "
                       "to someone who asked. No lists, no headings, no preamble.",
    },
    {
        "id": "thorough",
        "weight": 0.25,
        "max_bytes": 1400,
        "instruction": "Answer properly and completely, in flowing prose paragraphs. "
                       "Do not use bullet points, numbered lists, or headings.",
    },
)

# Register is chosen per TURN, so one conversation mixes lengths the way a real
# one does: a short exchange, then a question that earns a fuller answer.
BASE_SYSTEM = (
    "You are a knowledgeable, warm person having a real conversation. You are not writing "
    "documentation. Never open with a restatement of the question, never sign off, and never "
    "mention being an AI. Never begin a reply with Sure, Certainly, Of course, Absolutely, "
    "Great question, or Happy to help. Start with the substance itself."
)

FOLLOWUP_SYSTEM = (
    "You write only the NEXT thing the PERSON says in this conversation. Not the reply to them "
    "-- the next thing they themselves say. It must follow naturally from what was just said: a "
    "follow-up question, a reaction, a new detail about their situation, or a push back. Output "
    "that one line and nothing else. No quotes, no name labels, no explanation."
)

SENTENCE_END_RE = re.compile(rb"[.!?][\"')\]]*(?=\s|$)")
LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+|^\s*\*\*[^*]+\*\*\s*:?\s*$")
DISCLAIMER_RE = re.compile(
    r"(?i)(?:^|(?<=[.!?]\s))[^.!?]*\b(?:as an ai|i'm an ai|i am an ai|language model|"
    r"consult a (?:professional|doctor|lawyer)|i (?:do not|don't) have personal)\b[^.!?]*[.!?]\s*"
)

MIN_KEEP_BYTES = 40

# Measured 2026-08-20: the first interview run opened 8.3% of replies with one of
# these, against a 0.22% baseline in mixed_chat -- about 38x, and a persistent
# register tic rather than sampling noise. Instructed against in BASE_SYSTEM and
# stripped here as a backstop, because instructions alone were not enough for
# reply length either.
FILLER_OPENER_RE = re.compile(
    r"(?i)^(?:sure|certainly|of course|absolutely|definitely|great question|good question|"
    r"happy to help|glad you asked)[!,.]*\s+")

# ------------------------------------------------------------------------------------
# Functions

def pick_register(rng):
    """Draw a length register. The blend is per-turn, not per-conversation."""
    roll = rng.random()
    cum = 0.0
    for reg in REGISTERS:
        cum += reg["weight"]
        if roll <= cum:
            return reg
    return REGISTERS[-1]


def strip_disclaimers(text):
    """Remove whole disclaimer sentences. Whole sentences only -- a partial cut
    leaves a fragment that trains the model to emit fragments."""
    return DISCLAIMER_RE.sub("", text).strip()


def strip_structure(text):
    """Drop list markers and headings, keeping their text as prose. Asked-for
    conversation should not arrive formatted like a manual; when it does anyway,
    the content is still usable but the scaffolding is not."""
    out = HEADING_RE.sub("", text)
    out = LIST_MARKER_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def trim_to_sentence(text, max_bytes):
    """Cut to the last complete sentence that fits in `max_bytes`.

    Never cuts mid-sentence. If not even the first sentence fits, the first
    sentence is kept whole and the ceiling is exceeded on purpose -- an
    over-long whole sentence is a better training example than a truncated one."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text.strip()
    ends = [m.end() for m in SENTENCE_END_RE.finditer(raw)]
    fitting = [e for e in ends if e <= max_bytes]
    if fitting:
        cut = raw[:fitting[-1]]
    elif ends:
        cut = raw[:ends[0]]
    else:
        return text.strip()
    return cut.decode("utf-8", "ignore").strip()


def strip_filler_opener(text):
    """Drop an assistant-register opener, keeping the sentence that follows.

    Re-capitalises what is left: "Sure, you can..." must not become "you can...",
    which would be a lowercase sentence start baked into the corpus."""
    out = FILLER_OPENER_RE.sub("", text or "", count=1).lstrip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def clean_reply(text, max_bytes):
    """Full post-pass on one teacher answer: opener, disclaimers, scaffolding, ceiling."""
    out = strip_filler_opener(text or "")
    out = strip_structure(strip_disclaimers(out))
    out = trim_to_sentence(out, max_bytes)
    return out.strip()


def ask(client, messages, system, temperature, max_tokens, cancel_check=None,
        watch=None, kind=""):
    """One teacher call. `watch` is told what went out, when the first word of the
    reply arrived, what came back and how long it all took -- the only chokepoint
    every interview call passes.

    The first-token mark is what separates SENDING (the request is out, nothing has
    come back) from RECEIVING (the reply is arriving). Providers that cannot stream
    never report it, and such a call reads as sending for its whole life."""
    if watch:
        watch(PHASE_START, kind, messages[-1]["content"], 0.0)
    started = time.perf_counter()

    def first_token():
        watch(PHASE_FIRST, kind, "", (time.perf_counter() - started) * 1000.0)

    try:
        out = client.complete(messages, temperature=temperature, max_tokens=max_tokens,
                              system=system, cancel_check=cancel_check,
                              on_first_token=first_token if watch else None)
    except Exception as e:
        # Reported and re-raised, never swallowed: a call that died after 60 s on
        # a socket timeout is the single most useful row in the panel, and the
        # caller still decides what the failure means.
        if watch:
            watch(PHASE_FAIL, kind, f"{type(e).__name__}: {e}",
                  (time.perf_counter() - started) * 1000.0)
        raise
    if watch:
        watch(PHASE_DONE, kind, out or "", (time.perf_counter() - started) * 1000.0)
    return out


def salvage(watch, turns):
    """Report that a conversation is being kept short rather than thrown away."""
    if watch and len(turns) >= MIN_SALVAGE_TURNS:
        watch(PHASE_SALVAGE, "", "", 0.0)


def next_user_turn(client, turns, temperature, cancel_check=None, watch=None):
    """Pass 1 on a live conversation: write only what the person says next."""
    transcript = "\n".join(
        f"{'PERSON' if t['role'] == ROLE_USER else 'THEM'}: {t['text']}" for t in turns)
    reply = ask(client, [{"role": ROLE_USER, "content": transcript + "\n\nPERSON:"}],
                FOLLOWUP_SYSTEM, temperature, 200, cancel_check, watch, CALL_FOLLOWUP)
    line = (reply or "").strip().split("\n")[0].strip()
    line = re.sub(r'^(?:PERSON|USER)\s*:\s*', "", line, flags=re.I).strip().strip('"')
    return line


def build_conversation(client, opener, depth, seed=0, temperature=0.9, cancel_check=None,
                       watch=None):
    """One conversation: ask the opener, keep the real reply, follow up, repeat.

    `depth` counts assistant turns. Returns the turn list, or None if the very
    first answer came back empty -- a conversation with no reply is not a record.

    A teacher failure part way through ends the conversation at its last complete
    exchange instead of raising. Stop is one of those failures (the client raises
    TeacherCancelled from its cancel check), so pressing Stop keeps the turns that
    were already generated rather than paying for them and discarding them."""
    rng = random.Random(seed)
    turns = [{"role": ROLE_USER, "text": opener}]
    for i in range(depth):
        reg = pick_register(rng)
        history = [{"role": t["role"], "content": t["text"]} for t in turns]
        try:
            raw = ask(client, history, BASE_SYSTEM + " " + reg["instruction"],
                      temperature, 1600, cancel_check, watch, CALL_ANSWER)
        except TeacherError:
            salvage(watch, turns)
            break
        reply = clean_reply(raw, reg["max_bytes"])
        if len(reply.encode("utf-8")) < MIN_KEEP_BYTES:
            break
        turns.append({"role": ROLE_ASSISTANT, "text": reply})
        if i == depth - 1:
            break
        try:
            nxt = next_user_turn(client, turns, temperature, cancel_check, watch)
        except TeacherError:
            salvage(watch, turns)
            break
        if not nxt or len(nxt.encode("utf-8")) < 3:
            break
        turns.append({"role": ROLE_USER, "text": nxt})
    if len(turns) < 2:
        return None
    if turns[-1]["role"] == ROLE_USER:      # never end on an unanswered question
        turns.pop()
    return turns
