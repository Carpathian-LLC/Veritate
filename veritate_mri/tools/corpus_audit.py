# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Scores a ChatML corpus the way the 2026-08-20 audit scored the shipped library:
#   by UNIQUE TURNS and UNIQUE CONTENT BYTES, never by file size. That audit found
#   chat_5gb holding 5.14 GB of bytes over 708 unique user turns (376 KB of real
#   text, one turn repeated 1,298,507 times) while passing every size-based check
#   the platform had. Size is not a quality signal and this module does not report
#   it as one.
# - Thresholds are calibrated against the corpora that survived that audit, with
#   headroom below the weakest passing one: mixed_chat (98.8% unique user turns,
#   99.2% unique content bytes, median assistant turn 265 B, 4.5 artifacts per 1k),
#   cogito (95.9% unique user turns) and recall_curr (4.3 artifacts per 1k) set the
#   binding edges. They are a floor to clear, not a target to hit. sft_idk fails
#   deliberately (67.7% unique user turns) because repeated refusal phrasings are
#   its point -- a corpus with a reason to repeat is a judgement call, not a bug.
# - Streams in chunks so a multi-GB bin costs bounded memory. Turn hashes are
#   16-byte digests, so the peak cost scales with UNIQUE turns, not total turns --
#   which is exactly backwards from the failure mode being detected, and cheap.
# veritate_mri/tools/corpus_audit.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import os
import re
import statistics

# ------------------------------------------------------------------------------------
# Constants

CHUNK_BYTES = 32 * 1024 * 1024
OVERLAP_BYTES = 256 * 1024          # a single turn is assumed to fit in this
DIGEST_BYTES = 16

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

TURN_RE = re.compile(rb"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)

# Measured floors: mixed_chat's profile, rounded down. See module notes.
MIN_UNIQUE_USER_RATIO = 0.95
MIN_UNIQUE_CONTENT_RATIO = 0.85
MIN_MEDIAN_ASSISTANT_BYTES = 200
MAX_ARTIFACTS_PER_1K = 5.0

SHORT_TURN_BYTES = 40

# Each pattern is one way generated text betrays its generator rather than its
# content. Reported per 1,000 assistant turns.
#
# `scope` matters and its absence was a real defect: register tics belong to the
# ASSISTANT voice, so counting them anywhere in the file charged a corpus for user
# turns saying ordinary human things. Measured 2026-08-20 on interview_v1 -- two
# user turns opening "Sure, I'm trying to decide between..." were counted as
# assistant artifacts and failed an otherwise clean corpus. Structural damage
# (mojibake, markup, truncation) is a defect wherever it appears and stays scoped
# to the whole file.
ASSISTANT_SCOPE = "assistant"
ANY_SCOPE = "any"

ARTIFACT_PATTERNS = (
    (ASSISTANT_SCOPE, "ai_disclaimer",  rb"(?i)as an ai (language )?model|i am an ai\b|i'm an ai\b"
                                        rb"|as a language model"),
    (ASSISTANT_SCOPE, "no_personal",    rb"(?i)i (do not|don't) have (personal |the ability |access )"),
    (ASSISTANT_SCOPE, "canned_refusal", rb"(?i)i'm sorry, (but )?(i )?(can(no|')t|am unable)"),
    (ASSISTANT_SCOPE, "filler_opener",  rb"(?i)(^|\n)(sure[!,]|certainly[!,]|of course[!,]|great question)"),
    (ANY_SCOPE, "mojibake",       rb"\xc3\xa2\xe2\x82\xac|\xef\xbf\xbd|&amp;|&quot;|&#\d+;"),
    (ANY_SCOPE, "html_tags",      rb"<(div|p|br|span|a href|img|table)\b"),
    (ANY_SCOPE, "truncated",      rb"\.\.\.\s*<\|im_end\|>"),
    (ANY_SCOPE, "empty_turn",     rb"<\|im_start\|>assistant\n\s*<\|im_end\|>"),
    (ANY_SCOPE, "repeated_char",  rb"(.)\1{15,}"),
    (ANY_SCOPE, "template_leak",  rb"\{\{|\}\}|\[INSERT|<PLACEHOLDER|\[/?INST\]"),
)

# ------------------------------------------------------------------------------------
# Functions

def _ratio(num, den):
    return (num / den) if den else 0.0


def _iter_chunks(path):
    """Yield overlapping byte windows so no turn is lost at a chunk boundary."""
    with open(path, "rb") as f:
        carry = b""
        while True:
            block = f.read(CHUNK_BYTES)
            if not block:
                break
            buf = carry + block
            yield buf
            carry = buf[-OVERLAP_BYTES:] if len(buf) > OVERLAP_BYTES else buf


def audit_bytes(data):
    """Score one in-memory ChatML blob. Same contract as audit_file."""
    return _score(_collect([data]))


def audit_file(path):
    """Score a ChatML .bin on disk. Returns the dict described in _score."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out = _score(_collect(_iter_chunks(path)))
    out["path"] = path
    return add_size_context(out, os.path.getsize(path))


def _collect(blobs):
    """Walk turns once, keeping unique-turn digests, assistant lengths, artifacts.

    Overlapping windows re-present boundary turns, so every turn is keyed by
    (role, digest, absolute offset) for counting -- a turn genuinely repeated
    later in the file has a different offset and still counts as a repeat."""
    seen = {ROLE_USER: set(), ROLE_ASSISTANT: set()}
    totals = {ROLE_USER: 0, ROLE_ASSISTANT: 0}
    counted = set()
    unique_content_bytes = 0
    total_content_bytes = 0
    assistant_lengths = []
    short_assistants = 0
    artifacts = {name: 0 for _, name, _ in ARTIFACT_PATTERNS}
    artifact_spans = set()
    base = 0

    for buf in blobs:
        for m in TURN_RE.finditer(buf):
            role = m.group(1).decode()
            text = m.group(2)
            key = (base + m.start(), role, hashlib.blake2b(text, digest_size=DIGEST_BYTES).digest())
            if key in counted:
                continue
            counted.add(key)
            digest = key[2]
            totals[role] += 1
            total_content_bytes += len(text)
            if digest not in seen[role]:
                seen[role].add(digest)
                unique_content_bytes += len(text)
            if role == ROLE_ASSISTANT:
                assistant_lengths.append(len(text))
                if len(text) < SHORT_TURN_BYTES:
                    short_assistants += 1
        for scope, name, pat in ARTIFACT_PATTERNS:
            if scope == ASSISTANT_SCOPE:
                haystacks = [(m.start(2), m.group(2)) for m in TURN_RE.finditer(buf)
                             if m.group(1) == b"assistant"]
            else:
                haystacks = [(0, buf)]
            for offset, hay in haystacks:
                for m in re.finditer(pat, hay):
                    span = (name, base + offset + m.start())
                    if span not in artifact_spans:
                        artifact_spans.add(span)
                        artifacts[name] += 1
        base += max(0, len(buf) - OVERLAP_BYTES)

    return {
        "unique_user": len(seen[ROLE_USER]),
        "unique_assistant": len(seen[ROLE_ASSISTANT]),
        "total_user": totals[ROLE_USER],
        "total_assistant": totals[ROLE_ASSISTANT],
        "unique_content_bytes": unique_content_bytes,
        "total_content_bytes": total_content_bytes,
        "assistant_lengths": assistant_lengths,
        "short_assistants": short_assistants,
        "artifacts": artifacts,
    }


def _score(acc):
    """Turn raw counts into the ratios, the per-check verdicts, and a pass flag."""
    total_user = acc["total_user"]
    total_asst = acc["total_assistant"]
    lengths = acc["assistant_lengths"]
    artifact_total = sum(acc["artifacts"].values())

    unique_user_ratio = _ratio(acc["unique_user"], total_user)
    unique_asst_ratio = _ratio(acc["unique_assistant"], total_asst)
    median_asst = statistics.median(lengths) if lengths else 0
    artifacts_per_1k = _ratio(artifact_total, total_asst) * 1000

    unique_content_ratio = _ratio(acc["unique_content_bytes"], acc["total_content_bytes"])

    checks = [
        {"id": "unique_content_ratio", "label": "unique content bytes",
         "value": unique_content_ratio, "floor": MIN_UNIQUE_CONTENT_RATIO, "unit": "ratio",
         "passed": unique_content_ratio >= MIN_UNIQUE_CONTENT_RATIO,
         "why": "the share of turn text that is not a copy of text already in the corpus"},
        {"id": "unique_user_turns", "label": "unique user turns",
         "value": unique_user_ratio, "floor": MIN_UNIQUE_USER_RATIO, "unit": "ratio",
         "passed": unique_user_ratio >= MIN_UNIQUE_USER_RATIO,
         "why": "repeated prompts teach the model one conversation many times"},
        {"id": "median_assistant_bytes", "label": "median assistant turn",
         "value": median_asst, "floor": MIN_MEDIAN_ASSISTANT_BYTES, "unit": "bytes",
         "passed": median_asst >= MIN_MEDIAN_ASSISTANT_BYTES,
         "why": "short replies cannot teach sustained conversation"},
        {"id": "artifacts_per_1k", "label": "artifacts per 1k assistant turns",
         "value": artifacts_per_1k, "ceiling": MAX_ARTIFACTS_PER_1K, "unit": "rate",
         "passed": artifacts_per_1k <= MAX_ARTIFACTS_PER_1K,
         "why": "disclaimers, mojibake and template leaks get memorized verbatim"},
    ]

    return {
        "total_user_turns": total_user,
        "total_assistant_turns": total_asst,
        "unique_user_turns": acc["unique_user"],
        "unique_assistant_turns": acc["unique_assistant"],
        "unique_user_ratio": unique_user_ratio,
        "unique_assistant_ratio": unique_asst_ratio,
        "unique_content_bytes": acc["unique_content_bytes"],
        "total_content_bytes": acc["total_content_bytes"],
        "unique_content_ratio": _ratio(acc["unique_content_bytes"], acc["total_content_bytes"]),
        "median_assistant_bytes": median_asst,
        "short_assistant_turns": acc["short_assistants"],
        "artifacts": {k: v for k, v in acc["artifacts"].items() if v},
        "artifacts_per_1k": artifacts_per_1k,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def add_size_context(report, size_bytes):
    """Attach the packed file size and the marker overhead it implies. Size is
    context, never a check: the 2026-08-20 audit exists because size passed
    while the corpus was 99.99% copies."""
    report["size_bytes"] = size_bytes
    report["marker_overhead_bytes"] = max(0, size_bytes - report["total_content_bytes"])
    return report
