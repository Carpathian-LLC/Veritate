# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Quality and cleaning layer for byte-level (vocab=256) pretraining corpora. Owns the
#   "is this text clean enough to train on" concern: unicode normalization, control and
#   zero-width stripping, Gopher/C4-style quality heuristics, PII redaction, and near
#   duplicate detection. Assembly builders consume it; this module has no side effects
#   and writes nothing.
# - Ratio, line, word, and shingle statistics are measured on a bounded prefix of each
#   document so per-document cost stays flat across a 20 GB stream.
# - Deduper stores 64-bit band signatures only, never document text, and stops growing
#   at DEDUP_MAX_SIGNATURES. Pure stdlib, deterministic, no seeded randomness.
# - Records in an assembled corpus are separated by the literal bytes <|endoftext|>
#   (developer_documentation/corpus/framing.md); separators are not passed through here.
# veritate_mri/tools/corpus_filters.py
# ------------------------------------------------------------------------------------
# Imports

import hashlib
import re
import struct
import unicodedata

# ------------------------------------------------------------------------------------
# Constants

NORMAL_FORM = "NFC"
EMPTY = ""
NEWLINE = "\n"
DOUBLE_NEWLINE = "\n\n"
ZERO_RATIO = 0.0

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff\u00ad"
REPLACEMENT_CHAR = "\ufffd"
TERMINAL_PUNCT = (".", "!", "?", "\u3002", "\uff01", "\uff1f")

MIN_DOC_CHARS = 256
MAX_DOC_CHARS = 1_000_000
QUALITY_SAMPLE_CHARS = 8_000

MIN_ALPHA_RATIO = 0.65
MAX_UPPER_RATIO = 0.25
MAX_DIGIT_RATIO = 0.20
MAX_SYMBOL_RATIO = 0.20
MIN_MEAN_LINE_CHARS = 25.0
MAX_DUP_LINE_RATIO = 0.30
MIN_UNIQUE_WORDS = 20
MIN_UNIQUE_WORD_RATIO = 0.25
REPEAT_NGRAM_N = 10
MAX_REPEAT_NGRAM_RATIO = 0.20
MIN_NGRAM_SAMPLES = 2
MAX_REPLACEMENT_CHARS = 3

BOILERPLATE_MARKERS = (
    "lorem ipsum",
    "all rights reserved",
    "click here",
    "accept cookies",
    "cookie policy",
    "cookie preferences",
    "consent to the use of cookies",
)

REASON_TOO_SHORT = "too_short"
REASON_TOO_LONG = "too_long"
REASON_BAD_ENCODING = "bad_encoding"
REASON_LOW_ALPHA = "low_alpha_ratio"
REASON_UPPERCASE = "excessive_uppercase"
REASON_DIGITS = "excessive_digits"
REASON_SYMBOLS = "excessive_symbols"
REASON_SHORT_LINES = "short_mean_line"
REASON_DUP_LINES = "duplicate_lines"
REASON_FEW_WORDS = "too_few_unique_words"
REASON_REPETITION = "low_word_diversity"
REASON_REPEAT_NGRAMS = "repeated_ngrams"
REASON_NO_TERMINAL_PUNCT = "no_terminal_punctuation"
REASON_BOILERPLATE = "boilerplate"

EMAIL_PLACEHOLDER = "[email]"
PHONE_PLACEHOLDER = "[phone]"
NUMBER_PLACEHOLDER = "[number]"

DEDUP_NGRAM = 5
DEDUP_THRESHOLD = 0.8
DEDUP_SCAN_CHARS = 20_000
DEDUP_MAX_SIGNATURES = 40_000_000
MINHASH_SKETCH_SIZE = 64
MIN_BAND_ROWS = 2
SHINGLE_HASH_BYTES = 8
BAND_KEY_BYTES = 8
BYTE_ORDER = "big"
PACK_FMT = "<{}Q"
SHINGLE_JOIN = " "

_ZERO_WIDTH_TABLE = str.maketrans(EMPTY, EMPTY, ZERO_WIDTH_CHARS)
_LINE_BREAK_RE = re.compile(r"\r\n?|\u2028|\u2029")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[^\W_]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")

# ------------------------------------------------------------------------------------
# Functions

def clean_document(text):
    if not text:
        return None
    text = unicodedata.normalize(NORMAL_FORM, text)
    text = text.translate(_ZERO_WIDTH_TABLE)
    text = _LINE_BREAK_RE.sub(NEWLINE, text)
    text = _CONTROL_RE.sub(EMPTY, text)
    text = _TRAILING_WS_RE.sub(EMPTY, text)
    text = _BLANK_RUN_RE.sub(DOUBLE_NEWLINE, text)
    return text.strip() or None


def _char_ratios(text):
    alpha = upper = digit = symbol = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            if ch.isupper():
                upper += 1
        elif ch.isdigit():
            digit += 1
        elif not ch.isspace():
            symbol += 1
    n = len(text)
    return alpha / n, upper / n, digit / n, symbol / n


def _line_ratios(text):
    lines = [line for line in text.split(NEWLINE) if line.strip()]
    mean_length = sum(len(line) for line in lines) / len(lines)
    return mean_length, 1.0 - len(set(lines)) / len(lines)


def _word_ratios(text):
    words = _WORD_RE.findall(text.lower())
    unique = set(words)
    return len(unique), len(unique) / len(words)


def _repeat_ngram_ratio(text):
    words = _WORD_RE.findall(text.lower())
    total = len(words) - REPEAT_NGRAM_N + 1
    if total < MIN_NGRAM_SAMPLES:
        return ZERO_RATIO
    grams = {tuple(words[i:i + REPEAT_NGRAM_N]) for i in range(total)}
    return 1.0 - len(grams) / total


def quality_reject_reason(text):
    length = len(text)
    if length < MIN_DOC_CHARS:
        return REASON_TOO_SHORT
    if length > MAX_DOC_CHARS:
        return REASON_TOO_LONG
    if text.count(REPLACEMENT_CHAR) > MAX_REPLACEMENT_CHARS:
        return REASON_BAD_ENCODING
    sample = text[:QUALITY_SAMPLE_CHARS]
    alpha, upper, digit, symbol = _char_ratios(sample)
    if upper > MAX_UPPER_RATIO:
        return REASON_UPPERCASE
    if digit > MAX_DIGIT_RATIO:
        return REASON_DIGITS
    if symbol > MAX_SYMBOL_RATIO:
        return REASON_SYMBOLS
    if alpha < MIN_ALPHA_RATIO:
        return REASON_LOW_ALPHA
    mean_line, dup_lines = _line_ratios(sample)
    if mean_line < MIN_MEAN_LINE_CHARS:
        return REASON_SHORT_LINES
    if dup_lines > MAX_DUP_LINE_RATIO:
        return REASON_DUP_LINES
    unique_words, unique_ratio = _word_ratios(sample)
    if unique_words < MIN_UNIQUE_WORDS:
        return REASON_FEW_WORDS
    if unique_ratio < MIN_UNIQUE_WORD_RATIO:
        return REASON_REPETITION
    if _repeat_ngram_ratio(sample) > MAX_REPEAT_NGRAM_RATIO:
        return REASON_REPEAT_NGRAMS
    if not any(mark in sample for mark in TERMINAL_PUNCT):
        return REASON_NO_TERMINAL_PUNCT
    lowered = sample.lower()
    if any(mark in lowered for mark in BOILERPLATE_MARKERS):
        return REASON_BOILERPLATE
    return None


def strip_pii(text):
    text = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    text = _PHONE_RE.sub(PHONE_PLACEHOLDER, text)
    return _LONG_DIGITS_RE.sub(NUMBER_PLACEHOLDER, text)


def _band_rows(threshold):
    # Rows per band whose LSH detection point sits closest to threshold.
    best_rows = MIN_BAND_ROWS
    best_error = None
    for rows in range(MIN_BAND_ROWS, MINHASH_SKETCH_SIZE + 1):
        if MINHASH_SKETCH_SIZE % rows:
            continue
        detection = (1.0 / (MINHASH_SKETCH_SIZE // rows)) ** (1.0 / rows)
        error = abs(threshold - detection)
        if best_error is None or error < best_error:
            best_rows, best_error = rows, error
    return best_rows


def _shingle_hashes(text, ngram):
    words = _WORD_RE.findall(text[:DEDUP_SCAN_CHARS].lower())
    return {
        int.from_bytes(
            hashlib.blake2b(
                SHINGLE_JOIN.join(words[i:i + ngram]).encode(),
                digest_size=SHINGLE_HASH_BYTES,
            ).digest(),
            BYTE_ORDER,
        )
        for i in range(len(words) - ngram + 1)
    }


class DocumentDeduper:
    def __init__(self, ngram=DEDUP_NGRAM, threshold=DEDUP_THRESHOLD,
                 max_signatures=DEDUP_MAX_SIGNATURES):
        self.ngram = ngram
        self.rows = _band_rows(threshold)
        self.max_signatures = max_signatures
        self._seen = set()
        self._cached_text = None
        self._cached_keys = ()

    def _band_keys(self, text):
        if text is self._cached_text:
            return self._cached_keys
        sketch = sorted(_shingle_hashes(text, self.ngram))[:MINHASH_SKETCH_SIZE]
        keys = tuple(
            int.from_bytes(
                hashlib.blake2b(
                    struct.pack(PACK_FMT.format(self.rows), *sketch[i:i + self.rows]),
                    digest_size=BAND_KEY_BYTES,
                ).digest(),
                BYTE_ORDER,
            )
            for i in range(0, len(sketch) - self.rows + 1, self.rows)
        )
        self._cached_text, self._cached_keys = text, keys
        return keys

    def is_duplicate(self, text):
        return any(key in self._seen for key in self._band_keys(text))

    def add(self, text):
        if len(self._seen) < self.max_signatures:
            self._seen.update(self._band_keys(text))


class FilterStats:
    def __init__(self):
        self.total = 0
        self.kept = 0
        self.duplicates = 0
        self.rejected = {}

    def note_reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self):
        return {
            "total": self.total,
            "kept": self.kept,
            "duplicates": self.duplicates,
            "rejected": dict(self.rejected),
        }


def filter_stream(docs, deduper=None, stats=None):
    for doc in docs:
        if stats is not None:
            stats.total += 1
        text = clean_document(doc)
        if text is None:
            if stats is not None:
                stats.note_reject(REASON_TOO_SHORT)
            continue
        reason = quality_reject_reason(text)
        if reason is not None:
            if stats is not None:
                stats.note_reject(reason)
            continue
        text = strip_pii(text)
        if deduper is not None:
            if deduper.is_duplicate(text):
                if stats is not None:
                    stats.duplicates += 1
                continue
            deduper.add(text)
        if stats is not None:
            stats.kept += 1
        yield text
