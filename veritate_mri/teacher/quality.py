# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pure quality filters for generated samples. length window, json sanity,
#   code fence strip, simhash dedup. no IO, no external deps.
# veritate_mri/teacher/quality.py
# ------------------------------------------------------------------------------------
# Imports:

import json

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_MIN_CHARS = 20
DEFAULT_MAX_CHARS = 8192
DEFAULT_HAMMING_THRESHOLD = 5
_SIMHASH_BITS = 64
_SIMHASH_MASK = (1 << _SIMHASH_BITS) - 1
_FENCE_JSON = "```json"
_FENCE_PLAIN = "```"

# ------------------------------------------------------------------------------------
# Functions

def is_length_ok(text, min_chars, max_chars):
    n = len(text)
    return n >= min_chars and n <= max_chars


def is_json_valid(text):
    try:
        json.loads(text.strip())
        return True
    except (ValueError, TypeError):
        return False


def strip_code_fence(text):
    s = text.strip()
    if s.startswith(_FENCE_JSON):
        s = s[len(_FENCE_JSON):]
    elif s.startswith(_FENCE_PLAIN):
        s = s[len(_FENCE_PLAIN):]
    else:
        return text
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith(_FENCE_PLAIN):
        s = s[:-len(_FENCE_PLAIN)]
    return s.strip()


def simhash64(text):
    tokens = text.split()
    if not tokens:
        return 0
    counts = [0] * _SIMHASH_BITS
    for tok in tokens:
        h = hash(tok) & _SIMHASH_MASK
        for i in range(_SIMHASH_BITS):
            if h & (1 << i):
                counts[i] += 1
            else:
                counts[i] -= 1
    out = 0
    for i in range(_SIMHASH_BITS):
        if counts[i] > 0:
            out |= (1 << i)
    return out


def _hamming(a, b):
    return bin(a ^ b).count("1")


def is_near_dup(h, seen, hamming_threshold=DEFAULT_HAMMING_THRESHOLD):
    for s in seen:
        if _hamming(h, s) <= hamming_threshold:
            return True
    return False
