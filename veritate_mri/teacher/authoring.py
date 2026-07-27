# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - corpus authoring: turns the editable spec at veritate_mri/data/authoring/
#   corpus_spec.json into teacher prompts, then gates every returned record.
# - one teacher call returns a JSONL batch, so volume scales with the call count
#   rather than with a hand-enumerated prompt file.
# - gates: schema, null byte, length, em dash, banned phrase (word boundary),
#   exact dup, repeated opening, near dup, and a rolling distinct-5-gram ratio.
# - RecordGate is called from the SynthJob main thread only; it is not thread safe.
# veritate_mri/teacher/authoring.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, deque

from readers import paths as paths_mod

from .quality import DEFAULT_HAMMING_THRESHOLD, is_near_dup, simhash64, strip_code_fence

# ------------------------------------------------------------------------------------
# Constants

RECORD_KEY = "record"
GENRE_KEY = "genre"
VOICE_KEY = "voice"
TURNS_KEY = "turns"
TEXT_KEY = "text"
ROLE_KEY = "role"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
SCHEMA_TURNS = "turns"
SCHEMA_TEXT = "text"
GENRE_PLACEHOLDER = "<genre>"

EM_DASH_REJECT = "reject"

REJECT_JSON = "invalid json"
REJECT_SCHEMA = "schema mismatch"
REJECT_NULL = "null byte"
REJECT_SHORT = "too short"
REJECT_LONG = "too long"
REJECT_EM_DASH = "em dash"
REJECT_BANNED = "banned phrase"
REJECT_MISSING_MARKER = "missing required marker"
REJECT_TURN_COUNT = "turn count out of range"
REJECT_EXACT_DUP = "exact duplicate"
REJECT_OPENING = "repeated opening"
REJECT_NEAR_DUP = "near duplicate"
REJECT_UNKNOWN_GENRE = "unknown genre"

_WORD_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_BAN_PREFIX = r"(?<![a-z0-9])(?:"
_BAN_SUFFIX = r")(?![a-z0-9])"

# ------------------------------------------------------------------------------------
# Functions

def load_spec(path=None):
    with open(path or paths_mod.authoring_spec_path(), encoding="utf-8") as f:
        return json.load(f)


def save_spec(spec, path=None):
    target = path or paths_mod.authoring_spec_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    os.replace(tmp, target)


def genre_by_id(spec, genre_id):
    for g in spec["genres"]:
        if g["id"] == genre_id:
            return g
    return None


def normalize(text):
    return _SPACE_RE.sub(" ", _WORD_RE.sub(" ", text.lower())).strip()


def record_text(record, schema):
    if schema == SCHEMA_TURNS:
        return "\n".join(t.get(TEXT_KEY, "") for t in record.get(TURNS_KEY, []))
    return record.get(TEXT_KEY, "")


def assistant_text(record, schema):
    if schema == SCHEMA_TURNS:
        return "\n".join(t.get(TEXT_KEY, "") for t in record.get(TURNS_KEY, [])
                         if t.get(ROLE_KEY) == ROLE_ASSISTANT)
    return record.get(TEXT_KEY, "")


def compile_ban_re(phrases):
    return re.compile(_BAN_PREFIX + "|".join(re.escape(p) for p in phrases) + _BAN_SUFFIX)


def distinct_ngram_ratio(word_lists, n):
    seen = set()
    total = 0
    for words in word_lists:
        for i in range(len(words) - n + 1):
            seen.add(hash_words(words[i:i + n]))
            total += 1
    if total == 0:
        return 1.0
    return len(seen) / total


def hash_words(words):
    return hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest()


def plan_calls(spec, genre_ids, target_bytes):
    """Split a byte target across genres by weight, returning calls per genre."""
    gates = spec["gates"]
    est = int(gates["est_bytes_per_record"])
    chosen = [g for g in spec["genres"] if g["id"] in genre_ids]
    total_weight = sum(float(g.get("weight", 0.0)) for g in chosen) or float(len(chosen))
    out = {}
    for g in chosen:
        share = (float(g.get("weight", 0.0)) or 1.0) / total_weight
        per_call = int(g["records_per_call"]) * est
        out[g["id"]] = max(1, math.ceil(target_bytes * share / per_call))
    return out


def build_prompts(spec, calls_per_genre, seed, id_prefix):
    """One prompt per teacher call. Voice pool and situation rotate deterministically.

    The stable half (contract, brief, schema, ban list) goes in `system` as one shared
    string per genre: it keeps a 400k-call plan under a few hundred MB and gives the
    provider a cacheable prefix."""
    template = spec["prompt_template"]
    banned = ", ".join(spec["gates"]["banned_phrases"])
    rng = random.Random(seed)
    prompts = []
    for genre_id in sorted(calls_per_genre):
        g = genre_by_id(spec, genre_id)
        # the schema line ships with a <genre> placeholder; bind it to this call's genre
        # so the model copies the id verbatim instead of inventing a descriptive one
        schema_line = spec["schemas"][g["schema"]]["line"].replace(GENRE_PLACEHOLDER, genre_id)
        system = spec["system_template"].format(
            contract=spec["authoring_contract"], genre=genre_id, brief=g["brief"],
            schema_line=schema_line, banned=banned)
        voices = list(g["voices"])
        situations = list(g["situations"])
        for i in range(int(calls_per_genre[genre_id])):
            pool = rng.sample(voices, min(len(voices), int(g["records_per_call"])))
            prompts.append({"id": f"{id_prefix}{genre_id}_{i:06d}",
                            "category": genre_id,
                            "genre": genre_id,
                            "system": system,
                            "prompt": template.format(
                                voices=", ".join(pool),
                                situation=situations[i % len(situations)],
                                variation=f"{genre_id}-{i:06d}",
                                schema_line=schema_line,
                                n=int(g["records_per_call"]))})
    return prompts


class RecordGate:
    def __init__(self, spec):
        self.spec = spec
        gates = spec["gates"]
        self.gates = gates
        self.ban_re = compile_ban_re(gates["banned_phrases"])
        self.em_dash_chars = tuple(gates["em_dash_chars"])
        self.char_rewrites = dict(gates["char_rewrites"])
        self.em_dash_reject = gates["em_dash_policy"] == EM_DASH_REJECT
        self.ngram_n = int(gates["ngram_n"])
        self.ngram_floor = float(gates["ngram_distinct_floor"])
        self.recompute_every = int(gates["ngram_recompute_every"])
        self.window = deque(maxlen=int(gates["ngram_window_records"]))
        self.seen = set()
        self.openings = Counter()
        self.simhashes = set()
        self.rejects = Counter()
        self.per_genre = Counter()
        self.records = 0
        self.bytes = 0
        self.rewritten = 0
        self.ratio = 1.0

    def seed_from_file(self, samples_path):
        """Rebuild dedup state from an existing job so a resume keeps the gates honest."""
        if not os.path.isfile(samples_path):
            return
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rec = row.get(RECORD_KEY)
                if not isinstance(rec, dict):
                    continue
                g = genre_by_id(self.spec, rec.get(GENRE_KEY, ""))
                if g is None:
                    continue
                self._accept(rec, g["schema"], rec.get(GENRE_KEY, ""))

    def stats(self):
        return {
            "records": self.records,
            "bytes": self.bytes,
            "em_dash_rewritten": self.rewritten,
            "ngram_ratio": round(self.ratio, 4),
            "ngram_floor": self.ngram_floor,
            "ngram_below_floor": self.ratio < self.ngram_floor,
            "rejects": dict(self.rejects.most_common()),
            "per_genre": dict(self.per_genre),
        }

    def __call__(self, prompt, response):
        genre_id = prompt["genre"]
        g = genre_by_id(self.spec, genre_id)
        schema_name = g["schema"]
        schema = self.spec["schemas"][schema_name]
        kept, why = [], []
        for line in strip_code_fence(response).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                why.append(REJECT_JSON)
                self.rejects[REJECT_JSON] += 1
                continue
            # genre is a property of the call, not of the content: stamp it rather
            # than reject a model that wrote good text under its own genre label
            if isinstance(rec, dict):
                rec[GENRE_KEY] = genre_id
            reason = self._check(rec, g, schema_name, schema, genre_id)
            if reason is not None:
                why.append(reason)
                self.rejects[reason] += 1
                continue
            self._accept(rec, schema_name, genre_id)
            kept.append(rec)
        return kept, why

    def _check(self, rec, g, schema_name, schema, genre_id):
        reason = _check_schema(rec, schema_name, schema, genre_id)
        if reason is not None:
            return reason
        if schema_name == SCHEMA_TURNS:
            n = len(rec[TURNS_KEY])
            if n < int(g["min_turns"]) or n > int(g["max_turns"]):
                return REJECT_TURN_COUNT
            marker = g.get("require_in_first_user", "")
            if marker and marker not in rec[TURNS_KEY][0][TEXT_KEY]:
                return REJECT_MISSING_MARKER
        txt = record_text(rec, schema_name)
        if "\x00" in txt:
            return REJECT_NULL
        if len(txt) < int(self.gates["min_chars"]):
            return REJECT_SHORT
        if len(txt) > int(self.gates["max_chars"]):
            return REJECT_LONG
        if any(c in txt for c in self.em_dash_chars):
            if self.em_dash_reject:
                return REJECT_EM_DASH
            self._rewrite(rec, schema_name)
            self.rewritten += 1
            txt = record_text(rec, schema_name)
        if self.ban_re.search(assistant_text(rec, schema_name).lower()):
            return REJECT_BANNED
        norm = normalize(txt)
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if digest in self.seen:
            return REJECT_EXACT_DUP
        if self.openings[norm[:int(self.gates["opening_prefix_chars"])]] >= int(self.gates["opening_cap"]):
            return REJECT_OPENING
        if is_near_dup(simhash64(txt), self.simhashes,
                       int(self.gates.get("near_dup_hamming", DEFAULT_HAMMING_THRESHOLD))):
            return REJECT_NEAR_DUP
        return None

    def _rewrite(self, rec, schema_name):
        if schema_name == SCHEMA_TURNS:
            for t in rec[TURNS_KEY]:
                t[TEXT_KEY] = _apply_rewrites(t[TEXT_KEY], self.char_rewrites)
        else:
            rec[TEXT_KEY] = _apply_rewrites(rec[TEXT_KEY], self.char_rewrites)

    def _accept(self, rec, schema_name, genre_id):
        txt = record_text(rec, schema_name)
        norm = normalize(txt)
        self.seen.add(hashlib.sha1(norm.encode("utf-8")).hexdigest())
        self.openings[norm[:int(self.gates["opening_prefix_chars"])]] += 1
        self.simhashes.add(simhash64(txt))
        self.window.append(norm.split())
        self.records += 1
        self.per_genre[genre_id] += 1
        self.bytes += len(txt.encode("utf-8"))
        if self.records % self.recompute_every == 0:
            self.ratio = distinct_ngram_ratio(self.window, self.ngram_n)


def _apply_rewrites(text, rewrites):
    for src, dst in rewrites.items():
        text = text.replace(src, dst)
    return text


def _check_schema(rec, schema_name, schema, genre_id):
    if not isinstance(rec, dict) or set(rec.keys()) != set(schema["keys"]):
        return REJECT_SCHEMA
    if rec.get(GENRE_KEY) != genre_id:
        return REJECT_SCHEMA
    if not isinstance(rec.get(VOICE_KEY), str) or not rec[VOICE_KEY].strip():
        return REJECT_SCHEMA
    if schema_name == SCHEMA_TEXT:
        if not isinstance(rec.get(TEXT_KEY), str) or not rec[TEXT_KEY].strip():
            return REJECT_SCHEMA
        return None
    turns = rec.get(TURNS_KEY)
    if not isinstance(turns, list) or len(turns) < 2:
        return REJECT_SCHEMA
    roles = schema["roles"]
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or set(t.keys()) != set(schema["turn_keys"]):
            return REJECT_SCHEMA
        if t[ROLE_KEY] != roles[i % len(roles)]:
            return REJECT_SCHEMA
        if not isinstance(t.get(TEXT_KEY), str) or not t[TEXT_KEY].strip():
            return REJECT_SCHEMA
    return None
