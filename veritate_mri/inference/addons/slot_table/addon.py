# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - slot table addon. tracks named entities, gendered anchors, and a rolling
#   byte window. biases next-byte logits to suppress doc-boundary collapse,
#   repetition loops, n-gram echoes, and wrong-gender pronoun completions;
#   boosts already-seen named entities at word-start positions.
# - implements the addon contract: reset(), observe(byte_int),
#   bias_logits(logits) -> logits.
# veritate_mri/inference/addons/slot_table/addon.py
# ------------------------------------------------------------------------------------
# Imports:

import re

# ------------------------------------------------------------------------------------
# Constants

NEG_INF                = float("-inf")
WORD_START_TRIGGERS    = (b" ", b"\n", b"\t")
WORD_BREAK_BYTES       = set(b" \n\t.,!?;:'\"()[]")
SENTENCE_END_BYTES     = (b".", b"!", b"?")
NAME_INTRO_PATTERN     = re.compile(r"\bnamed\s+([A-Z][a-z]{2,15})(?=[\s.,!?;:'\"\)])")

GENDER_FEMALE = "female"
GENDER_MALE   = "male"

GENDER_ANCHORS = {
    GENDER_FEMALE: (
        b"girl", b"woman", b"mom", b"mommy", b"mother", b"sister", b"aunt",
        b"queen", b"princess", b"lady", b"daughter", b"grandma", b"grandmother",
        b" she ", b" her ", b" hers ", b" herself ",
    ),
    GENDER_MALE: (
        b"boy", b"man", b"dad", b"daddy", b"father", b"brother", b"uncle",
        b"king", b"prince", b"son", b"grandpa", b"grandfather",
        b" he ", b" his ", b" him ", b" himself ",
    ),
}

GENDERED_PRONOUNS = {
    b"he":      GENDER_MALE,
    b"his":     GENDER_MALE,
    b"him":     GENDER_MALE,
    b"himself": GENDER_MALE,
    b"she":     GENDER_FEMALE,
    b"her":     GENDER_FEMALE,
    b"hers":    GENDER_FEMALE,
    b"herself": GENDER_FEMALE,
}


# ------------------------------------------------------------------------------------
# Functions

class Addon:
    """slot-table inference addon."""

    def __init__(self,
                 block_bytes=(0,),
                 ngram_n=4,
                 ngram_penalty=2.0,
                 rep_penalty=1.10,
                 rep_lookback=64,
                 name_boost=0.5,
                 pronoun_penalty=4.0,
                 window_bytes=256):
        self.block_bytes     = tuple(block_bytes)
        self.ngram_n         = int(ngram_n)
        self.ngram_penalty   = float(ngram_penalty)
        self.rep_penalty     = float(rep_penalty)
        self.rep_lookback    = int(rep_lookback)
        self.name_boost      = float(name_boost)
        self.pronoun_penalty = float(pronoun_penalty)
        self.window_bytes    = int(window_bytes)

        self.window   = bytearray()
        self.names    = []
        self.gender   = None
        self.word_buf = bytearray()

    def reset(self):
        self.window.clear()
        self.names.clear()
        self.gender = None
        self.word_buf.clear()

    def observe(self, byte_int):
        self.window.append(byte_int)
        if len(self.window) > self.window_bytes:
            del self.window[: len(self.window) - self.window_bytes]
        if byte_int in WORD_BREAK_BYTES:
            self.word_buf.clear()
        else:
            self.word_buf.append(byte_int)
        self._refresh_names()
        self._refresh_gender()

    def bias_logits(self, logits):
        out = logits.clone()
        for b in self.block_bytes:
            out[b] = NEG_INF

        if self.rep_penalty > 1.0 and len(self.window) > 0:
            recent = self.window[-self.rep_lookback :]
            seen = set(int(x) for x in recent)
            for b in seen:
                v = out[b]
                if v > 0:
                    out[b] = v / self.rep_penalty
                else:
                    out[b] = v * self.rep_penalty

        if self.ngram_n >= 2 and self.ngram_penalty > 0.0 and len(self.window) >= self.ngram_n - 1:
            suffix = bytes(self.window[-(self.ngram_n - 1):])
            forbidden = self._ngram_continuations(suffix)
            for b in forbidden:
                cur = float(out[b])
                if cur != NEG_INF:
                    out[b] = cur - self.ngram_penalty

        if self.name_boost > 0.0 and self.names and self._at_word_start() and not self._at_sentence_start():
            seen_first_bytes = set()
            for nm in self.names:
                if nm:
                    seen_first_bytes.add(nm[0])
            for b in seen_first_bytes:
                if out[b] != NEG_INF:
                    out[b] = out[b] + self.name_boost

        if self.pronoun_penalty > 0.0 and self.gender is not None:
            for b in self._pronoun_forbidden_next_bytes():
                cur = float(out[b])
                if cur != NEG_INF:
                    out[b] = cur - self.pronoun_penalty

        return out

    def _refresh_names(self):
        try:
            text = self.window.decode("utf-8", errors="replace")
        except Exception:
            return
        for m in NAME_INTRO_PATTERN.finditer(text):
            self._add_name(m.group(1))

    def _refresh_gender(self):
        buf = bytes(self.window).lower()
        best_pos = -1
        best_g   = None
        for g, anchors in GENDER_ANCHORS.items():
            for a in anchors:
                idx = buf.rfind(a)
                if idx > best_pos:
                    best_pos = idx
                    best_g   = g
        self.gender = best_g

    def _add_name(self, name_str):
        b = name_str.encode("utf-8", errors="replace")
        if b and b not in self.names:
            self.names.append(b)

    def _at_word_start(self):
        return len(self.window) > 0 and bytes([self.window[-1]]) in WORD_START_TRIGGERS

    def _at_sentence_start(self):
        if len(self.window) < 2:
            return True
        last2 = bytes(self.window[-2:])
        if last2[-1:] in WORD_START_TRIGGERS and last2[-2:-1] in SENTENCE_END_BYTES:
            return True
        return False

    def _pronoun_forbidden_next_bytes(self):
        prefix = bytes(self.word_buf).lower()
        target = self.gender
        out = set()
        for word, g in GENDERED_PRONOUNS.items():
            if g == target:
                continue
            if not word.startswith(prefix):
                continue
            if len(word) <= len(prefix):
                continue
            next_b = word[len(prefix)]
            collides = False
            for ow, og in GENDERED_PRONOUNS.items():
                if og != target:
                    continue
                if ow.startswith(prefix) and len(ow) > len(prefix) and ow[len(prefix)] == next_b:
                    collides = True
                    break
            if not collides:
                out.add(next_b)
                if 0x61 <= next_b <= 0x7a and len(prefix) == 0:
                    out.add(next_b - 0x20)
        return out

    def _ngram_continuations(self, suffix):
        if len(self.window) < self.ngram_n:
            return ()
        n = self.ngram_n
        out = set()
        buf = bytes(self.window)
        end = len(buf) - n
        i = 0
        while i <= end:
            if buf[i : i + n - 1] == suffix:
                out.add(buf[i + n - 1])
            i += 1
        return out
