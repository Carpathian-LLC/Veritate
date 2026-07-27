// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - C port of veritate_mri/addons/slot_table/addon.py. mirrors the rolling-buffer
//   slot table that biases next-byte logits to suppress doc-boundary collapse,
//   repetition loops, n-gram echoes, wrong-gender pronoun completions, and
//   boosts already-seen named entities at word-start positions.
// - port intent: byte-for-byte equivalent biases vs the python addon when both
//   are fed the same rolling window. matched against python in the parity test
//   under docs/c_engine_ternary_moe_tracking.md phase G.
// - vtable factory is addon_slot_table_new(), registered in
//   veritate_engine/src/addons.c::ADDON_FACTORIES.
// veritate_engine/src/addons/slot_table.c
// ------------------------------------------------------------------------------------
// Imports

#include "../addons.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define WINDOW_BYTES         256
#define WORD_BUF_MAX         32
#define MAX_NAMES            16
#define MAX_NAME_LEN         16
#define DEFAULT_NGRAM_N      4
#define NGRAM_PENALTY        2.0f
#define REP_PENALTY          1.10f
#define REP_LOOKBACK         64
#define NAME_BOOST           0.5f
#define PRONOUN_PENALTY      4.0f
#define BLOCK_BYTE_NUL       0x00
#define NEG_INF              (-1.0e30f)

#define GENDER_NONE   0
#define GENDER_FEMALE 1
#define GENDER_MALE   2

// gender anchor lists. mirror python addon's GENDER_ANCHORS exactly.
// matched as substrings in the lowercased window; rightmost match wins.
static const char* GENDER_ANCHORS_FEMALE[] = {
    "girl", "woman", "mom", "mommy", "mother", "sister", "aunt",
    "queen", "princess", "lady", "daughter", "grandma", "grandmother",
    " she ", " her ", " hers ", " herself ",
    NULL
};

static const char* GENDER_ANCHORS_MALE[] = {
    "boy", "man", "dad", "daddy", "father", "brother", "uncle",
    "king", "prince", "son", "grandpa", "grandfather",
    " he ", " his ", " him ", " himself ",
    NULL
};

// gendered pronoun lists (lowercase). pronoun trie is computed on-demand from
// the current word_buf prefix; this is short enough that a flat list is fine.
static const char* PRONOUNS_FEMALE[] = { "she", "her", "hers", "herself", NULL };
static const char* PRONOUNS_MALE[]   = { "he",  "his", "him",  "himself", NULL };


// ------------------------------------------------------------------------------------
// State

typedef struct {
    uint8_t  window[WINDOW_BYTES];
    int32_t  window_len;
    uint8_t  word_buf[WORD_BUF_MAX];
    int32_t  word_buf_len;
    int32_t  gender;
    char     names[MAX_NAMES][MAX_NAME_LEN];
    int32_t  name_count;
} slot_state_t;


// ------------------------------------------------------------------------------------
// Helpers

static inline int is_word_break(uint8_t b) {
    return b == ' ' || b == '\n' || b == '\t' || b == '.' || b == ',' ||
           b == '!' || b == '?' || b == ';'  || b == ':' || b == '\'' ||
           b == '"' || b == '(' || b == ')'  || b == '[' || b == ']';
}

static inline int is_word_start_trigger(uint8_t b) {
    return b == ' ' || b == '\n' || b == '\t';
}

static inline int is_sentence_end_byte(uint8_t b) {
    return b == '.' || b == '!' || b == '?';
}

static inline uint8_t to_lower_ascii(uint8_t b) {
    return (b >= 'A' && b <= 'Z') ? (uint8_t)(b - 'A' + 'a') : b;
}

static int at_word_start(const slot_state_t* s) {
    if (s->window_len <= 0) return 0;
    return is_word_start_trigger(s->window[s->window_len - 1]);
}

static int at_sentence_start(const slot_state_t* s) {
    if (s->window_len < 2) return 1;
    uint8_t prev2 = s->window[s->window_len - 2];
    uint8_t prev1 = s->window[s->window_len - 1];
    return is_word_start_trigger(prev1) && is_sentence_end_byte(prev2);
}

// rfind: position of the last occurrence of needle (length nlen) inside
// haystack (length hlen). returns -1 if not found.
static int32_t bytes_rfind(const uint8_t* haystack, int32_t hlen,
                           const char* needle, int32_t nlen) {
    if (nlen <= 0 || nlen > hlen) return -1;
    for (int32_t i = hlen - nlen; i >= 0; i--) {
        int32_t match = 1;
        for (int32_t j = 0; j < nlen; j++) {
            if (haystack[i + j] != (uint8_t)needle[j]) { match = 0; break; }
        }
        if (match) return i;
    }
    return -1;
}

static void refresh_gender(slot_state_t* s) {
    // build a lowercase mirror of the window for substring scanning. cheap at
    // WINDOW_BYTES=256.
    uint8_t lower[WINDOW_BYTES];
    for (int32_t i = 0; i < s->window_len; i++) {
        lower[i] = to_lower_ascii(s->window[i]);
    }
    int32_t best_pos = -1;
    int32_t best_g   = GENDER_NONE;
    for (const char** a = GENDER_ANCHORS_FEMALE; *a != NULL; a++) {
        int32_t nlen = (int32_t)strlen(*a);
        int32_t pos  = bytes_rfind(lower, s->window_len, *a, nlen);
        if (pos > best_pos) { best_pos = pos; best_g = GENDER_FEMALE; }
    }
    for (const char** a = GENDER_ANCHORS_MALE; *a != NULL; a++) {
        int32_t nlen = (int32_t)strlen(*a);
        int32_t pos  = bytes_rfind(lower, s->window_len, *a, nlen);
        if (pos > best_pos) { best_pos = pos; best_g = GENDER_MALE; }
    }
    s->gender = best_g;
}

static void add_name(slot_state_t* s, const uint8_t* src, int32_t len) {
    if (len <= 0 || len >= MAX_NAME_LEN) return;
    for (int32_t i = 0; i < s->name_count; i++) {
        if ((int32_t)strlen(s->names[i]) == len &&
            memcmp(s->names[i], src, (size_t)len) == 0) return;
    }
    if (s->name_count >= MAX_NAMES) return;
    memcpy(s->names[s->name_count], src, (size_t)len);
    s->names[s->name_count][len] = '\0';
    s->name_count++;
}

// scan the window for the pattern: "named " <Capital> <lower>{2..15} <terminator>.
// terminator is any of [space, period, comma, !, ?, ;, :, ', ", )]. mirrors the
// python NAME_INTRO_PATTERN regex without the regex engine.
static void refresh_names(slot_state_t* s) {
    static const char marker[] = "named ";
    static const int32_t marker_len = (int32_t)(sizeof(marker) - 1);
    if (s->window_len < marker_len + 3) return;

    for (int32_t i = 0; i + marker_len + 3 <= s->window_len; i++) {
        // word-boundary check: marker must follow start-of-window or a word
        // break. lowercased compare for "named ".
        if (i > 0) {
            uint8_t prev = s->window[i - 1];
            if (!(prev == ' ' || prev == '\n' || prev == '\t' ||
                  prev == '.' || prev == ',' || prev == '"' ||
                  prev == '\''|| prev == '(')) continue;
        }
        int32_t k;
        for (k = 0; k < marker_len; k++) {
            if (to_lower_ascii(s->window[i + k]) != (uint8_t)marker[k]) break;
        }
        if (k != marker_len) continue;

        int32_t name_start = i + marker_len;
        if (name_start >= s->window_len) break;
        uint8_t cap = s->window[name_start];
        if (cap < 'A' || cap > 'Z') continue;

        int32_t end = name_start + 1;
        while (end < s->window_len && end - name_start < MAX_NAME_LEN - 1) {
            uint8_t c = s->window[end];
            if (c < 'a' || c > 'z') break;
            end++;
        }
        int32_t name_len = end - name_start;
        if (name_len < 3 || name_len > 16) continue;
        if (end >= s->window_len) continue;  // need a terminator in-window

        uint8_t term = s->window[end];
        if (!(term == ' '  || term == '.'  || term == ','  || term == '!' ||
              term == '?'  || term == ';'  || term == ':'  || term == '\'' ||
              term == '"'  || term == ')'  || term == '\n' || term == '\t')) continue;

        add_name(s, s->window + name_start, name_len);
    }
}

// is the prefix `pref` a strict prefix of `word`?
static int word_starts_with(const char* word, const uint8_t* pref, int32_t pref_len) {
    int32_t wlen = (int32_t)strlen(word);
    if (pref_len >= wlen) return 0;
    for (int32_t i = 0; i < pref_len; i++) {
        if (to_lower_ascii(pref[i]) != (uint8_t)word[i]) return 0;
    }
    return 1;
}

// fill `out` with bytes that, appended to `pref`, would extend toward any
// pronoun in `forbidden_pronouns`, but only if no pronoun in `same_gender_pronouns`
// would also be extended by that same byte (collision: prefix + byte is ambiguous).
// the python addon does the same collision check.
static int32_t pronoun_forbidden_next_bytes(const uint8_t* pref, int32_t pref_len,
                                            const char** forbidden_pronouns,
                                            const char** same_gender_pronouns,
                                            uint8_t* out) {
    int32_t n_out = 0;
    uint8_t added[256];
    memset(added, 0, sizeof(added));
    for (const char** w = forbidden_pronouns; *w != NULL; w++) {
        if (!word_starts_with(*w, pref, pref_len)) continue;
        uint8_t next_b = (uint8_t)(*w)[pref_len];
        // collision: any same-gender word that starts with pref + next_b?
        int collides = 0;
        for (const char** ow = same_gender_pronouns; *ow != NULL; ow++) {
            int32_t owlen = (int32_t)strlen(*ow);
            if (owlen <= pref_len) continue;
            if (!word_starts_with(*ow, pref, pref_len)) continue;
            if ((uint8_t)(*ow)[pref_len] == next_b) { collides = 1; break; }
        }
        if (collides) continue;
        if (!added[next_b]) {
            added[next_b] = 1;
            out[n_out++] = next_b;
        }
        // capitalize at sentence-start position (pref_len == 0)
        if (pref_len == 0 && next_b >= 'a' && next_b <= 'z') {
            uint8_t up = (uint8_t)(next_b - ('a' - 'A'));
            if (!added[up]) { added[up] = 1; out[n_out++] = up; }
        }
    }
    return n_out;
}


// ------------------------------------------------------------------------------------
// Vtable

static void slot_reset(addon_t* self) {
    slot_state_t* s = (slot_state_t*)self->state;
    s->window_len   = 0;
    s->word_buf_len = 0;
    s->gender       = GENDER_NONE;
    s->name_count   = 0;
    memset(s->window,   0, sizeof(s->window));
    memset(s->word_buf, 0, sizeof(s->word_buf));
    for (int32_t i = 0; i < MAX_NAMES; i++) s->names[i][0] = '\0';
}

static void slot_observe(addon_t* self, int byte_int) {
    slot_state_t* s = (slot_state_t*)self->state;
    uint8_t b = (uint8_t)(byte_int & 0xFF);

    // append to window (slide on overflow). 256-byte memmove on overflow is
    // ~50 ns vs a ~1 ms forward pass; negligible.
    if (s->window_len < WINDOW_BYTES) {
        s->window[s->window_len++] = b;
    } else {
        memmove(s->window, s->window + 1, WINDOW_BYTES - 1);
        s->window[WINDOW_BYTES - 1] = b;
    }

    if (is_word_break(b)) {
        s->word_buf_len = 0;
    } else if (s->word_buf_len < WORD_BUF_MAX) {
        s->word_buf[s->word_buf_len++] = b;
    }

    refresh_names(s);
    refresh_gender(s);
}

static void slot_bias_logits(addon_t* self, float* logits, int32_t vocab) {
    slot_state_t* s = (slot_state_t*)self->state;

    // 1. block bytes (NUL kills the doc-boundary collapse)
    if (BLOCK_BYTE_NUL < vocab) logits[BLOCK_BYTE_NUL] = NEG_INF;

    // 2. repetition penalty over the recent lookback window
    if (s->window_len > 0) {
        int32_t lo = s->window_len > REP_LOOKBACK
                   ? s->window_len - REP_LOOKBACK
                   : 0;
        uint8_t seen[256];
        memset(seen, 0, sizeof(seen));
        for (int32_t i = lo; i < s->window_len; i++) {
            seen[s->window[i]] = 1;
        }
        for (int32_t b = 0; b < vocab && b < 256; b++) {
            if (!seen[b]) continue;
            float v = logits[b];
            if (v == NEG_INF) continue;
            if (v > 0.0f) logits[b] = v / REP_PENALTY;
            else          logits[b] = v * REP_PENALTY;
        }
    }

    // 3. n-gram block: forbid bytes that would extend the current
    //    (ngram_n - 1)-byte suffix into an n-gram already present in the window.
    if (DEFAULT_NGRAM_N >= 2 && s->window_len >= DEFAULT_NGRAM_N - 1) {
        int32_t suf_len = DEFAULT_NGRAM_N - 1;
        const uint8_t* suffix = s->window + s->window_len - suf_len;
        uint8_t forbid[256];
        memset(forbid, 0, sizeof(forbid));
        int32_t end = s->window_len - DEFAULT_NGRAM_N;
        for (int32_t i = 0; i <= end; i++) {
            int32_t match = 1;
            for (int32_t j = 0; j < suf_len; j++) {
                if (s->window[i + j] != suffix[j]) { match = 0; break; }
            }
            if (match) forbid[s->window[i + suf_len]] = 1;
        }
        for (int32_t b = 0; b < vocab && b < 256; b++) {
            if (!forbid[b]) continue;
            float v = logits[b];
            if (v == NEG_INF) continue;
            logits[b] = v - NGRAM_PENALTY;
        }
    }

    // 4. name boost: at word-start (not sentence-start), bump the first byte
    //    of any seen named entity. small bias; the model still drives.
    if (NAME_BOOST > 0.0f && s->name_count > 0 &&
        at_word_start(s) && !at_sentence_start(s)) {
        uint8_t boosted[256];
        memset(boosted, 0, sizeof(boosted));
        for (int32_t i = 0; i < s->name_count; i++) {
            if (s->names[i][0] == '\0') continue;
            uint8_t fb = (uint8_t)s->names[i][0];
            if (!boosted[fb]) {
                boosted[fb] = 1;
                if (fb < vocab && logits[fb] != NEG_INF) {
                    logits[fb] = logits[fb] + NAME_BOOST;
                }
            }
        }
    }

    // 5. pronoun forbidden bytes: if a gender anchor is set, suppress bytes
    //    that would extend the current word_buf prefix toward a wrong-gender
    //    pronoun (he/his/him for female anchors, she/her/hers for male).
    if (PRONOUN_PENALTY > 0.0f && s->gender != GENDER_NONE) {
        const char** forbidden;
        const char** same_gender;
        if (s->gender == GENDER_FEMALE) {
            forbidden   = PRONOUNS_MALE;
            same_gender = PRONOUNS_FEMALE;
        } else {
            forbidden   = PRONOUNS_FEMALE;
            same_gender = PRONOUNS_MALE;
        }
        uint8_t forbid_bytes[16];
        int32_t n_forbid = pronoun_forbidden_next_bytes(
            s->word_buf, s->word_buf_len, forbidden, same_gender, forbid_bytes);
        for (int32_t i = 0; i < n_forbid; i++) {
            uint8_t b = forbid_bytes[i];
            if (b >= vocab) continue;
            float v = logits[b];
            if (v == NEG_INF) continue;
            logits[b] = v - PRONOUN_PENALTY;
        }
    }
}

static void slot_destroy(addon_t* self) {
    if (self == NULL) return;
    if (self->state != NULL) { free(self->state); self->state = NULL; }
}

static const addon_vtable_t SLOT_VTABLE = {
    slot_reset,
    slot_observe,
    slot_bias_logits,
    slot_destroy,
};


// ------------------------------------------------------------------------------------
// Factory

addon_t* addon_slot_table_new(void) {
    addon_t* a = (addon_t*)malloc(sizeof(addon_t));
    if (a == NULL) return NULL;
    slot_state_t* s = (slot_state_t*)malloc(sizeof(slot_state_t));
    if (s == NULL) { free(a); return NULL; }
    s->window_len   = 0;
    s->word_buf_len = 0;
    s->gender       = GENDER_NONE;
    s->name_count   = 0;
    memset(s->window,   0, sizeof(s->window));
    memset(s->word_buf, 0, sizeof(s->word_buf));
    for (int32_t i = 0; i < MAX_NAMES; i++) s->names[i][0] = '\0';
    a->id     = "slot_table";
    a->vtable = &SLOT_VTABLE;
    a->state  = s;
    return a;
}
