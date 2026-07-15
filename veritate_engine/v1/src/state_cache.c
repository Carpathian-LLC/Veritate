// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - persistent v13 prompt/state cache. the resumable per-request state is the six
//   hybrid_t fields hybrid_reset touches (kv_k, kv_v, rec_state, conv_ring,
//   slot_count, pos) plus the post-step logits + final hidden. snapshot at prefix
//   length L, restore on the longest present prefix so prefill resumes mid-stream.
// - key = twin rolling hashes over model_id + tokens (prefix-consistent, so an
//   extended prompt hits the base snapshot). filename is the primary hash hex;
//   the header carries the secondary hash + shape guards as a collision guard.
// - all OS file I/O goes through the portability shim; this TU stays OS-free
//   (rule 34). off the decode path: lookup once per prefill, store + evict after.
// veritate_engine/src/state_cache.c
// ------------------------------------------------------------------------------------
// Imports:

#include "state_cache.h"

#include "portability.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define SC_ENV_DIR        "VERITATE_STATE_CACHE"
#define SC_ENV_CAP_MB     "VERITATE_STATE_CACHE_MB"
#define SC_ENV_LOG        "VERITATE_STATE_CACHE_LOG"

#define SC_MAGIC          "VSTC"
#define SC_FORMAT_VERSION 1u
#define SC_MIN_PREFIX     8
#define SC_DEFAULT_CAP_MB 4096
#define SC_BYTES_PER_MB   (1024ULL * 1024ULL)

#define SC_HASH_PRIME     1099511628211ULL
#define SC_CHK_PRIME      1000000000000000003ULL
#define SC_CHK_SEED       0x9E3779B97F4A7C15ULL
#define SC_FNV_OFFSET     14695981039346656037ULL
#define SC_FNV_PRIME      1099511628211ULL

#define SC_HEX_LEN        16
#define SC_PATH_CHARS     1024
#define SC_NAME_CHARS     64

// ------------------------------------------------------------------------------------

typedef struct {
    char     magic[4];
    uint32_t format_version;
    uint64_t model_id;
    uint32_t model_version;
    uint32_t hidden;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t seq;
    uint32_t n_enc;
    uint32_t n_global;
    uint32_t n_dec;
    uint32_t conv_kernel;
    uint32_t vocab;
    uint32_t dtype;
    uint32_t prefix_len;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t prefix_chk;
} sc_header_t;

_Static_assert(sizeof(sc_header_t) == 80, "sc_header_t must be 80 bytes, packed");

typedef struct {
    uint64_t* v;
    size_t    n;
    size_t    cap;
} sc_hashset_t;

typedef struct {
    char     name[SC_NAME_CHARS];
    uint64_t size;
    uint64_t mtime_ns;
} sc_file_t;

typedef struct {
    sc_file_t* v;
    size_t     n;
    size_t     cap;
    uint64_t   total;
} sc_filelist_t;

// ------------------------------------------------------------------------------------
// Functions

static const char* sc_dir(void) {
    const char* d = getenv(SC_ENV_DIR);
    return (d && *d) ? d : NULL;
}

int state_cache_enabled(void) {
    return sc_dir() != NULL;
}

static int sc_log_on(void) {
    const char* s = getenv(SC_ENV_LOG);
    return s && *s;
}

// ------------------------------------------------------------------------------------

uint64_t state_cache_model_id(const char* bin_path) {
    uint64_t size = 0, mtime_ns = 0;
    veritate_stat(bin_path, &size, &mtime_ns);
    uint64_t h = SC_FNV_OFFSET;
    for (const unsigned char* p = (const unsigned char*)bin_path; *p; p++)
        h = (h ^ (uint64_t)*p) * SC_FNV_PRIME;
    h = (h ^ size) * SC_FNV_PRIME;
    h = (h ^ mtime_ns) * SC_FNV_PRIME;
    return h;
}

// rolling hashes seeded by model_id. hh[L] names the file, chk[L] guards it.
// prefix-consistent: hh/chk[L] over a longer token stream match the base at L.
static void sc_roll(uint64_t model_id, const int32_t* tokens, int32_t n,
                    uint64_t* hh, uint64_t* chk) {
    hh[0]  = model_id;
    chk[0] = model_id ^ SC_CHK_SEED;
    for (int32_t L = 1; L <= n; L++) {
        uint64_t t = (uint64_t)(uint32_t)tokens[L - 1];
        hh[L]  = hh[L - 1]  * SC_HASH_PRIME + t;
        chk[L] = chk[L - 1] * SC_CHK_PRIME  + t;
    }
}

static void sc_key_path(const char* dir, uint64_t hh, char* out, size_t out_sz) {
    snprintf(out, out_sz, "%s/%016llx", dir, (unsigned long long)hh);
}

// ------------------------------------------------------------------------------------

static void sc_hashset_add(const char* name, uint64_t size, uint64_t mtime_ns, void* ctx) {
    (void)size; (void)mtime_ns;
    if (strlen(name) != SC_HEX_LEN) return;
    char* end = NULL;
    uint64_t val = strtoull(name, &end, 16);
    if (end != name + SC_HEX_LEN || *end != '\0') return;
    sc_hashset_t* s = (sc_hashset_t*)ctx;
    if (s->n == s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 64;
        uint64_t* nv = (uint64_t*)realloc(s->v, nc * sizeof(uint64_t));
        if (!nv) return;
        s->v = nv; s->cap = nc;
    }
    s->v[s->n++] = val;
}

static int sc_hashset_has(const sc_hashset_t* s, uint64_t h) {
    for (size_t i = 0; i < s->n; i++) if (s->v[i] == h) return 1;
    return 0;
}

// ------------------------------------------------------------------------------------

static int sc_header_matches(const sc_header_t* hd, const hybrid_t* h,
                             uint64_t model_id, int32_t L, uint64_t chk) {
    return memcmp(hd->magic, SC_MAGIC, 4) == 0 &&
           hd->format_version == SC_FORMAT_VERSION &&
           hd->model_version  == (uint32_t)VERITATE_MODEL_VERSION_HYBRID &&
           hd->model_id       == model_id &&
           hd->hidden         == (uint32_t)h->hidden &&
           hd->heads          == (uint32_t)h->heads &&
           hd->head_dim       == (uint32_t)h->head_dim &&
           hd->seq            == (uint32_t)h->seq &&
           hd->n_enc          == (uint32_t)h->n_enc &&
           hd->n_global       == (uint32_t)h->n_global &&
           hd->n_dec          == (uint32_t)h->n_dec &&
           hd->conv_kernel    == (uint32_t)h->conv_kernel &&
           hd->vocab          == (uint32_t)h->vocab &&
           hd->dtype          == (uint32_t)h->dtype &&
           hd->prefix_len     == (uint32_t)L &&
           hd->prefix_chk     == chk;
}

// read the compact snapshot at prefix L into h. returns 1 on success, 0 if the
// file is short or shape-mismatched (caller keeps scanning lower prefixes).
static int sc_read_payload(FILE* f, hybrid_t* h, int32_t L) {
    const int32_t H = h->hidden, S = h->seq;
    const int32_t n_local = h->n_enc + h->n_dec;
    const size_t rows = (size_t)L * H;
    for (int32_t li = 0; li < n_local; li++)
        if (fread(h->kv_k + (size_t)li * S * H, sizeof(float), rows, f) != rows) return 0;
    for (int32_t li = 0; li < n_local; li++)
        if (fread(h->kv_v + (size_t)li * S * H, sizeof(float), rows, f) != rows) return 0;
    const size_t rec  = (size_t)h->n_global * h->heads * h->head_dim * h->head_dim;
    const size_t conv = (size_t)h->n_global * (h->conv_kernel - 1) * 3 * H;
    if (fread(h->rec_state, sizeof(float), rec,  f) != rec)  return 0;
    if (fread(h->conv_ring, sizeof(float), conv, f) != conv) return 0;
    if (fread(h->logits, sizeof(float), (size_t)h->vocab, f) != (size_t)h->vocab) return 0;
    if (fread(h->u,      sizeof(float), (size_t)H, f) != (size_t)H) return 0;
    int32_t sc = 0, pos = 0;
    if (fread(&sc,  sizeof(int32_t), 1, f) != 1) return 0;
    if (fread(&pos, sizeof(int32_t), 1, f) != 1) return 0;
    h->slot_count = sc;
    h->pos = pos;
    return 1;
}

int32_t state_cache_try_restore(hybrid_t* h, uint64_t model_id,
                                const int32_t* tokens, int32_t n, int has_trace) {
    const char* dir = sc_dir();
    if (!dir || n <= 0) return 0;
    // trace frames are emitted for the last prompt position, so a traced prefill
    // must re-step at least position n-1: cap the usable prefix to n-1.
    int32_t ceiling = has_trace ? n - 1 : n;
    if (ceiling < SC_MIN_PREFIX) return 0;

    uint64_t* hh  = (uint64_t*)malloc((size_t)(n + 1) * sizeof(uint64_t));
    uint64_t* chk = (uint64_t*)malloc((size_t)(n + 1) * sizeof(uint64_t));
    sc_hashset_t present = {0};
    if (!hh || !chk) { free(hh); free(chk); return 0; }
    sc_roll(model_id, tokens, n, hh, chk);
    veritate_dir_list(dir, sc_hashset_add, &present);

    int32_t restored = 0;
    char path[SC_PATH_CHARS];
    for (int32_t L = ceiling; L >= SC_MIN_PREFIX; L--) {
        if (!sc_hashset_has(&present, hh[L])) continue;
        sc_key_path(dir, hh[L], path, sizeof(path));
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        sc_header_t hd;
        if (fread(&hd, sizeof(hd), 1, f) == 1 &&
            sc_header_matches(&hd, h, model_id, L, chk[L]) &&
            sc_read_payload(f, h, L)) {
            fclose(f);
            veritate_touch(path);
            restored = L;
            break;
        }
        fclose(f);
    }

    free(present.v);
    free(hh);
    free(chk);
    if (sc_log_on()) {
        if (restored) fprintf(stderr, "state_cache: restored L=%d\n", restored);
        else          fprintf(stderr, "state_cache: miss\n");
    }
    return restored;
}

// ------------------------------------------------------------------------------------

static int sc_write_payload(FILE* f, const hybrid_t* h, int32_t L) {
    const int32_t H = h->hidden, S = h->seq;
    const int32_t n_local = h->n_enc + h->n_dec;
    const size_t rows = (size_t)L * H;
    for (int32_t li = 0; li < n_local; li++)
        if (fwrite(h->kv_k + (size_t)li * S * H, sizeof(float), rows, f) != rows) return 0;
    for (int32_t li = 0; li < n_local; li++)
        if (fwrite(h->kv_v + (size_t)li * S * H, sizeof(float), rows, f) != rows) return 0;
    const size_t rec  = (size_t)h->n_global * h->heads * h->head_dim * h->head_dim;
    const size_t conv = (size_t)h->n_global * (h->conv_kernel - 1) * 3 * H;
    if (fwrite(h->rec_state, sizeof(float), rec,  f) != rec)  return 0;
    if (fwrite(h->conv_ring, sizeof(float), conv, f) != conv) return 0;
    if (fwrite(h->logits, sizeof(float), (size_t)h->vocab, f) != (size_t)h->vocab) return 0;
    if (fwrite(h->u,      sizeof(float), (size_t)H, f) != (size_t)H) return 0;
    if (fwrite(&h->slot_count, sizeof(int32_t), 1, f) != 1) return 0;
    if (fwrite(&h->pos,        sizeof(int32_t), 1, f) != 1) return 0;
    return 1;
}

static void sc_filelist_add(const char* name, uint64_t size, uint64_t mtime_ns, void* ctx) {
    if (strlen(name) >= SC_NAME_CHARS) return;
    sc_filelist_t* fl = (sc_filelist_t*)ctx;
    if (fl->n == fl->cap) {
        size_t nc = fl->cap ? fl->cap * 2 : 64;
        sc_file_t* nv = (sc_file_t*)realloc(fl->v, nc * sizeof(sc_file_t));
        if (!nv) return;
        fl->v = nv; fl->cap = nc;
    }
    sc_file_t* e = &fl->v[fl->n++];
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->size = size;
    e->mtime_ns = mtime_ns;
    fl->total += size;
}

static int sc_mtime_cmp(const void* a, const void* b) {
    uint64_t ma = ((const sc_file_t*)a)->mtime_ns, mb = ((const sc_file_t*)b)->mtime_ns;
    return (ma > mb) - (ma < mb);
}

static void sc_evict(const char* dir) {
    const char* cs = getenv(SC_ENV_CAP_MB);
    long cap_mb = SC_DEFAULT_CAP_MB;
    if (cs && *cs) { long v = atol(cs); if (v > 0) cap_mb = v; }
    uint64_t cap = (uint64_t)cap_mb * SC_BYTES_PER_MB;

    sc_filelist_t fl = {0};
    veritate_dir_list(dir, sc_filelist_add, &fl);
    if (fl.total > cap) {
        qsort(fl.v, fl.n, sizeof(sc_file_t), sc_mtime_cmp);
        char path[SC_PATH_CHARS];
        for (size_t i = 0; i < fl.n && fl.total > cap; i++) {
            snprintf(path, sizeof(path), "%s/%s", dir, fl.v[i].name);
            if (veritate_remove(path) == 0) fl.total -= fl.v[i].size;
        }
    }
    free(fl.v);
}

void state_cache_store(hybrid_t* h, uint64_t model_id,
                       const int32_t* tokens, int32_t n) {
    const char* dir = sc_dir();
    if (!dir || n < SC_MIN_PREFIX) return;
    veritate_mkdir(dir);

    uint64_t hh = model_id, chk = model_id ^ SC_CHK_SEED;
    for (int32_t L = 1; L <= n; L++) {
        uint64_t t = (uint64_t)(uint32_t)tokens[L - 1];
        hh  = hh  * SC_HASH_PRIME + t;
        chk = chk * SC_CHK_PRIME  + t;
    }

    char key[SC_PATH_CHARS];
    sc_key_path(dir, hh, key, sizeof(key));
    if (veritate_stat(key, NULL, NULL) == 0) return;

    char tmp[SC_PATH_CHARS];
    snprintf(tmp, sizeof(tmp), "%s.%llu.tmp", key, (unsigned long long)veritate_now_ns());
    FILE* f = fopen(tmp, "wb");
    if (!f) return;

    sc_header_t hd;
    memset(&hd, 0, sizeof(hd));
    memcpy(hd.magic, SC_MAGIC, 4);
    hd.format_version = SC_FORMAT_VERSION;
    hd.model_id       = model_id;
    hd.model_version  = (uint32_t)VERITATE_MODEL_VERSION_HYBRID;
    hd.hidden = (uint32_t)h->hidden; hd.heads = (uint32_t)h->heads;
    hd.head_dim = (uint32_t)h->head_dim; hd.seq = (uint32_t)h->seq;
    hd.n_enc = (uint32_t)h->n_enc; hd.n_global = (uint32_t)h->n_global;
    hd.n_dec = (uint32_t)h->n_dec; hd.conv_kernel = (uint32_t)h->conv_kernel;
    hd.vocab = (uint32_t)h->vocab; hd.dtype = (uint32_t)h->dtype;
    hd.prefix_len = (uint32_t)n; hd.prefix_chk = chk;

    int ok = fwrite(&hd, sizeof(hd), 1, f) == 1 && sc_write_payload(f, h, n);
    fclose(f);
    if (!ok || veritate_rename(tmp, key) != 0) { veritate_remove(tmp); return; }

    if (sc_log_on()) fprintf(stderr, "state_cache: stored L=%d\n", n);
    sc_evict(dir);
}
