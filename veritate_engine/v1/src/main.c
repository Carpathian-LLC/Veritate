// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - veritate entry point. dispatches `chat`, `chat_spec`, `chat_traced`, `trace`,
//   `bench`, or default benchmark.
// ------------------------------------------------------------------------------------

#include "veritate.h"
#include "portability.h"
#include "addons.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

#ifdef _WIN32
    #include <fcntl.h>
    #include <io.h>
    #include <windows.h>
    double now_ms(void) {
        LARGE_INTEGER freq, ctr;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&ctr);
        return (double)ctr.QuadPart * 1000.0 / (double)freq.QuadPart;
    }
#else
    #include <time.h>
    double now_ms(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
    }
#endif

// ------------------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
static void fill_random_int8(int8_t* buf, size_t n, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        buf[i] = (int8_t)((s >> 16) & 0xFF);
    }
}

static int verify_match(const int32_t* a, const int32_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("mismatch at %zu: scalar=%d avx2=%d\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}
#endif

// ------------------------------------------------------------------------------------
// chat mode — read lines from stdin, generate text, write to stdout
// ------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------
// confidence — four-component calibrated score
// ------------------------------------------------------------------------------------

static float sigmoidf(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

// top-K bytes by logit, ordered descending. K must be <= V.
// fills out[K] with byte indices. simple insertion sort over K-element ring; O(V*K).
static void top_k_bytes_by_logit(const int32_t* logits, int32_t V, int32_t K, uint8_t* out) {
    int32_t vals[VERITATE_CAND_TOPK];
    uint8_t idxs[VERITATE_CAND_TOPK];
    for (int32_t i = 0; i < K; i++) { vals[i] = INT32_MIN; idxs[i] = 0; }
    for (int32_t v = 0; v < V; v++) {
        int32_t l = logits[v];
        if (l <= vals[K - 1]) continue;
        int32_t j = K - 1;
        while (j > 0 && vals[j - 1] < l) {
            vals[j] = vals[j - 1];
            idxs[j] = idxs[j - 1];
            j--;
        }
        vals[j] = l;
        idxs[j] = (uint8_t)v;
    }
    for (int32_t i = 0; i < K; i++) out[i] = idxs[i];
}

// margin: (logit_top - logit_second) / sigma_logit. zero on degenerate sigma.
static float compute_margin(const int32_t* logits, int32_t V) {
    int32_t top = logits[0], second = logits[0];
    for (int32_t v = 1; v < V; v++) {
        if (logits[v] > top) { second = top; top = logits[v]; }
        else if (logits[v] > second) second = logits[v];
    }
    double mean = 0.0;
    for (int32_t v = 0; v < V; v++) mean += (double)logits[v];
    mean /= (double)V;
    double var = 0.0;
    for (int32_t v = 0; v < V; v++) {
        double d = (double)logits[v] - mean;
        var += d * d;
    }
    var /= (double)V;
    float sigma = (float)sqrt(var);
    if (sigma <= 1e-6f) return 0.0f;
    return (float)(top - second) / sigma;
}

// entropy score: 1 - H(softmax(logits)) / log2(V).
static float compute_entropy_score(const int32_t* logits, int32_t V) {
    int32_t mx = logits[0];
    for (int32_t v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
    const double inv_scale = 1.0 / 1024.0;
    double sum = 0.0;
    double* e = (double*)malloc((size_t)V * sizeof(double));
    for (int32_t v = 0; v < V; v++) {
        e[v] = exp((double)(logits[v] - mx) * inv_scale);
        sum += e[v];
    }
    if (sum <= 0.0) { free(e); return 0.0f; }
    double H = 0.0;
    double inv_log2 = 1.0 / log(2.0);
    for (int32_t v = 0; v < V; v++) {
        double p = e[v] / sum;
        if (p > 0.0) H += -p * log(p) * inv_log2;
    }
    free(e);
    double log2V = log((double)V) / log(2.0);
    float E = 1.0f - (float)(H / (log2V > 0.0 ? log2V : 1.0));
    if (E < 0.0f) E = 0.0f;
    if (E > 1.0f) E = 1.0f;
    return E;
}

// lens consistency: fraction of layers whose argmax(lens_logits[L]) == sampled byte.
static float compute_lens_consistency(const veritate_shape_t* sh,
                                      const int32_t* lens_pos_block, uint8_t sampled) {
    int32_t hit = 0;
    for (int32_t L = 0; L < sh->layers; L++) {
        const int32_t* row = lens_pos_block + (size_t)L * sh->vocab;
        int32_t mx = row[0]; int32_t am = 0;
        for (int32_t v = 1; v < sh->vocab; v++) if (row[v] > mx) { mx = row[v]; am = v; }
        if (am == (int32_t)sampled) hit++;
    }
    return (float)hit / (float)sh->layers;
}

// residual stability: mean pearson r of (residual_post[L] * embed[byte]) across layer pairs.
static float compute_residual_stab(const trace_record_t* trace, const model_t* m,
                                   int32_t pos, uint8_t sampled) {
    const int32_t H = m->shape.hidden, S = m->shape.seq, Ln = m->shape.layers;
    const int8_t* erow = m->embed + (size_t)sampled * H;
    float* v_prev = (float*)malloc((size_t)H * sizeof(float));
    float* v_curr = (float*)malloc((size_t)H * sizeof(float));
    int have_prev = 0;
    double sum_corr = 0.0;
    int32_t pairs = 0;
    for (int32_t L = 0; L < Ln; L++) {
        const int16_t* r = trace->residual_post + ((size_t)L * S + pos) * H;
        for (int32_t h = 0; h < H; h++) v_curr[h] = (float)r[h] * (float)erow[h];
        if (have_prev) {
            double mc = 0, mp = 0;
            for (int32_t h = 0; h < H; h++) { mc += v_curr[h]; mp += v_prev[h]; }
            mc /= H; mp /= H;
            double sc = 0, sp = 0, sxy = 0;
            for (int32_t h = 0; h < H; h++) {
                double dc = v_curr[h] - mc;
                double dp = v_prev[h] - mp;
                sc += dc * dc; sp += dp * dp; sxy += dc * dp;
            }
            double denom = sqrt(sc * sp);
            double r_pearson = denom > 1e-12 ? sxy / denom : 0.0;
            if (r_pearson < -1.0) r_pearson = -1.0;
            if (r_pearson >  1.0) r_pearson =  1.0;
            sum_corr += r_pearson;
            pairs++;
        }
        memcpy(v_prev, v_curr, (size_t)H * sizeof(float));
        have_prev = 1;
    }
    free(v_prev); free(v_curr);
    if (pairs <= 0) return 0.0f;
    return (float)(sum_corr / (double)pairs);
}

// allocate a trace_record_t with all buffers sized to shape.
static trace_record_t* trace_alloc(const veritate_shape_t* sh, int32_t real_len) {
    trace_record_t* t = (trace_record_t*)calloc(1, sizeof(*t));
    if (!t) return NULL;
    const size_t S = (size_t)sh->seq, H = (size_t)sh->hidden, F = (size_t)sh->ffn;
    const size_t L = (size_t)sh->layers, V = (size_t)sh->vocab, NH = (size_t)sh->heads;
    t->residual_pre     = (int16_t*)            malloc(L * S * H * sizeof(int16_t));
    t->residual_post    = (int16_t*)            malloc(L * S * H * sizeof(int16_t));
    t->ffn_neurons      = (int8_t*)             malloc(L * S * F);
    t->final_act        = (int8_t*)             malloc(H);
    t->prompt_bytes     = (uint8_t*)            malloc((size_t)real_len > 0 ? (size_t)real_len : 1);
    t->top_predictions  = (trace_prediction_t*) malloc(sizeof(trace_prediction_t) * VERITATE_TRACE_TOPK);
    t->attention_scores = (float*)              malloc(L * NH * S * S * sizeof(float));
    t->lens_logits      = (int32_t*)            malloc(L * S * V * sizeof(int32_t));
    return t;
}

static void trace_free(trace_record_t* t) {
    if (!t) return;
    free(t->residual_pre); free(t->residual_post); free(t->ffn_neurons);
    free(t->final_act); free(t->prompt_bytes); free(t->top_predictions);
    free(t->attention_scores); free(t->lens_logits); free(t);
}

// fill trace->top_predictions from a hidden state via tied-embedding LM head.
static void trace_top_predictions(const model_t* m, const int8_t* hidden, trace_prediction_t* out) {
    const int32_t V = m->shape.vocab;
    int32_t* logits = (int32_t*)malloc((size_t)V * sizeof(int32_t));
    int8_t*  taken  = (int8_t*) calloc((size_t)V, 1);
    matmul_int8_vnni_prep(hidden, &m->lm_head, logits, 1);
    for (int32_t k = 0; k < VERITATE_TRACE_TOPK; k++) {
        int32_t best = -2147483647 - 1, idx = 0;
        for (int32_t v = 0; v < V; v++) {
            if (!taken[v] && logits[v] > best) { best = logits[v]; idx = v; }
        }
        taken[idx] = 1;
        out[k].token = (uint8_t)idx;
        out[k].logit = best;
    }
    free(logits); free(taken);
}

static int chat_loop(void) {
    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    int loaded = 0;
    if (model_path) {
        if (model_load(&model, model_path) == 0) loaded = 1;
        else fprintf(stderr, "load failed: %s\n", model_path);
    }
    if (!loaded) model_init_random(&model, 42);

    const int32_t S = model.shape.seq, H = model.shape.hidden;
    static kv_cache_t cache;
    kv_cache_init(&cache, &model.shape);
    int8_t* hidden = (int8_t*)malloc((size_t)H);
    uint32_t rng = (uint32_t)now_ms();

    char* line = (char*)malloc((size_t)S + 2);
    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    while (1) {
        fprintf(stderr, "> ");
        fflush(stderr);
        if (!fgets(line, S + 2, stdin)) break;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;

        int32_t n = tokenize_bytes(line, tokens, S);
        for (int32_t i = n; i < S; i++) tokens[i] = 0;

        forward(&model, &cache, tokens, n, hidden, NULL, NULL);

        int32_t budget = S - n;
        if (budget > 256) budget = 256;
        for (int32_t step = 0; step < budget; step++) {
            int32_t next = sample_token(&model, hidden, 0.8f, 40, &rng);
            int32_t b = next & 0xFF;
            if (b == 0) break;
            putchar(b);
            fflush(stdout);
            if (cache.len >= S) break;
            forward_decode(&model, &cache, next, hidden, NULL);
        }
        putchar('\n');
        fflush(stdout);
        cache.len = 0;
    }
    free(line); free(tokens); free(hidden);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}

// ------------------------------------------------------------------------------------
// chat_traced — read prompts from stdin, stream tokens + per-token activation slices
// to stdout in a binary frame protocol. one process, persistent model, full mri.
//
// stdin  (text, line-buffered):  "<temp> <top_k> <max_new>\n<prompt>\n"   per turn
// stdout (binary): one TFRM frame per generated token, then a TEND end marker per turn.
// frame format (TFRM, version 8):
//   'TFRM' u32_pos u32_real_len u8_byte u8_argmax_byte u8_pad[2]    (16-byte header)
//   for each layer L in 0..layers-1:
//     int16 residual_pre[hidden]
//     int16 residual_post[hidden]
//     int8  ffn_neurons[ffn]
//     int8  attn_row_q[heads][seq]      per-row int8, scale = max_abs(row) / 127
//     float attn_row_scale[heads]       row[i] ~= q[i] * scale
//     int32 lens_logits[vocab]
//   int8  final_act[hidden]
//   int32 logits[vocab]
//   float decisiveness[layers]
//   float bd_scale[layers]              (per-layer fp32 multiplier for dla_entry.w)
//   dla_entry_t dla_picked[VERITATE_DLA_TOPK]
//   dla_entry_t dla_argmax[VERITATE_DLA_TOPK]
//   float margin                  (M_t)
//   float entropy                 (E_t)
//   float lens_consistency        (L_t)
//   float residual_stab           (S_t)
//   float confidence              (calibrated probability)
//   uint16 cand_count             (v8 — count of next-byte candidates with DLA, == VERITATE_CAND_TOPK)
//   uint8  cand_bytes[cand_count] (v8 — byte values, ordered by logit descending)
//   dla_entry_t dla_cand[cand_count][VERITATE_DLA_TOPK]   (v8 — per-candidate DLA, ordered to match cand_bytes)
//   int16  ablation_layer         (v8 — -1 if no ablation active for this token)
//   int16  ablation_neuron        (v8 — -1 if no ablation active for this token)
//   'TEND' u32_pos                (8 bytes — end of turn)
// ------------------------------------------------------------------------------------

static int chat_traced_loop(void) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    int loaded = 0;
    if (model_path) {
        if (model_load(&model, model_path) == 0) loaded = 1;
        else {
            fprintf(stderr, "load failed: %s\n", model_path);
            fflush(stderr);
            return 1;
        }
    }
    if (!loaded) model_init_random(&model, 42);

    // confidence weights — env override or auto-look next to the bin.
    const char* cw_env = getenv("VERITATE_CONFIDENCE_WEIGHTS");
    if (cw_env && *cw_env) {
        if (confidence_weights_load(&model, cw_env) != 0)
            fprintf(stderr, "confidence weights load failed: %s\n", cw_env);
    } else if (model_path) {
        char auto_path[1024];
        const char* slash = strrchr(model_path, '/');
        const char* bslash = strrchr(model_path, '\\');
        const char* sep = (slash && (!bslash || slash > bslash)) ? slash : bslash;
        if (sep) {
            size_t dlen = (size_t)(sep - model_path);
            if (dlen + 32 < sizeof(auto_path)) {
                memcpy(auto_path, model_path, dlen);
                memcpy(auto_path + dlen, "/confidence_weights.json", 25);
                auto_path[dlen + 24] = '\0';
                confidence_weights_load(&model, auto_path);
            }
        }
    }

    const veritate_shape_t* sh = &model.shape;
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn, V = sh->vocab, NH = sh->heads;
    const int32_t Ln = sh->layers;

    static kv_cache_t cache;
    kv_cache_init(&cache, sh);
    int8_t*  hidden = (int8_t*) malloc((size_t)H);
    int32_t* logits = (int32_t*)malloc((size_t)V * sizeof(int32_t));
    uint32_t rng = (uint32_t)now_ms();

    trace_record_t* trace = trace_alloc(sh, S);
    if (!trace || !trace->residual_pre || !trace->residual_post || !trace->ffn_neurons ||
        !trace->final_act || !trace->attention_scores || !trace->lens_logits) {
        fprintf(stderr, "trace alloc failed\n");
        return 1;
    }

    int32_t* tokens         = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    int8_t*  attn_q         = (int8_t*) malloc((size_t)NH * S);
    float*   attn_scale_buf = (float*)  malloc((size_t)NH * sizeof(float));
    float*   decisiveness   = (float*)  malloc((size_t)Ln * sizeof(float));
    int32_t* lens_pos_block = (int32_t*)malloc((size_t)Ln * V * sizeof(int32_t));
    int8_t*  ffn_pos_block  = (int8_t*) malloc((size_t)Ln * F);

    fprintf(stderr, "ready\n");
    fflush(stderr);

    char header[256];
    char* prompt_line = (char*)malloc((size_t)S + 4);
    char  prev_addons_csv[128] = "";

    while (1) {
        if (!fgets(header, sizeof(header), stdin)) break;
        float temp = 0.7f;
        int   top_k = 40;
        int   max_new = 200;
        int   ablate_layer = -1;
        int   ablate_neuron = -1;
        char  addons_csv[128] = "";
        // header format: "temp top_k max_new ablate_l ablate_n [addons_csv]\n"
        // addons_csv is optional; missing means "use the env-var chain or none".
        // empty token "-" explicitly clears any previously-installed chain.
        sscanf(header, "%f %d %d %d %d %127s",
               &temp, &top_k, &max_new, &ablate_layer, &ablate_neuron, addons_csv);
        veritate_set_ablation(ablate_layer, ablate_neuron);

        // per-request addon chain swap. only rebuild when csv changes; common
        // case (single chain, no swaps) reuses the previous chain.
        if (strcmp(addons_csv, prev_addons_csv) != 0) {
            addon_chain_t* old_chain = addons_get_global();
            addon_chain_t* new_chain = NULL;
            if (addons_csv[0] != '\0' && strcmp(addons_csv, "-") != 0) {
                new_chain = addons_build_chain(addons_csv);
            }
            addons_set_global(new_chain);
            if (old_chain != NULL) {
                addon_chain_free(old_chain);
                free(old_chain);
            }
            strncpy(prev_addons_csv, addons_csv, sizeof(prev_addons_csv) - 1);
            prev_addons_csv[sizeof(prev_addons_csv) - 1] = '\0';
        }
        if (!fgets(prompt_line, S + 4, stdin)) break;
        size_t plen = strlen(prompt_line);
        while (plen > 0 && (prompt_line[plen - 1] == '\n' || prompt_line[plen - 1] == '\r'))
            prompt_line[--plen] = '\0';
        if (plen == 0) continue;

        int32_t n = tokenize_bytes(prompt_line, tokens, S);
        for (int32_t i = n; i < S; i++) tokens[i] = 0;

        cache.len = 0;
        forward(&model, &cache, tokens, n, hidden, trace, NULL);

        // addons see the prompt before the first sample, then each sampled byte.
        addon_chain_t* g_chain = addons_get_global();
        if (g_chain != NULL && g_chain->count > 0) {
            addon_chain_reset(g_chain);
            for (int32_t i = 0; i < n; i++) {
                addon_chain_observe(g_chain, tokens[i] & 0xFF);
            }
        }

        int32_t budget = S - n;
        if (budget > max_new) budget = max_new;

        for (int32_t step = 0; step < budget; step++) {
            int32_t pos = (step == 0) ? (n - 1) : (cache.len - 1);

            int32_t argmax_v = 0;
            int32_t next = sample_token_ext(&model, hidden, temp, top_k, &rng, logits, &argmax_v);
            uint8_t b = (uint8_t)(next & 0xFF);
            uint8_t ab = (uint8_t)(argmax_v & 0xFF);
            if (g_chain != NULL && g_chain->count > 0) {
                addon_chain_observe(g_chain, (int)b);
            }

            // header
            uint8_t hdr[16];
            memcpy(hdr, "TFRM", 4);
            uint32_t u_pos = (uint32_t)pos;
            uint32_t u_rl  = (uint32_t)(cache.len);
            memcpy(hdr + 4, &u_pos, 4);
            memcpy(hdr + 8, &u_rl,  4);
            hdr[12] = b; hdr[13] = ab; hdr[14] = 0; hdr[15] = 0;
            fwrite(hdr, 1, 16, stdout);

            // per-layer slices at position `pos`
            for (int32_t L = 0; L < Ln; L++) {
                int16_t* rpre  = trace->residual_pre  + ((size_t)L * S + pos) * H;
                int16_t* rpost = trace->residual_post + ((size_t)L * S + pos) * H;
                int8_t*  fn    = trace->ffn_neurons   + ((size_t)L * S + pos) * F;
                float*   attn  = trace->attention_scores + (size_t)L * NH * S * S;
                int32_t* lens  = trace->lens_logits   + ((size_t)L * S + pos) * V;
                fwrite(rpre,  sizeof(int16_t), (size_t)H, stdout);
                fwrite(rpost, sizeof(int16_t), (size_t)H, stdout);
                fwrite(fn,    sizeof(int8_t),  (size_t)F, stdout);
                for (int32_t h = 0; h < NH; h++) {
                    float* row = attn + ((size_t)h * S + pos) * S;
                    float max_abs = 0.0f;
                    for (int32_t i = 0; i < S; i++) {
                        float a = row[i] < 0 ? -row[i] : row[i];
                        if (a > max_abs) max_abs = a;
                    }
                    float scale = max_abs / 127.0f;
                    float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
                    attn_scale_buf[h] = scale;
                    int8_t* row_q = attn_q + (size_t)h * S;
                    for (int32_t i = 0; i < S; i++) {
                        float q = row[i] * inv;
                        int32_t r = (int32_t)(q < 0 ? q - 0.5f : q + 0.5f);
                        if (r >  127) r =  127;
                        if (r < -128) r = -128;
                        row_q[i] = (int8_t)r;
                    }
                }
                fwrite(attn_q,        sizeof(int8_t), (size_t)NH * S, stdout);
                fwrite(attn_scale_buf, sizeof(float), (size_t)NH, stdout);
                fwrite(lens, sizeof(int32_t), (size_t)V, stdout);
            }

            // final_act and full logits
            fwrite(trace->final_act, sizeof(int8_t), (size_t)H, stdout);
            fwrite(logits, sizeof(int32_t), (size_t)V, stdout);

            // decision-trace fields (v8): per-layer decisiveness + DLA top-K.
            for (int32_t L = 0; L < Ln; L++) {
                int32_t* src = trace->lens_logits + ((size_t)L * S + pos) * V;
                memcpy(lens_pos_block + (size_t)L * V, src, (size_t)V * sizeof(int32_t));
            }
            decisiveness_compute(sh, lens_pos_block, decisiveness);
            fwrite(decisiveness, sizeof(float), (size_t)Ln, stdout);
            fwrite(model.byte_direction_scale, sizeof(float), (size_t)Ln, stdout);

            for (int32_t L = 0; L < Ln; L++) {
                int8_t* src = trace->ffn_neurons + ((size_t)L * S + pos) * F;
                memcpy(ffn_pos_block + (size_t)L * F, src, (size_t)F);
            }
            dla_entry_t dla_picked[VERITATE_DLA_TOPK];
            dla_entry_t dla_argmax[VERITATE_DLA_TOPK];
            dla_top(&model, ffn_pos_block, b,  dla_picked);
            dla_top(&model, ffn_pos_block, ab, dla_argmax);
            fwrite(dla_picked, sizeof(dla_entry_t), VERITATE_DLA_TOPK, stdout);
            fwrite(dla_argmax, sizeof(dla_entry_t), VERITATE_DLA_TOPK, stdout);

            float margin           = compute_margin(logits, V);
            float entropy_score    = compute_entropy_score(logits, V);
            float lens_consistency = compute_lens_consistency(sh, lens_pos_block, b);
            float residual_stab    = compute_residual_stab(trace, &model, pos, b);
            float confidence;
            if (model.cw_loaded) {
                float z = model.cw_M * margin
                        + model.cw_E * entropy_score
                        + model.cw_L * lens_consistency
                        + model.cw_S * residual_stab
                        + model.cw_b;
                confidence = sigmoidf(z);
            } else {
                float z = 0.5f * (margin + entropy_score + lens_consistency + residual_stab) - 1.0f;
                confidence = sigmoidf(z);
            }
            float conf5[5] = { margin, entropy_score, lens_consistency, residual_stab, confidence };
            fwrite(conf5, sizeof(float), 5, stdout);

            // v8 — per-candidate DLA + ablation echo.
            uint16_t cand_count = (uint16_t)VERITATE_CAND_TOPK;
            uint8_t  cand_bytes[VERITATE_CAND_TOPK];
            top_k_bytes_by_logit(logits, V, VERITATE_CAND_TOPK, cand_bytes);
            dla_entry_t dla_cand[VERITATE_CAND_TOPK][VERITATE_DLA_TOPK];
            for (int32_t i = 0; i < VERITATE_CAND_TOPK; i++) {
                dla_top(&model, ffn_pos_block, (int32_t)cand_bytes[i], dla_cand[i]);
            }
            fwrite(&cand_count, sizeof(uint16_t), 1, stdout);
            fwrite(cand_bytes, sizeof(uint8_t), VERITATE_CAND_TOPK, stdout);
            fwrite(dla_cand, sizeof(dla_entry_t), (size_t)VERITATE_CAND_TOPK * VERITATE_DLA_TOPK, stdout);
            int32_t ab_l_i32, ab_n_i32;
            veritate_get_ablation(&ab_l_i32, &ab_n_i32);
            int16_t ablation_layer  = (int16_t)ab_l_i32;
            int16_t ablation_neuron = (int16_t)ab_n_i32;
            fwrite(&ablation_layer,  sizeof(int16_t), 1, stdout);
            fwrite(&ablation_neuron, sizeof(int16_t), 1, stdout);

            fflush(stdout);

            if (b == 0) break;
            if (cache.len >= S) break;

            forward_decode(&model, &cache, next, hidden, trace);
        }

        // end-of-turn marker
        uint8_t end[8];
        memcpy(end, "TEND", 4);
        uint32_t u_done = (uint32_t)cache.len;
        memcpy(end + 4, &u_done, 4);
        fwrite(end, 1, 8, stdout);
        fflush(stdout);

        cache.len = 0;
    }

    free(prompt_line); free(tokens); free(hidden); free(logits);
    free(attn_q); free(attn_scale_buf); free(decisiveness);
    free(lens_pos_block); free(ffn_pos_block);
    trace_free(trace);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}


// ------------------------------------------------------------------------------------
// chat_greedy — target-alone greedy baseline. fixed prompt+budget via stdin lines.
// used for the spec-decoding byte-identity invariant test and the tok/s baseline.
// ------------------------------------------------------------------------------------

static int chat_greedy_loop(int budget_override) {
    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    if (!model_path || model_load(&model, model_path) != 0) {
        fprintf(stderr, "chat_greedy needs VERITATE_MODEL_PATH\n");
        return 1;
    }
    const int32_t S = model.shape.seq, H = model.shape.hidden;
    static kv_cache_t cache;
    kv_cache_init(&cache, &model.shape);
    int8_t* hidden = (int8_t*)malloc((size_t)H);
    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    char* line = (char*)malloc((size_t)S + 4);
    uint32_t rng = 0;
    while (1) {
        if (!fgets(line, S + 2, stdin)) break;
        size_t plen = strlen(line);
        while (plen > 0 && (line[plen - 1] == '\n' || line[plen - 1] == '\r')) line[--plen] = '\0';
        if (plen == 0) continue;
        int32_t n = tokenize_bytes(line, tokens, S);
        for (int32_t i = n; i < S; i++) tokens[i] = 0;
        cache.len = 0;
        forward(&model, &cache, tokens, n, hidden, NULL, NULL);
        int32_t budget = budget_override > 0 ? budget_override : (S - n);
        if (budget > S - n) budget = S - n;
        double t0 = now_ms();
        int32_t emitted = 0;
        for (int32_t step = 0; step < budget; step++) {
            int32_t next = sample_token(&model, hidden, 0.0f, 0, &rng);
            putchar(next & 0xFF);
            emitted++;
            if (cache.len >= S) break;
            if (step < budget - 1) forward_decode(&model, &cache, next, hidden, NULL);
        }
        fflush(stdout);
        double t1 = now_ms();
        double tps = emitted > 0 ? (double)emitted * 1000.0 / (t1 - t0) : 0.0;
        fprintf(stderr, "\n[greedy] emitted=%d  %.2f tok/s  %.2f ms\n", emitted, tps, t1 - t0);
        fflush(stderr);
        putchar('\n'); fflush(stdout);
    }
    free(line); free(tokens); free(hidden);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}

// ------------------------------------------------------------------------------------
// chat_spec — speculative decoding with a draft + target model.
//
// VERITATE_MODEL_PATH points to the target (large, accurate) model.
// VERITATE_DRAFT_PATH points to the draft (small, fast) model.
//
// Per outer step: draft proposes K=4 tokens auto-regressively (greedy). Target
// verifies all K in one batched matmul via forward_verify(K). Accept tokens
// until first argmax disagreement. Sample one fallback from target at the
// disagreement position. Extend both KV caches by the accepted run + fallback.
//
// Stream accepted bytes plain to stdout. Per-turn acceptance rate to stderr.
// Greedy invariant: spec output must equal target-greedy output byte-for-byte.
// ------------------------------------------------------------------------------------

#define VERITATE_SPEC_K 4

static int chat_speculative_loop(int budget_override) {
    static model_t target;
    static model_t draft;
    const char* tgt_path = getenv("VERITATE_MODEL_PATH");
    const char* drf_path = getenv("VERITATE_DRAFT_PATH");
    if (!tgt_path || !drf_path) {
        fprintf(stderr, "chat_spec needs VERITATE_MODEL_PATH and VERITATE_DRAFT_PATH\n");
        return 1;
    }
    if (model_load(&target, tgt_path) != 0) {
        fprintf(stderr, "target load failed: %s\n", tgt_path);
        return 1;
    }
    if (model_load(&draft, drf_path) != 0) {
        fprintf(stderr, "draft load failed: %s\n", drf_path);
        model_free(&target);
        return 1;
    }
    if (target.shape.vocab != draft.shape.vocab) {
        fprintf(stderr, "vocab mismatch: target=%d draft=%d\n",
                target.shape.vocab, draft.shape.vocab);
        model_free(&target); model_free(&draft);
        return 1;
    }

    const int32_t S_t = target.shape.seq;
    const int32_t S_d = draft.shape.seq;
    const int32_t S   = S_t < S_d ? S_t : S_d;
    const int32_t H_t = target.shape.hidden;
    const int32_t H_d = draft.shape.hidden;
    const int32_t V   = target.shape.vocab;

    static kv_cache_t cache_t;
    static kv_cache_t cache_d;
    kv_cache_init(&cache_t, &target.shape);
    kv_cache_init(&cache_d, &draft.shape);

    int8_t*  hidden_t   = (int8_t*) malloc((size_t)H_t);
    int8_t*  hidden_d   = (int8_t*) malloc((size_t)H_d);
    int8_t*  verify_h   = (int8_t*) malloc((size_t)VERITATE_SPEC_K * (size_t)H_t);
    int32_t* draft_toks = (int32_t*)malloc((size_t)VERITATE_SPEC_K * sizeof(int32_t));
    int32_t* logits_t   = (int32_t*)malloc((size_t)V * sizeof(int32_t));
    int32_t* tokens     = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    char*    line       = (char*)   malloc((size_t)S + 4);
    uint32_t rng = (uint32_t)now_ms();

    while (1) {
        fprintf(stderr, "> ");
        fflush(stderr);
        if (!fgets(line, S + 2, stdin)) break;
        size_t plen = strlen(line);
        while (plen > 0 && (line[plen - 1] == '\n' || line[plen - 1] == '\r')) line[--plen] = '\0';
        if (plen == 0) continue;

        int32_t n = tokenize_bytes(line, tokens, S);
        for (int32_t i = n; i < S; i++) tokens[i] = 0;

        cache_t.len = 0;
        cache_d.len = 0;
        forward(&target, &cache_t, tokens, n, hidden_t, NULL, NULL);
        forward(&draft,  &cache_d, tokens, n, hidden_d, NULL, NULL);

        int32_t budget = budget_override > 0 ? budget_override : (S - n);
        if (budget > S - n) budget = S - n;

        int64_t accepted_total = 0;
        int64_t rejected_total = 0;
        int32_t emitted = 0;
        double  t0 = now_ms();

        while (emitted < budget) {
            int32_t base = cache_t.len;
            int32_t Kmax = budget - emitted;
            if (Kmax > VERITATE_SPEC_K) Kmax = VERITATE_SPEC_K;
            if (base + Kmax > S) Kmax = S - base;
            if (Kmax <= 0) break;

            // draft proposes Kmax tokens auto-regressively (greedy).
            int32_t prev_d_len = cache_d.len;
            for (int32_t k = 0; k < Kmax; k++) {
                int32_t t = sample_token(&draft, hidden_d, 0.0f, 0, &rng);
                draft_toks[k] = t;
                if (k < Kmax - 1) {
                    if (cache_d.len >= S_d) { Kmax = k + 1; break; }
                    forward_decode(&draft, &cache_d, t, hidden_d, NULL);
                }
            }

            // target verifies all Kmax in one batched matmul. verify_h[r] is the
            // hidden state after absorbing draft_toks[r] — predicts position base+r+1.
            forward_verify(&target, &cache_t, Kmax, draft_toks, verify_h);

            // accept loop: target's pre-spec hidden_t predicts position base, so
            // it gates draft_toks[0]; verify_h[r-1] predicts position base+r and
            // gates draft_toks[r] for r in 1..Kmax-1. verify_h[Kmax-1] is the free
            // bonus prediction used for fallback when all Kmax matched.
            int32_t accepted = 0;
            int32_t fallback = -1;
            for (int32_t k = 0; k < Kmax; k++) {
                const int8_t* h = (k == 0) ? hidden_t
                                           : verify_h + (size_t)(k - 1) * H_t;
                int32_t argmax_v = 0;
                (void)sample_token_ext(&target, h, 0.0f, 0, &rng, logits_t, &argmax_v);
                if (argmax_v == draft_toks[k]) {
                    accepted++;
                } else {
                    fallback = argmax_v;
                    break;
                }
            }
            if (fallback < 0) {
                // all Kmax matched — bonus token from verify_h[Kmax-1].
                const int8_t* h = verify_h + (size_t)(Kmax - 1) * H_t;
                int32_t argmax_v = 0;
                (void)sample_token_ext(&target, h, 0.0f, 0, &rng, logits_t, &argmax_v);
                fallback = argmax_v;
            }

            // emit accepted bytes + fallback.
            for (int32_t k = 0; k < accepted; k++) {
                int32_t b = draft_toks[k] & 0xFF;
                putchar(b);
                emitted++;
            }
            if (emitted < budget) {
                putchar(fallback & 0xFF);
                emitted++;
            }
            fflush(stdout);

            accepted_total += accepted;
            rejected_total += (Kmax - accepted);

            // sync caches: rewind to base+accepted, append fallback to both.
            cache_t.len = base + accepted;
            cache_d.len = prev_d_len + accepted;
            if (cache_t.len < S_t && cache_d.len < S_d && emitted < budget) {
                forward_decode(&target, &cache_t, fallback, hidden_t, NULL);
                forward_decode(&draft,  &cache_d, fallback, hidden_d, NULL);
            } else {
                break;
            }
        }

        double t1 = now_ms();
        int64_t total = accepted_total + rejected_total;
        double rate = total > 0 ? (double)accepted_total / (double)total : 0.0;
        double tok_per_s = emitted > 0 ? (double)emitted * 1000.0 / (t1 - t0) : 0.0;
        fprintf(stderr, "\n[spec] accepted=%lld rejected=%lld rate=%.3f emitted=%d  %.2f tok/s  %.2f ms\n",
                (long long)accepted_total, (long long)rejected_total, rate, emitted, tok_per_s, t1 - t0);
        fflush(stderr);
        putchar('\n');
        fflush(stdout);
    }

    free(line); free(tokens); free(logits_t);
    free(draft_toks); free(verify_h); free(hidden_d); free(hidden_t);
    kv_cache_free(&cache_d);
    kv_cache_free(&cache_t);
    model_free(&draft);
    model_free(&target);
    return 0;
}

// ------------------------------------------------------------------------------------
// project mri trace mode — capture every internal activation to disk
// ------------------------------------------------------------------------------------

static int trace_mode(const char* prompt, const char* out_path) {
    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    int loaded = model_path && model_load(&model, model_path) == 0;
    if (!loaded) model_init_random(&model, 42);

    const veritate_shape_t* sh = &model.shape;
    const int32_t S = sh->seq, H = sh->hidden;
    static kv_cache_t cache;
    kv_cache_init(&cache, sh);
    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    int32_t n = tokenize_bytes(prompt, tokens, S);
    for (int32_t i = n; i < S; i++) tokens[i] = 0;

    trace_record_t* trace = trace_alloc(sh, n);
    if (!trace || !trace->residual_pre || !trace->residual_post || !trace->ffn_neurons ||
        !trace->final_act || !trace->prompt_bytes || !trace->top_predictions ||
        !trace->attention_scores || !trace->lens_logits) {
        fprintf(stderr, "trace alloc failed\n");
        return 1;
    }
    for (int32_t i = 0; i < n; i++) trace->prompt_bytes[i] = (uint8_t)tokens[i];

    int8_t* out_act = (int8_t*)malloc((size_t)H);
    double t0 = now_ms();
    forward(&model, &cache, tokens, n, out_act, trace, NULL);
    double t1 = now_ms();
    printf("traced forward: %.3f ms (real_len=%d)\n", t1 - t0, n);

    trace_top_predictions(&model, out_act, trace->top_predictions);

    if (trace_write(out_path, sh, trace, n) != 0) {
        fprintf(stderr, "trace_write failed: %s\n", out_path);
        return 1;
    }
    printf("wrote trace: %s\n", out_path);
    printf("top predictions:");
    for (int32_t k = 0; k < VERITATE_TRACE_TOPK; k++) {
        uint8_t tk = trace->top_predictions[k].token;
        printf("  '%c'(%d, logit %d)", (tk >= 32 && tk < 127) ? tk : '.', tk, trace->top_predictions[k].logit);
    }
    printf("\n");

    free(out_act); free(tokens);
    trace_free(trace);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}

// ------------------------------------------------------------------------------------
// bench mode — per-stage forward profiler + decode latency distribution
// ------------------------------------------------------------------------------------

static int dbl_cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static int bench_mode(int argc, char** argv) {
    int trials_fwd = argc > 2 ? atoi(argv[2]) : 50;
    int trials_dec = argc > 3 ? atoi(argv[3]) : 200;

    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    int loaded = model_path && model_load(&model, model_path) == 0;
    if (!loaded) model_init_random(&model, 42);
    int32_t cap = veritate_max_layers(&model);
    const veritate_shape_t* sh = &model.shape;
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn;
    printf("model: %s\n", loaded ? model_path : "random");
    printf("shape: hidden=%d layers=%d heads=%d ffn=%d seq=%d vocab=%d\n",
           sh->hidden, sh->layers, sh->heads, sh->ffn, sh->seq, sh->vocab);
    printf("max_layers: %d  (set VERITATE_MAX_LAYERS to vary)\n", cap);

    static kv_cache_t cache;
    kv_cache_init(&cache, sh);
    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    const char* prompt = "Once upon a time, there was a little girl who";
    int32_t prompt_n = tokenize_bytes(prompt, tokens, S);
    for (int32_t i = prompt_n; i < S; i++) tokens[i] = 0;
    int8_t* hidden = (int8_t*)malloc((size_t)H);

    forward(&model, &cache, tokens, S, hidden, NULL, NULL);  // warmup

    profile_t prof = {0};
    double* fwd_samples = (double*)malloc(sizeof(double) * trials_fwd);
    for (int t = 0; t < trials_fwd; t++) {
        double t0 = now_ms();
        forward(&model, &cache, tokens, S, hidden, NULL, &prof);
        fwd_samples[t] = now_ms() - t0;
    }
    qsort(fwd_samples, trials_fwd, sizeof(double), dbl_cmp);
    double p50 = fwd_samples[trials_fwd / 2];
    double p99 = fwd_samples[(int)(trials_fwd * 0.99)];

    printf("\n");
    printf("forward(prefill seq=%d) over %d trials\n", S, trials_fwd);
    printf("  min %8.3f ms   p50 %8.3f ms   p99 %8.3f ms\n",
           fwd_samples[0], p50, p99);

    double stages = prof.embed_ms + prof.ln_ms + prof.qkv_ms + prof.attn_ms +
                    prof.out_proj_ms + prof.ffn_up_ms + prof.gelu_ms + prof.ffn_down_ms;
    printf("\n");
    printf("per-stage breakdown (avg over %d forwards, all %d layers summed)\n", trials_fwd, sh->layers);
    const struct { const char* name; double ms; } phases[] = {
        {"embed           ", prof.embed_ms},
        {"layernorm       ", prof.ln_ms},
        {"qkv matmul      ", prof.qkv_ms},
        {"attention loops ", prof.attn_ms},
        {"out_proj matmul ", prof.out_proj_ms},
        {"ffn_up matmul   ", prof.ffn_up_ms},
        {"gelu            ", prof.gelu_ms},
        {"ffn_down matmul ", prof.ffn_down_ms},
    };
    for (size_t i = 0; i < sizeof(phases) / sizeof(phases[0]); i++) {
        printf("  %s %7.3f ms  (%5.1f %%)\n",
               phases[i].name, phases[i].ms / trials_fwd, 100.0 * phases[i].ms / stages);
    }

    forward(&model, &cache, tokens, S - trials_dec, hidden, NULL, NULL);
    veritate_mod_stats_reset();
    double* dec_samples = (double*)malloc(sizeof(double) * trials_dec);
    for (int t = 0; t < trials_dec; t++) {
        double t0 = now_ms();
        forward_decode(&model, &cache, 65 + (t % 26), hidden, NULL);
        dec_samples[t] = now_ms() - t0;
    }
    qsort(dec_samples, trials_dec, sizeof(double), dbl_cmp);
    double dp50 = dec_samples[trials_dec / 2];
    double dp99 = dec_samples[(int)(trials_dec * 0.99)];
    int64_t mod_calls = 0, mod_skipped = 0;
    veritate_mod_stats(&mod_calls, &mod_skipped);

    printf("\n");
    printf("forward_decode over %d trials\n", trials_dec);
    printf("  min %8.3f ms   p50 %8.3f ms   p99 %8.3f ms\n",
           dec_samples[0], dp50, dp99);
    if (mod_calls > 0) {
        printf("  mod gate: %lld calls, %lld skipped (%.1f%% blocks bypassed)\n",
               (long long)mod_calls, (long long)mod_skipped,
               100.0 * (double)mod_skipped / (double)mod_calls);
    }

    extern int32_t g_ffn_down_calls;
    extern int64_t g_ffn_down_nz_sum;
    extern int32_t g_ffn_down_sparse_calls;
    if (g_ffn_down_calls > 0) {
        double avg_nz = (double)g_ffn_down_nz_sum / (double)g_ffn_down_calls;
        printf("  ffn_down sparsity: avg n_nz=%.0f / %d (%.1f%% nonzero), sparse %d / %d (%.1f%%)\n",
               avg_nz, F, 100.0 * avg_nz / F,
               g_ffn_down_sparse_calls, g_ffn_down_calls,
               100.0 * g_ffn_down_sparse_calls / g_ffn_down_calls);
    }

    free(fwd_samples); free(dec_samples); free(hidden); free(tokens);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}

// ------------------------------------------------------------------------------------
// ppl mode — byte-level cross-entropy + decode latency on val data
// usage: ppl <val_file> <num_chunks> [chunk_len] [stride]
//   chunk_len defaults to V_SEQ (256). stride defaults to chunk_len.
// reports mean -log2(p[true_next_byte]) (bits/byte), perplexity (2^mean), and
// decode p50 over all decode steps run during the eval.
// ------------------------------------------------------------------------------------

#ifndef VERITATE_GELU_ZERO_THRESH
#define VERITATE_GELU_ZERO_THRESH 0
#endif

static int ppl_mode(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s ppl <val_file> <num_chunks> [chunk_len] [stride]\n", argv[0]);
        return 1;
    }
    const char* val_path  = argv[2];
    int num_chunks        = atoi(argv[3]);

    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    int loaded = 0;
    FILE* fp = fopen(val_path, "rb");
    if (!fp) { fprintf(stderr, "open failed: %s\n", val_path); return 1; }
    if (model_path) {
        if (model_load(&model, model_path) == 0) loaded = 1;
        else { fprintf(stderr, "load failed: %s\n", model_path); fclose(fp); return 1; }
    } else {
        fprintf(stderr, "VERITATE_MODEL_PATH unset\n");
        fclose(fp);
        return 1;
    }
    (void)loaded;

    const veritate_shape_t* sh = &model.shape;
    const int32_t S = sh->seq, H = sh->hidden, V = sh->vocab, F = sh->ffn;
    int chunk_len         = argc > 4 ? atoi(argv[4]) : S;
    int stride            = argc > 5 ? atoi(argv[5]) : chunk_len;
    if (chunk_len < 2 || chunk_len > S) chunk_len = S;
    if (num_chunks < 1) num_chunks = 1;
    if (stride < 1) stride = chunk_len;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < (long)chunk_len) { fprintf(stderr, "val file too small\n"); fclose(fp); model_free(&model); return 1; }

    static kv_cache_t cache;
    kv_cache_init(&cache, sh);
    int8_t*  hidden = (int8_t*) malloc((size_t)H);
    int32_t* logits = (int32_t*)malloc((size_t)V * sizeof(int32_t));
    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    uint8_t* buf    = (uint8_t*)malloc((size_t)S);

    double total_neg_log2 = 0.0;
    long total_tokens = 0;
    int chunks_done = 0;

    double* dec_samples = (double*)malloc(sizeof(double) * (size_t)num_chunks * (size_t)chunk_len);
    long n_dec_samples = 0;

    long max_offset = fsize - chunk_len;
    long step = (long)stride;
    long offset = 0;

    for (int c = 0; c < num_chunks; c++) {
        if (offset > max_offset) break;
        fseek(fp, offset, SEEK_SET);
        if (fread(buf, 1, (size_t)chunk_len, fp) != (size_t)chunk_len) break;
        offset += step;

        for (int i = 0; i < chunk_len; i++) tokens[i] = buf[i];

        cache.len = 0;
        forward(&model, &cache, tokens, 1, hidden, NULL, NULL);

        for (int i = 0; i < chunk_len - 1; i++) {
            matmul_int8_vnni_prep(hidden, &model.lm_head, logits, 1);
            int32_t max_logit = logits[0];
            for (int32_t v = 1; v < V; v++) if (logits[v] > max_logit) max_logit = logits[v];
            // hidden and embed share activation scale 32; int_logit = fp_logit * 32 * 32.
            const double inv_scale = 1.0 / 1024.0;
            double sum = 0.0;
            for (int32_t v = 0; v < V; v++) {
                sum += exp((double)(logits[v] - max_logit) * inv_scale);
            }
            int32_t true_next = tokens[i + 1] & 0xFF;
            if (V > 0) true_next = ((true_next % V) + V) % V;
            double log_p = (double)(logits[true_next] - max_logit) * inv_scale - log(sum);
            total_neg_log2 += -log_p / log(2.0);
            total_tokens++;

            if (i + 1 >= chunk_len - 1) break;

            double t0 = now_ms();
            forward_decode(&model, &cache, tokens[i + 1], hidden, NULL);
            dec_samples[n_dec_samples++] = now_ms() - t0;
        }
        chunks_done++;
    }
    fclose(fp);

    double mean_bpb = total_neg_log2 / (double)total_tokens;
    double ppl = pow(2.0, mean_bpb);

    qsort(dec_samples, (size_t)n_dec_samples, sizeof(double), dbl_cmp);
    double dp50 = n_dec_samples > 0 ? dec_samples[n_dec_samples / 2] : 0.0;
    double dp99 = n_dec_samples > 0 ? dec_samples[(int)(n_dec_samples * 0.99)] : 0.0;

    int gelu_thr = VERITATE_GELU_ZERO_THRESH;
    printf("ppl: chunks=%d chunk_len=%d tokens=%ld bpb=%.4f ppl=%.4f gelu_thr=%d\n",
           chunks_done, chunk_len, total_tokens, mean_bpb, ppl, gelu_thr);
    printf("decode p50=%.4f ms p99=%.4f ms (n=%ld)\n", dp50, dp99, n_dec_samples);

    extern int32_t g_ffn_down_calls;
    extern int64_t g_ffn_down_nz_sum;
    extern int32_t g_ffn_down_sparse_calls;
    if (g_ffn_down_calls > 0) {
        double avg_nz = (double)g_ffn_down_nz_sum / (double)g_ffn_down_calls;
        printf("ffn_down sparsity: avg n_nz=%.0f / %d (%.1f%% nonzero), sparse %d / %d (%.1f%%)\n",
               avg_nz, F, 100.0 * avg_nz / F,
               g_ffn_down_sparse_calls, g_ffn_down_calls,
               100.0 * g_ffn_down_sparse_calls / g_ffn_down_calls);
    }

    free(dec_samples); free(hidden); free(logits); free(tokens); free(buf);
    kv_cache_free(&cache);
    model_free(&model);
    return 0;
}

// ------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    // addon chain: configured via VERITATE_ADDONS=<csv>. NULL/empty -> no addons,
    // hot path is bit-identical to the pre-addon engine.
    const char* addons_csv = getenv("VERITATE_ADDONS");
    if (addons_csv && addons_csv[0] != '\0') {
        addons_set_global(addons_build_chain(addons_csv));
    }

    if (argc > 1 && strcmp(argv[1], "chat") == 0) return chat_loop();
    if (argc > 1 && strcmp(argv[1], "chat_spec") == 0) {
        int b = argc > 2 ? atoi(argv[2]) : 0;
        return chat_speculative_loop(b);
    }
    if (argc > 1 && strcmp(argv[1], "chat_greedy") == 0) {
        int b = argc > 2 ? atoi(argv[2]) : 0;
        return chat_greedy_loop(b);
    }
    if (argc > 1 && strcmp(argv[1], "chat_traced") == 0) return chat_traced_loop();
    if (argc > 1 && strcmp(argv[1], "trace") == 0) {
        const char* prompt   = argc > 2 ? argv[2] : "Once upon a time, ";
        const char* out_path = argc > 3 ? argv[3] : "veritate_trace.bin";
        return trace_mode(prompt, out_path);
    }
    if (argc > 1 && strcmp(argv[1], "bench") == 0) return bench_mode(argc, argv);
    if (argc > 1 && strcmp(argv[1], "ppl") == 0) return ppl_mode(argc, argv);

    printf("veritate v%s\n", VERITATE_VERSION);

    cpu_features_t feat;
    cpu_detect(&feat);
    cpu_print(&feat);

    dispatch_info_t info;
    dispatch_init(&feat, &info);
    printf("dispatch: matmul -> %s\n\n", info.matmul_backend);

    int parity_failures = 0;

#if defined(__x86_64__) || defined(_M_X64)
    const int32_t M = 1024, N = 1024, K = 1024;
    const size_t a_sz = (size_t)M * K;
    const size_t b_sz = (size_t)K * N;
    const size_t c_sz = (size_t)M * N;

    int8_t*  a   = (int8_t*) malloc(a_sz);
    int8_t*  b   = (int8_t*) malloc(b_sz);
    int32_t* c_ref = (int32_t*)malloc(c_sz * sizeof(int32_t));
    int32_t* c_av2 = (int32_t*)malloc(c_sz * sizeof(int32_t));
    int32_t* c_vn  = (int32_t*)malloc(c_sz * sizeof(int32_t));
    int32_t* c_mt  = (int32_t*)malloc(c_sz * sizeof(int32_t));

    fill_random_int8(a, a_sz, 1);
    fill_random_int8(b, b_sz, 2);

    printf("matmul %dx%d x %d (INT8)\n", M, K, N);

    double t0 = now_ms();
    matmul_int8_scalar(a, b, c_ref, M, N, K);
    double t_scalar = now_ms() - t0;
    printf("  scalar:        %9.3f ms   1.00x   (oracle)\n", t_scalar);

    double t1 = now_ms();
    matmul_int8_avx2(a, b, c_av2, M, N, K);
    double t_avx2 = now_ms() - t1;
    int ok_avx2 = verify_match(c_ref, c_av2, c_sz);
    printf("  avx2:          %9.3f ms  %5.1fx   %s\n",
           t_avx2, t_scalar / t_avx2, ok_avx2 ? "verify OK" : "FAIL");

    double t2 = now_ms();
    matmul_int8_vnni(a, b, c_vn, M, N, K);
    double t_vnni = now_ms() - t2;
    int ok_vnni = verify_match(c_ref, c_vn, c_sz);
    printf("  avx512_vnni:   %9.3f ms  %5.1fx   %s\n",
           t_vnni, t_scalar / t_vnni, ok_vnni ? "verify OK" : "FAIL");

    // warm up — first call pays thread pool init
    matmul_int8_vnni_mt(a, b, c_mt, M, N, K);

    double best_mt = 1e9, sum_mt = 0.0;
    const int trials = 20;
    for (int trial = 0; trial < trials; trial++) {
        double t3 = now_ms();
        matmul_int8_vnni_mt(a, b, c_mt, M, N, K);
        double dt = now_ms() - t3;
        if (dt < best_mt) best_mt = dt;
        sum_mt += dt;
    }
    int ok_mt = verify_match(c_ref, c_mt, c_sz);
    printf("  vnni_mt:       %9.3f ms  %5.1fx   %s   (best of %d, avg %.3f)\n",
           best_mt, t_scalar / best_mt, ok_mt ? "verify OK" : "FAIL",
           trials, sum_mt / trials);

    // prepped: weights pre-transposed once (real-inference path)
    prepped_b_t pb;
    double t_prep0 = now_ms();
    prep_b(b, N, K, &pb);
    double t_prep = now_ms() - t_prep0;

    matmul_int8_vnni_mt_prep(a, &pb, c_mt, M);  // warmup

    double best_prep = 1e9, sum_prep = 0.0;
    for (int trial = 0; trial < trials; trial++) {
        double t = now_ms();
        matmul_int8_vnni_mt_prep(a, &pb, c_mt, M);
        double dt = now_ms() - t;
        if (dt < best_prep) best_prep = dt;
        sum_prep += dt;
    }
    int ok_prep = verify_match(c_ref, c_mt, c_sz);
    printf("  vnni_mt_prep:  %9.3f ms  %5.1fx   %s   (best of %d, avg %.3f)\n",
           best_prep, t_scalar / best_prep, ok_prep ? "verify OK" : "FAIL",
           trials, sum_prep / trials);
    printf("                 (one-time prep_b cost: %.3f ms)\n", t_prep);

    free_prepped_b(&pb);

    double gate = best_prep;

    printf("\n");
    printf("sub-ms gate: %s  (best = %.3f ms)\n",
           gate < 1.0 ? "PASS" : "FAIL", gate);

    free(a); free(b); free(c_ref); free(c_av2); free(c_vn); free(c_mt);
#else
    printf("default-mode int8 matmul baseline is x86_64-only; portable forward/decode bench: \"%s bench\".\n",
           argc > 0 ? argv[0] : "veritate");
#endif

    // ------------------------------------------------------------------------------
    // int4 packed kernel — bit-match against scalar oracle, then bench at decode
    // shape. portable: matmul_int4_vnni_prep is supplied per-arch (AVX-512 VNNI
    // on x86_64, NEON SDOT on arm64). matmul_int8_vnni_prep is the dispatched
    // dense int8 matmul on every arch.
    // ------------------------------------------------------------------------------
    {
        const int32_t IK = V_HIDDEN;   // 768
        const int32_t II = V_FFN;      // 3072
        int8_t*  i4_a   = (int8_t*) veritate_aligned_alloc((size_t)IK,             64);
        int8_t*  i4_w   = (int8_t*) veritate_aligned_alloc((size_t)IK * II,        64);
        int32_t* c_ref4 = (int32_t*)veritate_aligned_alloc((size_t)II * sizeof(int32_t), 64);
        int32_t* c_avx4 = (int32_t*)veritate_aligned_alloc((size_t)II * sizeof(int32_t), 64);

        unsigned ss = 7u;
        for (int32_t p = 0; p < IK; p++) {
            ss = ss * 1103515245u + 12345u;
            i4_a[p] = (int8_t)((int32_t)((ss >> 16) & 0xFF) - 128);
        }
        for (size_t p = 0; p < (size_t)IK * II; p++) {
            ss = ss * 1103515245u + 12345u;
            i4_w[p] = (int8_t)(((int32_t)((ss >> 24) & 0x0F)) - 8);
        }

        prepped_b_int4_t pb4;
        prep_b_int4(i4_w, II, IK, &pb4);

        matmul_int4_scalar_prep(i4_a, &pb4, c_ref4, 1);
        matmul_int4_vnni_prep  (i4_a, &pb4, c_avx4, 1);

        int ok_i4 = 1;
        int first_mismatch = -1;
        for (int32_t j = 0; j < II; j++) {
            if (c_avx4[j] != c_ref4[j]) {
                if (first_mismatch < 0) first_mismatch = j;
                ok_i4 = 0;
            }
        }
        printf("\n");
        printf("int4 packed (m=1, k=%d, n=%d):\n", IK, II);
        printf("  scalar vs simd:               %s\n",
               ok_i4 ? "verify OK (bit-match)" : "FAIL");
        if (!ok_i4) {
            printf("  first mismatch at j=%d: scalar=%d simd=%d\n",
                   first_mismatch, c_ref4[first_mismatch], c_avx4[first_mismatch]);
            parity_failures++;
        }

        // bench decode shape (m=1) for int4 vs int8 reference
        prepped_b_t pb8;
        prep_b(i4_w, II, IK, &pb8);
        int32_t* c_int8d = (int32_t*)veritate_aligned_alloc((size_t)II * sizeof(int32_t), 64);

        for (int t = 0; t < 50; t++) {  // warmup
            matmul_int4_vnni_prep(i4_a, &pb4, c_avx4, 1);
            matmul_int8_vnni_prep(i4_a, &pb8, c_int8d, 1);
        }

        const int btrials = 500;
        double best_i8 = 1e9, best_i4 = 1e9;
        for (int t = 0; t < btrials; t++) {
            double t0 = now_ms();
            matmul_int8_vnni_prep(i4_a, &pb8, c_int8d, 1);
            double d = now_ms() - t0;
            if (d < best_i8) best_i8 = d;
        }
        for (int t = 0; t < btrials; t++) {
            double t0 = now_ms();
            matmul_int4_vnni_prep(i4_a, &pb4, c_avx4, 1);
            double d = now_ms() - t0;
            if (d < best_i4) best_i4 = d;
        }
        printf("  decode bench m=1 ffn_up shape (k=%d, n=%d), best of %d:\n",
               IK, II, btrials);
        printf("    int8 vnni_prep:           %8.4f ms\n", best_i8);
        printf("    int4 vnni_prep:           %8.4f ms   %.2fx\n",
               best_i4, best_i8 / best_i4);

        // ffn_down shape (k=3072, n=768)
        int8_t*  i4_a_dn = (int8_t*) veritate_aligned_alloc((size_t)V_FFN,                 64);
        int8_t*  i4_w_dn = (int8_t*) veritate_aligned_alloc((size_t)V_FFN * V_HIDDEN,      64);
        int32_t* c_ref_dn= (int32_t*)veritate_aligned_alloc((size_t)V_HIDDEN * sizeof(int32_t), 64);
        int32_t* c_a4_dn = (int32_t*)veritate_aligned_alloc((size_t)V_HIDDEN * sizeof(int32_t), 64);
        int32_t* c_a8_dn = (int32_t*)veritate_aligned_alloc((size_t)V_HIDDEN * sizeof(int32_t), 64);
        for (int32_t p = 0; p < V_FFN; p++) {
            ss = ss * 1103515245u + 12345u;
            i4_a_dn[p] = (int8_t)((int32_t)((ss >> 16) & 0xFF) - 128);
        }
        for (size_t p = 0; p < (size_t)V_FFN * V_HIDDEN; p++) {
            ss = ss * 1103515245u + 12345u;
            i4_w_dn[p] = (int8_t)(((int32_t)((ss >> 24) & 0x0F)) - 8);
        }
        prepped_b_int4_t pb4_dn;
        prep_b_int4(i4_w_dn, V_HIDDEN, V_FFN, &pb4_dn);
        prepped_b_t pb8_dn;
        prep_b(i4_w_dn, V_HIDDEN, V_FFN, &pb8_dn);

        matmul_int4_scalar_prep(i4_a_dn, &pb4_dn, c_ref_dn, 1);
        matmul_int4_vnni_prep  (i4_a_dn, &pb4_dn, c_a4_dn, 1);
        int ok_dn = 1;
        for (int32_t j = 0; j < V_HIDDEN; j++) if (c_a4_dn[j] != c_ref_dn[j]) ok_dn = 0;
        printf("  scalar vs simd (ffn_down k=%d n=%d):   %s\n",
               V_FFN, V_HIDDEN, ok_dn ? "verify OK" : "FAIL");
        if (!ok_dn) parity_failures++;

        for (int t = 0; t < 50; t++) {
            matmul_int4_vnni_prep(i4_a_dn, &pb4_dn, c_a4_dn, 1);
            matmul_int8_vnni_prep(i4_a_dn, &pb8_dn, c_a8_dn, 1);
        }
        double best_i8_dn = 1e9, best_i4_dn = 1e9;
        for (int t = 0; t < btrials; t++) {
            double t0 = now_ms();
            matmul_int8_vnni_prep(i4_a_dn, &pb8_dn, c_a8_dn, 1);
            double d = now_ms() - t0;
            if (d < best_i8_dn) best_i8_dn = d;
        }
        for (int t = 0; t < btrials; t++) {
            double t0 = now_ms();
            matmul_int4_vnni_prep(i4_a_dn, &pb4_dn, c_a4_dn, 1);
            double d = now_ms() - t0;
            if (d < best_i4_dn) best_i4_dn = d;
        }
        printf("  decode bench m=1 ffn_down shape (k=%d, n=%d), best of %d:\n",
               V_FFN, V_HIDDEN, btrials);
        printf("    int8 vnni_prep:           %8.4f ms\n", best_i8_dn);
        printf("    int4 vnni_prep:           %8.4f ms   %.2fx\n",
               best_i4_dn, best_i8_dn / best_i4_dn);

        free_prepped_b_int4(&pb4);
        free_prepped_b(&pb8);
        free_prepped_b_int4(&pb4_dn);
        free_prepped_b(&pb8_dn);
        veritate_aligned_free(i4_a); veritate_aligned_free(i4_w); veritate_aligned_free(c_ref4); veritate_aligned_free(c_avx4); veritate_aligned_free(c_int8d);
        veritate_aligned_free(i4_a_dn); veritate_aligned_free(i4_w_dn); veritate_aligned_free(c_ref_dn); veritate_aligned_free(c_a4_dn); veritate_aligned_free(c_a8_dn);
    }

    // ------------------------------------------------------------------------------
    // ternary kernel parity check (BitNet b1.58). trits in {-1, 0, +1} packed
    // 5-per-byte. scalar oracle vs vnni path; rule-23 contract.
    // x86_64 only: matmul_ternary_vnni_prep is the only SIMD ternary kernel
    // wired today. arm64 NEON ternary path lands in a follow-up.
    // ------------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
    {
        const int32_t TK = V_HIDDEN;   // 768
        const int32_t TN = V_FFN;      // 3072
        int8_t*  t_a   = (int8_t*) veritate_aligned_alloc((size_t)TK,             64);
        int8_t*  t_w   = (int8_t*) veritate_aligned_alloc((size_t)TK * TN,        64);
        int32_t* c_ref_t = (int32_t*)veritate_aligned_alloc((size_t)TN * sizeof(int32_t), 64);
        int32_t* c_simd_t= (int32_t*)veritate_aligned_alloc((size_t)TN * sizeof(int32_t), 64);

        unsigned ts = 1234567u;
        for (int32_t p = 0; p < TK; p++) {
            ts = ts * 1103515245u + 12345u;
            t_a[p] = (int8_t)((int32_t)((ts >> 16) & 0xFF) - 128);
        }
        for (size_t p = 0; p < (size_t)TK * TN; p++) {
            ts = ts * 1103515245u + 12345u;
            int32_t r = (int32_t)((ts >> 24) & 3);
            t_w[p] = (int8_t)(r == 0 ? 0 : (r == 1 ? 1 : (r == 2 ? -1 : 0)));
        }

        prepped_b_ternary_t pbt;
        prep_b_ternary(t_w, TN, TK, 1.0f, &pbt);

        matmul_ternary_scalar_prep(t_a, &pbt, c_ref_t,  1);
        matmul_ternary_vnni_prep  (t_a, &pbt, c_simd_t, 1);

        int ok_t = 1;
        int first_mismatch_t = -1;
        for (int32_t j = 0; j < TN; j++) {
            if (c_simd_t[j] != c_ref_t[j]) {
                if (first_mismatch_t < 0) first_mismatch_t = j;
                ok_t = 0;
            }
        }
        printf("\n");
        printf("ternary packed (m=1, k=%d, n=%d):\n", TK, TN);
        printf("  scalar vs simd:               %s\n",
               ok_t ? "verify OK (bit-match)" : "FAIL");
        if (!ok_t) {
            printf("  first mismatch at j=%d: scalar=%d simd=%d\n",
                   first_mismatch_t, c_ref_t[first_mismatch_t], c_simd_t[first_mismatch_t]);
            parity_failures++;
        }

        // pack/unpack round-trip on a length-TK row.
        uint8_t* packed = (uint8_t*)veritate_aligned_alloc(((size_t)TK + 4) / 5, 64);
        int8_t*  back   = (int8_t*) veritate_aligned_alloc((size_t)TK, 64);
        ternary_pack_row(t_w, TK, packed);
        ternary_unpack_row(packed, TK, back);
        int ok_pack = 1;
        for (int32_t i = 0; i < TK; i++) {
            if (back[i] != t_w[i]) { ok_pack = 0; break; }
        }
        printf("  pack/unpack round-trip:       %s\n",
               ok_pack ? "verify OK" : "FAIL");
        if (!ok_pack) parity_failures++;

        veritate_aligned_free(packed); veritate_aligned_free(back);
        free_prepped_b_ternary(&pbt);
        veritate_aligned_free(t_a); veritate_aligned_free(t_w);
        veritate_aligned_free(c_ref_t); veritate_aligned_free(c_simd_t);
    }
#endif  // __x86_64__ || _M_X64

    // v3 — single transformer block forward pass
    static model_t model;
    const char* model_path = getenv("VERITATE_MODEL_PATH");
    double t_init0 = now_ms();
    int loaded = 0;
    if (model_path) {
        if (model_load(&model, model_path) == 0) {
            loaded = 1;
        } else {
            printf("  model load failed (path=%s); falling back to random\n", model_path);
        }
    }
    if (!loaded) model_init_random(&model, 42);
    double t_init = now_ms() - t_init0;

    const veritate_shape_t* sh = &model.shape;
    const int32_t S = sh->seq, H = sh->hidden;

    printf("\n");
    printf("v3.1 transformer (vocab=%d seq=%d hidden=%d heads=%d ffn=%d layers=%d)\n",
           sh->vocab, sh->seq, sh->hidden, sh->heads, sh->ffn, sh->layers);
    printf("  model init (%s):     %8.3f ms\n",
           loaded ? "loaded" : "random", t_init);

    int32_t* tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
    const char* prompt = getenv("VERITATE_PROMPT");
    if (!prompt) prompt = "Hello, Veritate. Speak now: ";
    int32_t prompt_len = tokenize_bytes(prompt, tokens, S);
    for (int32_t i = prompt_len; i < S; i++) tokens[i] = 0;

    {
        const char* probe = "abc\x01\xff\x7e";
        int32_t pt[8];
        int32_t pn = tokenize_bytes(probe, pt, 8);
        char back[9];
        detokenize_bytes(pt, pn, back);
        int ok = (pn == 6) && (memcmp(probe, back, 6) == 0);
        printf("  tokenizer round-trip:         %s\n", ok ? "OK" : "MISMATCH");
    }

    static kv_cache_t cache;
    kv_cache_init(&cache, sh);
    int8_t* out_act = (int8_t*)malloc((size_t)H);
    forward(&model, &cache, tokens, S, out_act, NULL, NULL);  // warmup

    double best_fwd = 1e9, sum_fwd = 0.0;
    const int fwd_trials = 50;
    for (int t = 0; t < fwd_trials; t++) {
        double t0 = now_ms();
        forward(&model, &cache, tokens, S, out_act, NULL, NULL);
        double dt = now_ms() - t0;
        if (dt < best_fwd) best_fwd = dt;
        sum_fwd += dt;
    }
    printf("  forward pass (prefill seq=%d): %8.3f ms   (best of %d, avg %.3f)\n",
           S, best_fwd, fwd_trials, sum_fwd / fwd_trials);
    printf("  output[0..7]:                 ");
    for (int i = 0; i < 8; i++) printf("%4d ", out_act[i]);
    printf("\n");

#ifdef VERITATE_VERIFY_DECODE
    {
        const int32_t verify_n     = 47;
        const int32_t verify_token = 42;
        int32_t* verify_tokens = (int32_t*)malloc((size_t)S * sizeof(int32_t));
        for (int i = 0; i < verify_n; i++) verify_tokens[i] = tokens[i];
        verify_tokens[verify_n] = verify_token;

        int8_t* hidden_full   = (int8_t*)malloc((size_t)H);
        int8_t* hidden_decode = (int8_t*)malloc((size_t)H);
        forward(&model, &cache, verify_tokens, verify_n + 1, hidden_full, NULL, NULL);
        forward(&model, &cache, verify_tokens, verify_n, hidden_decode, NULL, NULL);
        forward_decode(&model, &cache, verify_token, hidden_decode, NULL);

        int max_abs_diff = 0;
        for (int i = 0; i < H; i++) {
            int diff = (int)hidden_full[i] - (int)hidden_decode[i];
            if (diff < 0) diff = -diff;
            if (diff > max_abs_diff) max_abs_diff = diff;
        }
        int decode_ok = max_abs_diff <= 1;
        printf("  decode vs full forward:       %s   (max int8 diff = %d)\n",
               decode_ok ? "OK (within 1 LSB)" : "MISMATCH", max_abs_diff);
        if (!decode_ok) parity_failures++;
        free(verify_tokens); free(hidden_full); free(hidden_decode);
    }

    {
        const int Ks[] = { 1, 2, 4, 8, 16 };
        const int n_K = sizeof(Ks) / sizeof(Ks[0]);
        const int32_t prefix_len = 64;
        int32_t verify_tokens[VERITATE_VERIFY_K_MAX];
        for (int32_t r = 0; r < VERITATE_VERIFY_K_MAX; r++) {
            verify_tokens[r] = (prefix_len + r * 7 + 13) & 0xFF;
        }
        static kv_cache_t cache_seed, cache_ref, cache_ver;
        kv_cache_init(&cache_seed, sh);
        kv_cache_init(&cache_ref,  sh);
        kv_cache_init(&cache_ver,  sh);
        int8_t* hidden_seed = (int8_t*)malloc((size_t)H);
        int8_t* out_ref     = (int8_t*)malloc((size_t)VERITATE_VERIFY_K_MAX * H);
        int8_t* out_ver     = (int8_t*)malloc((size_t)VERITATE_VERIFY_K_MAX * H);
        int8_t* hbuf        = (int8_t*)malloc((size_t)H);

        printf("  forward_verify vs K decodes:  ");
        int all_ok = 1;
        int worst = 0;
        for (int ki = 0; ki < n_K; ki++) {
            int K = Ks[ki];
            cache_seed.len = 0;
            forward(&model, &cache_seed, tokens, prefix_len, hidden_seed, NULL, NULL);
            kv_cache_copy(&cache_ref, &cache_seed);
            kv_cache_copy(&cache_ver, &cache_seed);

            for (int32_t r = 0; r < K; r++) {
                forward_decode(&model, &cache_ref, verify_tokens[r], hbuf, NULL);
                memcpy(out_ref + (size_t)r * H, hbuf, (size_t)H);
            }
            forward_verify(&model, &cache_ver, K, verify_tokens, out_ver);

            int diff = 0;
            for (int32_t i = 0; i < K * H; i++) {
                int d = (int)out_ver[i] - (int)out_ref[i];
                if (d < 0) d = -d;
                if (d > diff) diff = d;
            }
            if (diff > 1) all_ok = 0;
            if (diff > worst) worst = diff;
        }
        printf("%s   (max int8 diff = %d, K in {1,2,4,8,16})\n",
               all_ok ? "OK (within 1 LSB)" : "MISMATCH", worst);
        if (!all_ok) parity_failures++;
        free(hidden_seed); free(out_ref); free(out_ver); free(hbuf);
        kv_cache_free(&cache_seed); kv_cache_free(&cache_ref); kv_cache_free(&cache_ver);
    }
#endif

    const int32_t prompt_n = 48;
    const int32_t n_gen = 16;
    int32_t generated_greedy[16];
    int32_t generated_sample[16];
    uint32_t rng = 7;

    forward(&model, &cache, tokens, prompt_n, out_act, NULL, NULL);
    int8_t* out_greedy = (int8_t*)malloc((size_t)H);
    memcpy(out_greedy, out_act, (size_t)H);

    double t_gen0 = now_ms();
    for (int32_t step = 0; step < n_gen; step++) {
        int32_t next = sample_token(&model, out_act, 0.0f, 0, &rng);
        generated_greedy[step] = next;
        if (step < n_gen - 1) forward_decode(&model, &cache, next, out_act, NULL);
    }
    double t_gen = now_ms() - t_gen0;
    printf("  greedy %d tokens (temp=0):    %8.3f ms   (%.3f ms/token)\n",
           n_gen, t_gen, t_gen / (n_gen - 1));
    printf("  tokens:                       ");
    for (int32_t i = 0; i < n_gen; i++) printf("%4d ", generated_greedy[i]);
    printf("\n");

    forward(&model, &cache, tokens, prompt_n, out_act, NULL, NULL);
    for (int32_t step = 0; step < n_gen; step++) {
        int32_t next = sample_token(&model, out_act, 1500.0f, 40, &rng);
        generated_sample[step] = next;
        if (step < n_gen - 1) forward_decode(&model, &cache, next, out_act, NULL);
    }
    int sampled_diverse = 0;
    for (int32_t i = 0; i < n_gen; i++) {
        if (generated_sample[i] != generated_greedy[i]) { sampled_diverse = 1; break; }
    }
    printf("  sampled %d tokens (T=1500 k=40): %s\n",
           n_gen, sampled_diverse ? "differs from greedy" : "matches greedy");
    printf("  tokens:                       ");
    for (int32_t i = 0; i < n_gen; i++) printf("%4d ", generated_sample[i]);
    printf("\n");

    // decoded text, non-printable as \xHH
    printf("  prompt:                       \"%s\"\n", prompt);
    printf("  greedy text:                  \"");
    for (int32_t i = 0; i < n_gen; i++) {
        int32_t b = generated_greedy[i] & 0xFF;
        if (b >= 32 && b < 127) printf("%c", b); else printf("\\x%02x", b);
    }
    printf("\"\n");
    printf("  sampled text:                 \"");
    for (int32_t i = 0; i < n_gen; i++) {
        int32_t b = generated_sample[i] & 0xFF;
        if (b >= 32 && b < 127) printf("%c", b); else printf("\\x%02x", b);
    }
    printf("\"\n");

    free(out_greedy); free(out_act); free(tokens);
    kv_cache_free(&cache);
    model_free(&model);
#if defined(__x86_64__) || defined(_M_X64)
    if (!(ok_avx2 && ok_vnni && ok_mt && ok_prep)) parity_failures++;
#endif
    if (parity_failures > 0) {
        printf("\nparity: %d failure(s)\n", parity_failures);
        return 1;
    }
    return 0;
}
