// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - avx-512f + bw transformer hot-path kernels: score@V, softmax_rows, layernorm,
//   sparse single-row matmul for the post-gelu ffn_down path.
//   per-pair attn_dot / attn_hsum live inline in src/model.c (per-arch swap point).
// ------------------------------------------------------------------------------------

#include "../../src/veritate.h"

#include <immintrin.h>
#include <math.h>
#include <string.h>

void score_dot_v_avx512(const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out) {
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    for (int32_t j = 0; j < n_j; j++) {
        __m512i sj = _mm512_set1_epi32((int32_t)scores[j]);
        const int8_t* vp = v_base + (size_t)j * v_stride;
        a0 = _mm512_add_epi32(a0, _mm512_mullo_epi32(sj, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(vp +  0)))));
        a1 = _mm512_add_epi32(a1, _mm512_mullo_epi32(sj, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(vp + 16)))));
        a2 = _mm512_add_epi32(a2, _mm512_mullo_epi32(sj, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(vp + 32)))));
        a3 = _mm512_add_epi32(a3, _mm512_mullo_epi32(sj, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(vp + 48)))));
    }
    const __m512i half = _mm512_set1_epi32(16384);
    _mm_storeu_si128((__m128i*)(out +  0), _mm512_cvtsepi32_epi8(_mm512_srai_epi32(_mm512_add_epi32(a0, half), 15)));
    _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(_mm512_srai_epi32(_mm512_add_epi32(a1, half), 15)));
    _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(_mm512_srai_epi32(_mm512_add_epi32(a2, half), 15)));
    _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(_mm512_srai_epi32(_mm512_add_epi32(a3, half), 15)));
}

void softmax_rows_avx512(float* x, int16_t* out_q, int32_t rows, int32_t cols) {
    const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 ln2     = _mm512_set1_ps(0.6931471805599453f);
    const __m512 clamp   = _mm512_set1_ps(-87.0f);

    for (int32_t r = 0; r < rows; r++) {
        float*   row    = x     + (size_t)r * cols;
        int16_t* row_q  = out_q + (size_t)r * cols;

        __m512 vmax = _mm512_set1_ps(-1e38f);
        for (int32_t c = 0; c < cols; c += 16) {
            int32_t rem = cols - c;
            __mmask16 m = rem >= 16 ? 0xFFFF : (__mmask16)((1u << rem) - 1u);
            vmax = _mm512_mask_max_ps(vmax, m, vmax, _mm512_maskz_loadu_ps(m, row + c));
        }
        __m512 vmaxb = _mm512_set1_ps(_mm512_reduce_max_ps(vmax));

        __m512 vsum = _mm512_setzero_ps();
        for (int32_t c = 0; c < cols; c += 16) {
            int32_t rem = cols - c;
            __mmask16 m = rem >= 16 ? 0xFFFF : (__mmask16)((1u << rem) - 1u);
            __m512 v = _mm512_max_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, row + c), vmaxb), clamp);
            __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(v, inv_ln2), 0);
            __m512 r2 = _mm512_fnmadd_ps(n, ln2, v);
            __m512 e = _mm512_set1_ps(1.0f / 120.0f);
            e = _mm512_fmadd_ps(e, r2, _mm512_set1_ps(1.0f / 24.0f));
            e = _mm512_fmadd_ps(e, r2, _mm512_set1_ps(1.0f / 6.0f));
            e = _mm512_fmadd_ps(e, r2, _mm512_set1_ps(0.5f));
            e = _mm512_fmadd_ps(e, r2, _mm512_set1_ps(1.0f));
            e = _mm512_fmadd_ps(e, r2, _mm512_set1_ps(1.0f));
            v = _mm512_maskz_mov_ps(m, _mm512_scalef_ps(e, n));
            vsum = _mm512_add_ps(vsum, v);
            _mm512_mask_storeu_ps(row + c, m, v);
        }
        __m512 vqinv = _mm512_set1_ps(32768.0f / _mm512_reduce_add_ps(vsum));

        for (int32_t c = 0; c < cols; c += 16) {
            int32_t rem = cols - c;
            __mmask16 m = rem >= 16 ? 0xFFFF : (__mmask16)((1u << rem) - 1u);
            __m256i vi16 = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(
                _mm512_mul_ps(_mm512_maskz_loadu_ps(m, row + c), vqinv)));
            _mm256_mask_storeu_epi16(row_q + c, m, vi16);
        }
    }
}

void layernorm_i16_to_i8_avx512(const int16_t* x, int8_t* out, const int8_t* w,
                                int32_t rows, int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        const int16_t* row_in  = x   + (size_t)r * cols;
        int8_t*        row_out = out + (size_t)r * cols;

        __m512 vsumsq = _mm512_setzero_ps();
        for (int32_t c = 0; c < cols; c += 16) {
            __m256i in16 = _mm256_loadu_si256((const __m256i*)(row_in + c));
            __m512  v    = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(in16));
            vsumsq = _mm512_fmadd_ps(v, v, vsumsq);
        }
        float ms = _mm512_reduce_add_ps(vsumsq) / (float)cols;
        __m512 half_inv = _mm512_set1_ps(0.5f / sqrtf(ms + 1e-5f));

        for (int32_t c = 0; c < cols; c += 16) {
            __m256i in16 = _mm256_loadu_si256((const __m256i*)(row_in + c));
            __m512  v   = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(in16));
            __m512  wf  = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w + c))));
            __m512  vv  = _mm512_mul_ps(_mm512_mul_ps(v, wf), half_inv);
            _mm_storeu_si128((__m128i*)(row_out + c),
                _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(vv)));
        }
    }
}

// ------------------------------------------------------------------------------------
// sparse single-row matmul. m=1, decode-only. ffn_down hot path: input is
// post-gelu activation, 50-90% near-zero in trained models.
// pre-scans a for non-zeros, emits a compact (idx, val) list, then accumulates
// only the non-zero rows of b_rowmaj into c. bit-identical to the dense int32
// output (integer addition is associative; sum reordering preserves result).
// ------------------------------------------------------------------------------------

// prescan buffers sized at V_MAX_FFN so any runtime ffn dim up to the cap fits.
// caller (model_load) rejects shapes with ffn > V_MAX_FFN.
static int32_t s_nz_idx[V_MAX_FFN];
static int32_t s_nz_val[V_MAX_FFN];

static int32_t prescan_nonzero(const int8_t* a, int32_t k_dim,
                               int32_t* idx_out, int32_t* val_out) {
    int32_t n_nz = 0;
    for (int32_t k = 0; k < k_dim; k++) {
        int32_t v = a[k];
        if (v != 0) { idx_out[n_nz] = k; val_out[n_nz] = v; n_nz++; }
    }
    return n_nz;
}

static void sparse_accumulate(const prepped_b_t* p, int32_t n_nz,
                              const int32_t* idx, const int32_t* val, int32_t* c) {
    const int32_t N = p->n;
    const int8_t* B = p->b_rowmaj;
    memset(c, 0, (size_t)N * sizeof(int32_t));
    for (int32_t i = 0; i < n_nz; i++) {
        const int8_t* w_row = B + (size_t)idx[i] * N;
        __m512i av = _mm512_set1_epi32(val[i]);
        int32_t j = 0;
        for (; j + 64 <= N; j += 64) {
            __m512i o0 = _mm512_loadu_si512((const __m512i*)(c + j +  0));
            __m512i o1 = _mm512_loadu_si512((const __m512i*)(c + j + 16));
            __m512i o2 = _mm512_loadu_si512((const __m512i*)(c + j + 32));
            __m512i o3 = _mm512_loadu_si512((const __m512i*)(c + j + 48));
            __m512i w0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_row + j +  0)));
            __m512i w1 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_row + j + 16)));
            __m512i w2 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_row + j + 32)));
            __m512i w3 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_row + j + 48)));
            o0 = _mm512_add_epi32(o0, _mm512_mullo_epi32(av, w0));
            o1 = _mm512_add_epi32(o1, _mm512_mullo_epi32(av, w1));
            o2 = _mm512_add_epi32(o2, _mm512_mullo_epi32(av, w2));
            o3 = _mm512_add_epi32(o3, _mm512_mullo_epi32(av, w3));
            _mm512_storeu_si512((__m512i*)(c + j +  0), o0);
            _mm512_storeu_si512((__m512i*)(c + j + 16), o1);
            _mm512_storeu_si512((__m512i*)(c + j + 32), o2);
            _mm512_storeu_si512((__m512i*)(c + j + 48), o3);
        }
        for (; j + 16 <= N; j += 16) {
            __m512i o = _mm512_loadu_si512((const __m512i*)(c + j));
            __m512i w = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_row + j)));
            _mm512_storeu_si512((__m512i*)(c + j), _mm512_add_epi32(o, _mm512_mullo_epi32(av, w)));
        }
        for (; j < N; j++) c[j] += val[i] * (int32_t)w_row[j];
    }
}

void matmul_int8_sparse_decode(const int8_t* a, const prepped_b_t* p, int32_t* c) {
    int32_t n_nz = prescan_nonzero(a, p->k, s_nz_idx, s_nz_val);
    sparse_accumulate(p, n_nz, s_nz_idx, s_nz_val, c);
}

int32_t g_ffn_down_calls = 0;
int64_t g_ffn_down_nz_sum = 0;
int32_t g_ffn_down_sparse_calls = 0;

void ffn_down_decode(const int8_t* a, const prepped_b_t* p, int32_t* c) {
    int32_t n_nz = prescan_nonzero(a, p->k, s_nz_idx, s_nz_val);
    g_ffn_down_calls++;
    g_ffn_down_nz_sum += n_nz;
    if (p->b_rowmaj && n_nz * 2 < p->k) {
        g_ffn_down_sparse_calls++;
        sparse_accumulate(p, n_nz, s_nz_idx, s_nz_val, c);
    } else {
        matmul_int8_vnni_prep(a, p, c, 1);
    }
}
