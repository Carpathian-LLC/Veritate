// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - cpu feature detection + runtime kernel selection. one branch per arch.
// - x86_64 reads cpuid leaves 7 / 0x80000002..04. arm64-darwin reads sysctlbyname
//   for FEAT_DotProd / FEAT_I8MM and machdep.cpu.brand_string. arm64-linux uses
//   getauxval(AT_HWCAP) (added when the linux build lands).
// - dispatch fills both the matmul function pointer and the transformer hot-path
//   pointers (score_dot_v / softmax_rows / layernorm_i16_to_i8). every backend
//   matches the typedef'd signature bit-for-bit.
// veritate_engine/src/dispatch.c
// ------------------------------------------------------------------------------------
// Imports:

#include "veritate.h"
#include "portability.h"

#include <stdio.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(_MSC_VER) || (defined(__clang__) && defined(_WIN32))
        #include <intrin.h>
        static void cpuidex(int regs[4], int leaf, int subleaf) {
            __cpuidex(regs, leaf, subleaf);
        }
    #else
        #include <cpuid.h>
        static void cpuidex(int regs[4], int leaf, int subleaf) {
            __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
        }
    #endif
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
    #include <sys/sysctl.h>
#endif

// ------------------------------------------------------------------------------------
// Constants

#define CPUID_LEAF_BRAND0  0x80000002
#define CPUID_LEAF_BRAND1  0x80000003
#define CPUID_LEAF_BRAND2  0x80000004
#define CPUID_LEAF_EXT     7
#define CPUID_BIT_AVX2     5
#define CPUID_BIT_AVX512F  16
#define CPUID_BIT_VNNI     11

// ------------------------------------------------------------------------------------
// Functions

// runtime-dispatched kernels. defaults compile-time-pick the safest impl that
// links on every build: scalar matmul on every arch, scalar transformer hot
// path always. dispatch_init may upgrade these to a SIMD backend at startup.
matmul_int8_fn          matmul_int8         = matmul_int8_scalar;
score_dot_v_fn          score_dot_v         = score_dot_v_scalar;
softmax_rows_fn         softmax_rows        = softmax_rows_scalar;
layernorm_i16_to_i8_fn  layernorm_i16_to_i8 = layernorm_i16_to_i8_scalar;

// ------------------------------------------------------------------------------------

void cpu_detect(cpu_features_t* out) {
    memset(out, 0, sizeof(*out));

#if defined(__x86_64__) || defined(_M_X64)
    int regs[4] = {0};

    cpuidex(regs, CPUID_LEAF_BRAND0, 0); memcpy(out->brand,      regs, 16);
    cpuidex(regs, CPUID_LEAF_BRAND1, 0); memcpy(out->brand + 16, regs, 16);
    cpuidex(regs, CPUID_LEAF_BRAND2, 0); memcpy(out->brand + 32, regs, 16);

    cpuidex(regs, CPUID_LEAF_EXT, 0);
    out->avx2        = (regs[1] >> CPUID_BIT_AVX2)    & 1;
    out->avx512f     = (regs[1] >> CPUID_BIT_AVX512F) & 1;
    out->avx512_vnni = (regs[2] >> CPUID_BIT_VNNI)    & 1;

#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
    out->neon = 1;
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) == 0) out->neon_sdot = v ? 1 : 0;
    sz = sizeof(v); v = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM",    &v, &sz, NULL, 0) == 0) out->neon_i8mm = v ? 1 : 0;
    size_t bsz = sizeof(out->brand);
    if (sysctlbyname("machdep.cpu.brand_string", out->brand, &bsz, NULL, 0) != 0) {
        snprintf(out->brand, sizeof(out->brand), "Apple Silicon");
    }

#elif defined(__aarch64__) || defined(_M_ARM64)
    // non-Apple arm64: assume neon, conservative on dot/i8mm. linux port reads HWCAP later.
    out->neon = 1;
    snprintf(out->brand, sizeof(out->brand), "ARM64");
#else
    snprintf(out->brand, sizeof(out->brand), "scalar");
#endif
}

// ------------------------------------------------------------------------------------

void cpu_print(const cpu_features_t* feat) {
    printf("cpu: %s\n", feat->brand);
    printf("features:");
    if (feat->avx2)        printf(" avx2");
    if (feat->avx512f)     printf(" avx512f");
    if (feat->avx512_vnni) printf(" avx512_vnni");
    if (feat->neon)        printf(" neon");
    if (feat->neon_sdot)   printf(" sdot");
    if (feat->neon_i8mm)   printf(" i8mm");
    printf("\n");
}

// ------------------------------------------------------------------------------------

void dispatch_init(const cpu_features_t* feat, dispatch_info_t* out) {
#if defined(__x86_64__) || defined(_M_X64)
    if (feat->avx512_vnni) {
        matmul_int8         = matmul_int8_vnni_mt;
        score_dot_v         = score_dot_v_avx512;
        softmax_rows        = softmax_rows_avx512;
        layernorm_i16_to_i8 = layernorm_i16_to_i8_avx512;
        out->matmul_backend = "avx512_vnni_int8_mt";
    } else if (feat->avx2) {
        matmul_int8         = matmul_int8_avx2;
        out->matmul_backend = "avx2_int8";
    } else {
        matmul_int8         = matmul_int8_scalar;
        out->matmul_backend = "scalar_int8";
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    // arm64 v1 ships only the NEON SDOT path. the prep'd matmul, transformer
    // hot-path, and inline attn helpers all use vdotq_s32 unconditionally; a
    // chip without FEAT_DotProd would SIGILL on the model hot path. fail loud
    // at startup rather than at the first instruction.
    if (!feat->neon_sdot) {
        fprintf(stderr, "veritate: arm64 build requires FEAT_DotProd. unsupported cpu.\n");
        matmul_int8         = matmul_int8_scalar;
        out->matmul_backend = "scalar_int8";
    } else {
        matmul_int8         = matmul_int8_neon_sdot;
        score_dot_v         = score_dot_v_neon;
        softmax_rows        = softmax_rows_neon;
        layernorm_i16_to_i8 = layernorm_i16_to_i8_neon;
        out->matmul_backend = "neon_sdot_int8";
    }
#else
    (void)feat;
    matmul_int8         = matmul_int8_scalar;
    out->matmul_backend = "scalar_int8";
#endif
}
