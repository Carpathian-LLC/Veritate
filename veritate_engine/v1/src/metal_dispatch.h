// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - Metal compute path probe + int8 matmul verifier. Implementation is
//   ObjC-on-macOS only (metal_dispatch.m); everywhere else this header
//   compiles to no-ops via the METAL_DISPATCH_AVAILABLE macro.
// - Public API kept C-callable so the rest of the engine (which is plain C)
//   can use it without dragging in ObjC.
// - Phase 2 status: scaffold only. The actual int8 matmul shader runs but
//   has not been validated for correctness or tuned for any specific GPU.
//   Treat outputs as "shape-correct, values-uncertain" until verify-metal
//   reports bit-match against the CPU reference on the target machine.
// veritate_engine/v1/src/metal_dispatch.h
// ------------------------------------------------------------------------------------

#ifndef VERITATE_METAL_DISPATCH_H
#define VERITATE_METAL_DISPATCH_H

#include <stdint.h>

#if defined(__APPLE__)
#define METAL_DISPATCH_AVAILABLE 1
#else
#define METAL_DISPATCH_AVAILABLE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------------------------
// Capability probe

typedef struct {
    int     available;           // 1 if at least one Metal-capable device was found
    int     n_devices;           // number of devices enumerated
    int     selected_index;      // device the dispatch will use (0 if multiple)
    char    selected_name[96];   // human-readable name (e.g. "AMD FirePro D500")
    int     supports_family_apple_silicon; // MTLGPUFamilyApple* member
    int     supports_family_mac;           // MTLGPUFamilyMac*  member
    int     supports_family_common;        // MTLGPUFamilyCommon* member
    uint64_t recommended_max_working_set;  // bytes; 0 if unknown
    uint32_t max_threads_per_threadgroup;  // 0 if probe failed
    char    error[160];          // human-readable reason when available=0
} metal_caps_t;

// Fill `out` with whatever the runtime can detect. Never raises. On non-macOS
// builds always returns available=0 and a populated error string.
void metal_detect(metal_caps_t* out);

// Print metal_caps_t to stdout in the same style as cpu_print().
void metal_print(const metal_caps_t* caps);

// ------------------------------------------------------------------------------------
// Int8 matmul via Metal compute shader. Same signature shape as the CPU
// matmul kernels so dispatch.c could swap in eventually. Returns 0 on
// success, non-zero on any Metal error; in failure cases `err` (if non-null)
// is filled with a short reason.
//
// SHAPES (matches CPU reference in kernels/scalar/matmul_scalar.c):
//   a: [M x K] row-major
//   b: [K x N] column-major
//   c: [M x N] row-major
//
// PHASE 2 NOTE: first-pass shader is naive (one thread per output element,
// no shared memory tiling). Correctness target before performance.

int metal_matmul_int8(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       M,
    int32_t       N,
    int32_t       K,
    char*         err,
    int           err_cap
);

// One-shot self-test: runs metal_matmul_int8 on a small random matrix pair
// and compares bit-for-bit against the scalar CPU reference. Prints PASS /
// FAIL with diagnostic info. Returns 0 on PASS, non-zero otherwise.
int metal_verify(void);

#ifdef __cplusplus
}
#endif

#endif // VERITATE_METAL_DISPATCH_H
