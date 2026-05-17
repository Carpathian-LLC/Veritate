// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - First-pass int8 -> int32 matmul. One thread per output element. No
//   threadgroup shared memory. Naive on purpose: correctness before tuning.
// - Compatible with Metal 1 family (Mac Pro 2013 AMD Tahiti). Avoids
//   simdgroup intrinsics, fp16 atomics, and other newer-family features.
// - Shape convention: a is [M x K] row-major, b is [K x N] column-major,
//   c is [M x N] row-major. Same as kernels/scalar/matmul_scalar.c so
//   verify_metal can bit-compare against it.
// - PHASE 2: unverified on real hardware. expected to compile cleanly but
//   may need driver-specific tweaks. paired with src/metal_dispatch.m.
// veritate_engine/v1/kernels/metal/matmul_int8.metal
// ------------------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

// Parameter block kept tiny to fit within a single buffer-binding slot.
// MUST match the layout in metal_dispatch.m.
struct matmul_int8_params {
    int M;
    int N;
    int K;
};

// One thread = one output element c[m,n]. Grid is (M, N, 1).
// Threadgroup size is host-chosen via maxTotalThreadsPerThreadgroup; the
// kernel itself doesn't constrain it.
kernel void matmul_int8(
    device const char*                 a   [[buffer(0)]],
    device const char*                 b   [[buffer(1)]],
    device int*                        c   [[buffer(2)]],
    constant matmul_int8_params&       p   [[buffer(3)]],
    uint2                              gid [[thread_position_in_grid]]
) {
    int m = int(gid.x);
    int n = int(gid.y);
    if (m >= p.M || n >= p.N) return;

    int acc = 0;
    int row_base = m * p.K;
    int col_base = n * p.K;
    for (int k = 0; k < p.K; ++k) {
        // a is row-major: a[m,k] = a[m*K + k]
        // b is column-major: b[k,n] = b[n*K + k]
        int av = int(a[row_base + k]);
        int bv = int(b[col_base + k]);
        acc += av * bv;
    }
    c[m * p.N + n] = acc;
}
