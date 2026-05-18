// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - ObjC implementation of the Metal compute path. Only compiled on macOS.
//   Build script invokes clang with -ObjC and links -framework Metal
//   -framework Foundation.
// - First-pass: device probe + int8 matmul self-test. No model integration
//   yet. The path is intentionally verbose with err strings so that when
//   something goes wrong on an old GPU we get a useful traceback to iterate
//   on, not a silent return code.
// - default.metallib is expected to live alongside the executable. The
//   build script compiles kernels/metal/*.metal -> default.metallib and
//   places it next to the binary.
// veritate_engine/v1/src/metal_dispatch.m
// ------------------------------------------------------------------------------------

#include "metal_dispatch.h"

#if !METAL_DISPATCH_AVAILABLE

#include <string.h>
void metal_detect(metal_caps_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    snprintf(out->error, sizeof(out->error), "metal: this build was not compiled with Metal support");
}
void metal_print(const metal_caps_t* caps) {
    (void)caps;
    printf("metal: unavailable (not a macOS build)\n");
}
int metal_matmul_int8(const int8_t* a, const int8_t* b, int32_t* c,
                      int32_t M, int32_t N, int32_t K,
                      char* err, int err_cap) {
    (void)a; (void)b; (void)c; (void)M; (void)N; (void)K;
    if (err && err_cap > 0) snprintf(err, err_cap, "metal: not a macOS build");
    return -1;
}
int metal_verify(void) {
    printf("metal: unavailable (not a macOS build)\n");
    return -1;
}

#else // METAL_DISPATCH_AVAILABLE

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ------------------------------------------------------------------------------------
// Constants

#define METALLIB_BASENAME      "default.metallib"
#define MATMUL_FUNCTION_NAME   "matmul_int8"
#define VERIFY_M               8
#define VERIFY_N               8
#define VERIFY_K               16

// ------------------------------------------------------------------------------------
// Cached state. Lazily initialized on first metal_matmul_int8 call.

static id<MTLDevice>               g_device       = nil;
static id<MTLCommandQueue>         g_queue        = nil;
static id<MTLLibrary>              g_library      = nil;
static id<MTLComputePipelineState> g_matmul_pipe  = nil;
static int                         g_init_ok      = 0;
static char                        g_init_err[160] = {0};

// ------------------------------------------------------------------------------------

static void resolve_metallib_path(char* out, size_t cap) {
    // Look for default.metallib next to the running executable.
    char exe[1024]; uint32_t sz = (uint32_t)sizeof(exe);
    if (_NSGetExecutablePath(exe, &sz) != 0) {
        snprintf(out, cap, "./%s", METALLIB_BASENAME);
        return;
    }
    // Realpath to resolve any symlinks, then take dirname.
    char real[1024];
    if (!realpath(exe, real)) {
        snprintf(out, cap, "./%s", METALLIB_BASENAME);
        return;
    }
    char dir[1024];
    strncpy(dir, real, sizeof(dir) - 1); dir[sizeof(dir) - 1] = 0;
    char* d = dirname(dir);
    snprintf(out, cap, "%s/%s", d, METALLIB_BASENAME);
}


static id<MTLDevice> select_device(int* out_index) {
    // Prefer discrete (non-integrated, non-low-power) device when multiple
    // are present. Mac Pro 2013 has two FirePro D500s; both are discrete.
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
        if (out_index) *out_index = -1;
        return nil;
    }
    int picked = 0;
    for (NSUInteger i = 0; i < devices.count; i++) {
        id<MTLDevice> d = devices[i];
        if (!d.lowPower && !d.removable) { picked = (int)i; break; }
    }
    if (out_index) *out_index = picked;
    return devices[picked];
}


static int do_init(char* err, int err_cap) {
    if (g_init_ok) return 0;
    if (g_init_err[0]) {
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    int dev_idx = -1;
    id<MTLDevice> device = select_device(&dev_idx);
    if (!device) {
        snprintf(g_init_err, sizeof(g_init_err), "no Metal device found via MTLCopyAllDevices");
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        snprintf(g_init_err, sizeof(g_init_err), "newCommandQueue returned nil");
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    char libpath[1024];
    resolve_metallib_path(libpath, sizeof(libpath));
    NSString* path = [NSString stringWithUTF8String:libpath];
    NSError* nserr = nil;
    NSURL* url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&nserr];
    if (!library) {
        snprintf(g_init_err, sizeof(g_init_err),
                 "newLibraryWithURL failed at %s: %s",
                 libpath, nserr ? nserr.localizedDescription.UTF8String : "(no error)");
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@MATMUL_FUNCTION_NAME];
    if (!fn) {
        snprintf(g_init_err, sizeof(g_init_err),
                 "newFunctionWithName(%s) returned nil; check kernels/metal/*.metal compiled cleanly",
                 MATMUL_FUNCTION_NAME);
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    nserr = nil;
    id<MTLComputePipelineState> pipe =
        [device newComputePipelineStateWithFunction:fn error:&nserr];
    if (!pipe) {
        snprintf(g_init_err, sizeof(g_init_err),
                 "newComputePipelineStateWithFunction failed: %s",
                 nserr ? nserr.localizedDescription.UTF8String : "(no error)");
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_init_err);
        return -1;
    }
    g_device      = device;
    g_queue       = queue;
    g_library     = library;
    g_matmul_pipe = pipe;
    g_init_ok     = 1;
    return 0;
}


// ------------------------------------------------------------------------------------
// Public API

void metal_detect(metal_caps_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    out->n_devices = (int)devices.count;
    if (devices.count == 0) {
        snprintf(out->error, sizeof(out->error),
                 "MTLCopyAllDevices returned 0 devices (Metal stack not present?)");
        return;
    }
    int idx = -1;
    id<MTLDevice> dev = select_device(&idx);
    if (!dev) {
        snprintf(out->error, sizeof(out->error), "select_device returned nil after %d enumerated", out->n_devices);
        return;
    }
    out->available      = 1;
    out->selected_index = idx;
    strncpy(out->selected_name, dev.name.UTF8String, sizeof(out->selected_name) - 1);
    out->recommended_max_working_set = (uint64_t)dev.recommendedMaxWorkingSetSize;
    if (@available(macOS 10.15, *)) {
        out->supports_family_common = [dev supportsFamily:MTLGPUFamilyCommon1] ? 1 : 0;
        out->supports_family_mac    = [dev supportsFamily:MTLGPUFamilyMac1]    ? 1 : 0;
        out->supports_family_apple_silicon = [dev supportsFamily:MTLGPUFamilyApple1] ? 1 : 0;
    }
    // maxTotalThreadsPerThreadgroup is exposed via the function-level pipeline
    // state, not the device. We can't probe it without compiling a pipeline,
    // so do it best-effort here only if init has happened. Otherwise zero.
    if (g_init_ok && g_matmul_pipe) {
        out->max_threads_per_threadgroup = (uint32_t)g_matmul_pipe.maxTotalThreadsPerThreadgroup;
    }
}


void metal_print(const metal_caps_t* caps) {
    if (!caps) return;
    if (!caps->available) {
        printf("metal: unavailable (%s)\n", caps->error[0] ? caps->error : "no devices");
        return;
    }
    printf("metal: %s\n", caps->selected_name);
    printf("  device index: %d of %d\n", caps->selected_index, caps->n_devices);
    printf("  families: common=%d mac=%d apple=%d\n",
           caps->supports_family_common, caps->supports_family_mac, caps->supports_family_apple_silicon);
    if (caps->recommended_max_working_set)
        printf("  recommended working set: %.2f GB\n",
               (double)caps->recommended_max_working_set / (1024.0 * 1024.0 * 1024.0));
    if (caps->max_threads_per_threadgroup)
        printf("  max threads / threadgroup: %u\n", caps->max_threads_per_threadgroup);
}


int metal_matmul_int8(const int8_t* a, const int8_t* b, int32_t* c,
                      int32_t M, int32_t N, int32_t K,
                      char* err, int err_cap) {
    if (do_init(err, err_cap) != 0) return -1;
    if (!a || !b || !c || M <= 0 || N <= 0 || K <= 0) {
        if (err && err_cap > 0) snprintf(err, err_cap, "metal_matmul_int8: invalid args");
        return -2;
    }
    @autoreleasepool {
        size_t a_bytes = (size_t)M * (size_t)K;
        size_t b_bytes = (size_t)K * (size_t)N;
        size_t c_bytes = (size_t)M * (size_t)N * sizeof(int32_t);

        id<MTLBuffer> bufA = [g_device newBufferWithBytes:a length:a_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [g_device newBufferWithBytes:b length:b_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithLength:c_bytes
                                                  options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) {
            if (err && err_cap > 0)
                snprintf(err, err_cap, "newBuffer failed (a=%p b=%p c=%p)", bufA, bufB, bufC);
            return -3;
        }
        struct { int M, N, K; } params = { M, N, K };

        id<MTLCommandBuffer>        cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matmul_pipe];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger max_tpt = g_matmul_pipe.maxTotalThreadsPerThreadgroup;
        // Pick a 2D threadgroup shape that fits under the pipeline's cap. 16x16
        // (=256) is universally safe on every Metal device. Smaller on devices
        // that report a tighter cap.
        NSUInteger tx = 16, ty = 16;
        while (tx * ty > max_tpt && tx > 1) { if (ty > tx) ty >>= 1; else tx >>= 1; }
        MTLSize tg = MTLSizeMake(tx, ty, 1);
        MTLSize grid = MTLSizeMake((NSUInteger)M, (NSUInteger)N, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            if (err && err_cap > 0)
                snprintf(err, err_cap, "command buffer error: %s",
                         cmd.error.localizedDescription.UTF8String);
            return -4;
        }
        memcpy(c, bufC.contents, c_bytes);
    }
    return 0;
}


// ------------------------------------------------------------------------------------
// Self-test

extern void matmul_int8_scalar(const int8_t* a, const int8_t* b, int32_t* c,
                               int32_t M, int32_t N, int32_t K);

int metal_verify(void) {
    metal_caps_t caps;
    metal_detect(&caps);
    metal_print(&caps);
    if (!caps.available) {
        fprintf(stderr, "metal_verify: no Metal device. abort.\n");
        return -1;
    }
    int8_t  a[VERIFY_M * VERIFY_K];
    int8_t  b[VERIFY_K * VERIFY_N];
    int32_t c_metal[VERIFY_M * VERIFY_N];
    int32_t c_ref  [VERIFY_M * VERIFY_N];
    srand(1);
    for (size_t i = 0; i < sizeof(a); i++) a[i] = (int8_t)((rand() & 0xff) - 128);
    for (size_t i = 0; i < sizeof(b); i++) b[i] = (int8_t)((rand() & 0xff) - 128);

    matmul_int8_scalar(a, b, c_ref, VERIFY_M, VERIFY_N, VERIFY_K);

    char err[160] = {0};
    int rc = metal_matmul_int8(a, b, c_metal, VERIFY_M, VERIFY_N, VERIFY_K, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "metal_matmul_int8 failed rc=%d: %s\n", rc, err[0] ? err : "(no message)");
        return rc;
    }
    int mismatches = 0;
    for (int i = 0; i < VERIFY_M * VERIFY_N; i++) {
        if (c_metal[i] != c_ref[i]) {
            if (mismatches < 8) {
                fprintf(stderr, "  c[%d] metal=%d ref=%d\n", i, c_metal[i], c_ref[i]);
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("metal_verify: PASS (M=%d N=%d K=%d, %d outputs bit-match)\n",
               VERIFY_M, VERIFY_N, VERIFY_K, VERIFY_M * VERIFY_N);
        return 0;
    }
    printf("metal_verify: FAIL (%d/%d outputs mismatched)\n",
           mismatches, VERIFY_M * VERIFY_N);
    return 1;
}

#endif // METAL_DISPATCH_AVAILABLE
