# Metal compute path

## what it is

GPU compute backend for the C engine on macOS, used where PyTorch has no backend at all (discrete AMD GPUs on Intel Macs and AMD eGPUs: MPS is Apple-Silicon-only, ROCm is Linux-only). Two pieces: an Objective-C bridge that probes the device and dispatches shaders, and the Metal Shading Language kernels themselves.

- Bridge: [`veritate_engine/v1/src/metal_dispatch.h`](../../veritate_engine/v1/src/metal_dispatch.h) + `metal_dispatch.m`. C-callable so the rest of the engine (plain C) links against it without ObjC.
- Shader: [`veritate_engine/v1/kernels/metal/matmul_int8.metal`](../../veritate_engine/v1/kernels/metal/matmul_int8.metal). Naive int8 matmul, one thread per output element, no threadgroup shared memory. Metal 1 family compatible.

`METAL_DISPATCH_AVAILABLE` is 1 on `__APPLE__` and 0 elsewhere, so the header compiles to no-ops on other platforms and the engine links unchanged.

## api

```c
void metal_detect(metal_caps_t* out);            // capability probe, never raises
void metal_print(const metal_caps_t* caps);      // stdout report, cpu_print() style
int  metal_matmul_int8(const int8_t* a, const int8_t* b, int32_t* c,
                       int32_t M, int32_t N, int32_t K, char* err, int err_cap);
int  metal_verify(void);                         // self-test vs scalar reference; 0 = PASS
```

`metal_caps_t` carries `available`, `n_devices`, `selected_index`, `selected_name`, the three
`supports_family_*` flags, `recommended_max_working_set`, `max_threads_per_threadgroup`, and an
`error` string populated when `available == 0`.

Shapes match the scalar reference in `kernels/scalar/matmul_scalar.c`: `a` is `[M x K]` row-major,
`b` is `[K x N]` row-major, `c[m,n] = sum_k a[m,k] * b[k,n]`.

## cli

Both subcommands are compiled in only under `METAL_DISPATCH_AVAILABLE` (`src/main.c`):

- `veritate metal-info`: device list, families, working set, max threads per threadgroup. Exit 0 when a device is available.
- `veritate verify-metal`: runs the int8 shader on a small matrix pair and bit-compares against the scalar CPU reference. Exit 0 on PASS.

## build

[`veritate_engine/v1/build/build.sh`](../../veritate_engine/v1/build/build.sh) pass 2.5 (macOS only):

1. The ObjC bridge always compiles on macOS so the `metal_*` symbols resolve at link time; links `-framework Metal -framework Foundation -framework CoreGraphics`.
2. Every `.metal` under `kernels/metal/` compiles to `.air` via `xcrun -sdk macosx metal -c`, then links to `default.metallib` via `xcrun metallib`.

`xcrun metal` ships with full Xcode, not the Command Line Tools. On a CLT-only machine the build prints that, the engine still builds, and the GPU path is runtime-disabled: `verify-metal` reports `default.metallib not found` rather than failing the build. To enable, install Xcode, `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`, and rebuild.

## detection

The platform-side hardware dump exposes the same facts to Python through `sys_metrics.detect_specs()`:

- `capabilities.can_use_metal`: true on any macOS host.
- `gpus[i].metal_family`: Metal feature family string (`Metal4`, `mac2`).
- `gpus[i].vendor`: `APPLE` / `AMD` / `NVIDIA`.
- `gpus[i].vram_total`: bytes; null on Apple-Silicon unified memory, populated on discrete AMD.

The Metal path is a candidate when `sys.platform == "darwin"` and at least one GPU has `vendor` in (`AMD`, `Apple`) with a non-null `metal_family`. NVIDIA-on-Mac keeps the nvidia-smi + PyTorch CUDA stack.

## pitfalls

- The shader is correctness-first, not tuned: one thread per output element, no tiling. Run `verify-metal` on each target GPU before trusting values; a wrong-answer GPU kernel raises no fault.
- Threadgroup sizing must come from `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`, not a constant: the limit differs per family.
- Buffer alignment differs by GPU family (64-byte vs 16-byte).
- Discrete AMD VRAM is small (3 GB per GPU on a FirePro D500). KV cache pressure caps usable model size around 80-200M params at INT8 with a realistic context window.
- Multi-GPU boxes get one command queue per device with no automatic data parallelism. Dispatch picks device 0.
- `dispatch.c` still routes matmul to the CPU kernels. Wiring Metal into the forward path means extending `cpu_features_t` with a `metal_available` field and adding the branch there.
