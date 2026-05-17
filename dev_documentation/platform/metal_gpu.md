# Metal compute path (AMD GPUs on macOS) — Phase 2

## Why

PyTorch ships no AMD-on-macOS backend (MPS is Apple-Silicon-only, ROCm is Linux-only). To use the discrete AMD GPUs on a 2013/2019 Mac Pro for training or inference we have to write our own Metal compute shaders. Same applies to AMD eGPUs on Intel Macs.

## Where it lives

Planned tree (not yet created):

```
veritate_engine/v1/kernels/metal/
  matmul.metal              # int8 matmul threadgroup-tiled
  attention.metal           # SDPA flash-style (Apple Silicon already covered by MPS in torch)
  layernorm.metal
  softmax.metal
  gelu.metal
  README.md                 # shader-version compatibility notes
veritate_engine/v1/src/
  metal_dispatch.{h,m}      # Objective-C bridge: device probe, shader compile, command queue
```

Dispatch wired into `src/dispatch.c` alongside the existing CPU paths. `cpu_features_t` extends to `accel_features_t` with a `metal_available` field.

## Detection

The hardware dump already surfaces what's needed via `sys_metrics.detect_specs()`:
- `capabilities.can_use_metal` — true on any macOS host
- `gpus[i].metal_family` — Apple's Metal feature family string (e.g. `Metal4`, `mac2`)
- `gpus[i].vendor` — `APPLE` / `AMD` / `NVIDIA`
- `gpus[i].vram_total` — bytes (null on Apple Silicon unified-memory; populated on discrete AMD)

The Metal dispatch path activates when:
1. `sys.platform == "darwin"` AND
2. At least one GPU with `vendor in ("AMD", "Apple")` AND `metal_family != null`

NVIDIA on Mac (rare; pre-Mojave) keeps using nvidia-smi + the existing PyTorch CUDA stack.

## Why this needs on-machine iteration

Metal shaders compile only on macOS with a target GPU available. Compile errors surface as cryptic Metal Shading Language messages that depend on the exact macOS / Xcode / driver version. Performance counters (Xcode Instruments → Metal System Trace) require physical access. Without that loop:
- shader bugs → silent wrong-answer (no SIGFPE on GPU)
- threadgroup sizing → wildly wrong without `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
- buffer alignment → 64-byte vs 16-byte differs by GPU family
- memory bandwidth bound vs compute bound depends on M-series vs FirePro vs Vega

Estimated effort with on-machine access:
- Matmul int8 shader (Metal 1 family compat for FirePro D500): 3–5 days
- SDPA / attention: 4–7 days (this is the most complex)
- Norm + softmax + GELU: 1–2 days each
- Dispatch glue + Objective-C bridge: 2 days
- **Total: 2–3 focused weeks** for a usable inference path. Training would add another 1–2 weeks for backward shaders + grad accumulation.

## What ships today (scaffolded, ready to iterate)

- Hardware dump exposes Metal availability + per-GPU details (`detect_specs().gpus[].metal_family`, `.capabilities.can_use_metal`).
- **ObjC bridge:** [`veritate_engine/v1/src/metal_dispatch.{h,m}`](../../veritate_engine/v1/src/) — device probe, library load, pipeline cache, dispatch helper. Always compiles on macOS, no-op stubs on other OSes.
- **First-pass shader:** [`veritate_engine/v1/kernels/metal/matmul_int8.metal`](../../veritate_engine/v1/kernels/metal/matmul_int8.metal) — naive int8 matmul, one thread per output element, no threadgroup shared memory. Compatible with Metal 1 family (Mac Pro 2013 Tahiti ceiling).
- **Build script** ([`veritate_engine/v1/build/build.sh`](../../veritate_engine/v1/build/build.sh)) compiles `.metal` → `.metallib` via `xcrun metal` + `xcrun metallib` when full Xcode is present. Falls back gracefully (engine still builds, GPU path is runtime-disabled) when only Xcode CLT is installed.
- **CLI subcommands:**
  - `veritate metal-info` — prints device list, families, working-set, max threads/threadgroup.
  - `veritate verify-metal` — runs the int8 matmul shader on a small test matrix and bit-compares against the scalar CPU reference. Reports PASS or first 8 mismatched indices.

## Prerequisites to actually use it

`xcrun metal` ships with **full Xcode**, not Xcode Command Line Tools. On a CLT-only machine the build prints exactly that, and `verify-metal` returns "library not found" cleanly.

To enable on Mac Pro 2013 (assuming Monterey):
1. Install Xcode from the App Store (Xcode 14 is the last compatible with Monterey).
2. `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
3. Rebuild: `bash veritate_engine/v1/build/build.sh`
4. Run: `veritate_engine/v1/bin/macos/x86_64/veritate verify-metal`

If `verify-metal` PASSes, the shader is correct on that GPU and we can move to the next kernel. If it FAILs (mismatched outputs) or returns a Metal error, capture the stderr and we iterate.

## Mac Pro 2013 / FirePro D500 specifics

- GPU is AMD Tahiti (GCN 1.0), 2× 3 GB VRAM, supports **Metal 2 family** (`spdisplays_mtlgpufamilymac2` confirmed from a real diagnostics dump).
- macOS Monterey 12.7.6 (confirmed). Last Xcode compatible: **Xcode 14**.
- Metal 2 family unlocks more than I assumed: SIMD-group instructions, fp16 math, indirect command buffers. Good news for shader perf.
- VRAM is only 3 GB per GPU; even a small model's KV cache will pressure it. Plan on aggressive int8 quantization + sequence-length capping.
- Two GPUs in the box: Metal command queues per device, no automatic data-parallel. We'd manually split work across them or pick one as primary.

## GPU **inference** roadmap (tractable: 3–4 weeks focused work)

Phase 2 scaffold is here. Today: bridge compiles, `metal-info` works, shader code exists but unvalidated. To get from scaffold to "Veritate runs inference on the AMD GPUs":

1. **Install full Xcode 14 on the Mac Pro.** `xcrun metal` lives in Xcode proper, not CLT. Then `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`, rebuild.
2. **`verify-metal` PASSes.** First signal the matmul shader is bit-correct against the CPU reference. May need 1–3 rounds of iteration on shader indexing or threadgroup size.
3. **Write the other forward-pass shaders:** attention (SDPA-style), layernorm, softmax, gelu, embedding lookup. ~1–2 weeks of focused on-machine work each iteration.
4. **Backend integration:** add `metal` as an inference device option alongside `pytorch` and `c_engine`. Load model weights to Metal buffers once at init, dispatch the forward path per generation step.
5. **Two-GPU strategy:** simplest is "use device 0, keep device 1 as warm spare." Real data-parallel across both is a stretch goal — would cut sequence-length budget further.

What this gets you: byte-level inference at GPU speed (probably 5–10× faster than CPU on this hardware for small models). The 3 GB VRAM ceiling means you're capped at roughly **80–200M params for INT8** with realistic context windows.

## GPU **training** roadmap (hard: 6–10 weeks, marginal value on this hardware)

Training on Metal needs more than inference:
- All forward shaders (same as above)
- Backward shaders for each op (gradient computation; matches PyTorch's autograd surface)
- Optimizer step shader (AdamW kernel)
- Gradient accumulation buffer management
- Loss computation + backward seed

This is essentially "build a tiny PyTorch for AMD-on-macOS." Reasonable for a research project; **not a great use of time on a 2013 Mac Pro specifically** because:
- 3 GB VRAM caps the trainable model size to ~10–30M params with QAT
- Even at full GPU utilization, you'd be ~3–5× faster than CPU on this hardware
- CPU on this box trains 10M models at ~1000 tok/s already (your latest log)

Stronger move: **train on CPU here, serve inference on GPU here.** That's the Phase 2 inference roadmap above and it's the natural arc.

## What Sam can do RIGHT NOW to unblock Phase 2

```
# 1. Install Xcode 14 from App Store on the Mac Pro.
# 2. Point CLT at it:
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# 3. Rebuild — should now produce default.metallib:
bash veritate_engine/v1/build/build.sh
# Expect: "build.sh: built /path/to/default.metallib"

# 4. First signal:
veritate_engine/v1/bin/macos/x86_64/veritate verify-metal
# Outcomes:
#   "PASS" — shader is bit-correct; we move on to attention
#   "FAIL N/64 outputs mismatched" — shader has a bug; paste output, I iterate
#   "command buffer error: ..." — Metal driver complaint; paste output, I iterate
#   "library not found" — Xcode not switched, see step 2
```

Each `verify-metal` run is one diagnostic message you paste back. We iterate from there. Estimate: 1–3 rounds to get PASS, then each subsequent shader is 2–4 rounds.
