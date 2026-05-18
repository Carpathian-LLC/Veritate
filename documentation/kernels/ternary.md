# Ternary kernel

BitNet b1.58 weight matmul for the C engine. Per-tensor mean-abs scale, weight levels {-1, 0, +1}. Trits packed 5-per-byte at rest. Unpacked to int8 at compute time and dispatched through the same INT8 dot product the existing matmul kernels use.

This is a versioned contract. Adding, removing, or changing the signature of any function or field below requires updating this file in the same commit.

# ------------------------------------------------------------------------------------
# Rationale
# ------------------------------------------------------------------------------------

A 1B-class MEGA model with ternary weights is ~200 MB on disk vs ~1 GB INT8 vs ~4 GB FP32. The active per-byte footprint at top-1 routing is ~40 MB, which fits the 9800X3D's 96 MB L3 with attention + KV. Smaller than INT8 by 5x at the same accuracy band, this is the gating change that lets MEGA-class models run cache-resident on consumer hardware.

The trainer (`veritate_core/qat.py::fake_quant_weight_ternary`) does the QAT side; the engine implements the inference side. Accuracy is preserved because activation and output paths stay INT8: only the weight tier shrinks.

# ------------------------------------------------------------------------------------
# Packing format
# ------------------------------------------------------------------------------------

Pack 5 trits per byte using base-3 encoding:

```
byte = (((((t0+1) * 3 + (t1+1)) * 3 + (t2+1)) * 3 + (t3+1)) * 3 + (t4+1))
```

`t0..t4` ∈ {-1, 0, +1}. Output range: 0..242, so a uint8 byte holds the encoding exactly. Unpacking is the inverse:

```
t4 = (byte % 3) - 1; byte /= 3;
t3 = (byte % 3) - 1; byte /= 3;
t2 = (byte % 3) - 1; byte /= 3;
t1 = (byte % 3) - 1; byte /= 3;
t0 = byte - 1;
```

A row of `k` trits packs into `ceil(k / 5)` bytes. The last byte's tail trits past index k are written as 0 by the packer; the unpacker ignores them at consume time.

# ------------------------------------------------------------------------------------
# Prepped weight type
# ------------------------------------------------------------------------------------

```c
typedef struct {
    uint8_t* bt_packed;     // [n][ceil(k/5)]   row j holds 5 trits per byte
    int32_t* row_q24;       // [n]              per-row q24 multiplier for output requant
    float    gamma;          // per-tensor mean-abs scale (BitNet b1.58)
    int32_t  n;
    int32_t  k;
} prepped_b_ternary_t;
```

`row_q24` is symmetric in shape with the int4 path's `row_q24`: per-row scaling for the requant of int32 dot products into int8 outputs. The actual scale is derived from `gamma` and the model's `act_boost`, computed at prep time.

# ------------------------------------------------------------------------------------
# Kernel signatures
# ------------------------------------------------------------------------------------

```c
void prep_b_ternary(const int8_t* b_trits, int32_t n, int32_t k,
                    float gamma, prepped_b_ternary_t* out);
void free_prepped_b_ternary(prepped_b_ternary_t* p);

void matmul_ternary_scalar_prep(const int8_t* a, const prepped_b_ternary_t* p,
                                int32_t* c, int32_t m);
void matmul_ternary_vnni_prep(const int8_t* a, const prepped_b_ternary_t* p,
                              int32_t* c, int32_t m);
```

`b_trits` is a column-major int8 buffer where each value is in {-1, 0, +1}. The packer reads it, encodes into `bt_packed`, and frees its caller's buffer obligation.

`a` is row-major int8 activations of shape `[m x k]`.

`c` is row-major int32 output of shape `[m x n]`. Requant to int8 happens downstream via the same per-row `row_q24` multiplier the int4 path uses.

The scalar function is the rule-23 oracle. The VNNI function ships when it produces bit-identical int32 output on representative shapes.

Helpers exposed for testing:

```c
void ternary_pack_row(const int8_t* trits, int32_t k, uint8_t* out_bytes);
void ternary_unpack_row(const uint8_t* bytes, int32_t k, int8_t* out_trits);
```

# ------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------

`dispatch_init` selects the best available implementation at startup. Function-pointer aliases are not exposed at this stage (the matmul_int8 path remains the primary interface for non-ternary models). Models that flag ternary weights call the prep_ternary path explicitly through `model.c::forward_decode` once the MoE block lands.

A future bump will add a `quant_mode` field to the model header and route through the dispatch table per-block based on that field.

# ------------------------------------------------------------------------------------
# Memory budget
# ------------------------------------------------------------------------------------

A ternary linear of shape `[n=1280, k=1280]` (one MEGA expert's expert_up at hidden=1280) is:

- INT8: 1280 * 1280 = 1,638,400 bytes (~1.56 MB)
- INT4: 1280 * 1280 / 2 = 819,200 bytes (~0.78 MB)
- Ternary: 1280 * ceil(1280/5) = 1280 * 256 = 327,680 bytes (~0.31 MB)

5x smaller than INT8, 2.5x smaller than INT4. For 8 experts per layer at 12 layers (~98 expert linears total), the total weight footprint goes from ~150 MB INT8 down to ~30 MB ternary. This is the entire reason the MoE moonshot is shippable on the target hardware.

# ------------------------------------------------------------------------------------
# Bitwise parity contract
# ------------------------------------------------------------------------------------

Per claude_preflight rule 23: every kernel produces bitwise-identical output to its scalar reference before shipping. For the ternary path:

1. SIMD kernel must match `matmul_ternary_scalar_prep` int32-for-int32 on:
   - random trit weights at shapes `[n=1280, k=1280]`, `[n=3840, k=1280]`
   - random int8 activations in [-127, 127]
2. Round-trip parity for the packer: `ternary_unpack_row(ternary_pack_row(t)) == t` for all bit patterns.
3. End-to-end forward parity (phase E): cos-distance < 0.01 between engine and PyTorch MEGA forward at every layer, on a representative checkpoint.

# ------------------------------------------------------------------------------------
# What this contract does NOT cover
# ------------------------------------------------------------------------------------

- INT4 path (see `kernels/int4.md` -- not yet written; current source of truth is `veritate.h`'s `prepped_b_int4_t` block).
- MoE routing and per-expert dispatch (see `documentation/kernels/moe.md` -- to be written in phase C).
- Model format change to record `quant_mode` and `n_experts` (phase D, requires a build note).
- Confidence head, decision tracing, ablation: orthogonal, unaffected by this kernel.
