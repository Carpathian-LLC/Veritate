# MoE block

Mixture-of-Experts FFN for the C engine. Replaces the standard `ffn_up` / `ffn_down` pair with a per-block router and N independent expert FFNs. Top-1 routing only today: each token's hidden state is dot-producted against a `[n_experts x hidden]` router matrix; the argmax expert handles the FFN for that token. Top-K > 1 (weighted combine) is reserved for a follow-up.

This is a versioned contract. Adding, removing, or changing the signature of any function or field below requires updating this file in the same commit.

# ------------------------------------------------------------------------------------
# Rationale

Total parameters scale linearly with `n_experts` while active parameters per byte stay near a single expert's footprint. Combined with ternary weights ([documentation/kernels/ternary.md](ternary.md)), an 8-expert 1B-class model has a per-byte active footprint that fits the 96 MB L3 of the 9800X3D. Without MoE, the same accuracy band requires a dense 1B model that misses cache catastrophically.

MEGA is the canonical MoE trainer (`trainers/veritate_mega/`). It produces top-1 8-expert checkpoints. The engine reads those.

# ------------------------------------------------------------------------------------
# block layout

```c
typedef struct {
    /* ... attention path unchanged ... */
    int32_t      n_experts;     // 1 = no MoE
    int32_t      router_topk;   // experts active per token; 1 = sticky single-expert
    prepped_b_t  router;        // [n_experts x hidden]
    prepped_b_t* experts_up;    // [n_experts] of [ffn x hidden]
    prepped_b_t* experts_down;  // [n_experts] of [hidden x ffn]
} block_t;
```

When `n_experts == 1`, `experts_up` and `experts_down` are NULL and the standard `ffn_up` / `ffn_down` fields hold the single FFN. Hot path is unchanged for non-MoE models.

When `n_experts > 1`, the per-expert pointers are populated and the router matrix is allocated. The standard `ffn_up` / `ffn_down` are unused.

# ------------------------------------------------------------------------------------
# v11 binary layout

Per-block, after `ln2_w`:

```
when n_experts == 1:
  ffn_up   [ffn x hidden]      (existing v9 layout)
  ffn_down [hidden x ffn]      (existing v9 layout)

when n_experts > 1:
  router   [n_experts x hidden]   (uniform per-tensor scale)
  for each expert e in 0..n_experts:
    experts_up[e]   [ffn x hidden]    (uniform per-tensor scale)
    experts_down[e] [hidden x ffn]    (uniform per-tensor scale)
```

Per-row scales are not used in v11 -- uniform per-tensor scale per weight block. The exporter writes the calibrated scale into the prepped block's `scale_q24` field; the loader's `load_b()` reads it.

# ------------------------------------------------------------------------------------
# routing

Top-1 routing is the only mode wired today:

```c
static int32_t route_token_top1(const block_t* blk, const int8_t* hidden_row,
                                int32_t scratch_n_experts, int32_t* scratch);
```

Computes `router . hidden_row -> n_experts logits`, returns the argmax. The activation passed in is the post-`ln2` int8 buffer, the same input the standard `ffn_up` reads. Soft routing (full softmax + sampling) is not used; argmax suffices for top-1 because softmax-then-argmax is equivalent.

The loader refuses any v11 binary with `router_topk > 1` -- the multi-expert weighted combine path is reserved.

# ------------------------------------------------------------------------------------
# forward integration

`forward_decode` (single-token decode) consults `blk->n_experts` after `ln2`. When > 1, it routes the token, picks the expert, and dispatches `ffn_up` and `ffn_down` against that expert's `prepped_b_t` blocks. The rest of the forward path (attention, residual, lens projection, ablation hook) is unchanged.

`forward` (prefill) and `forward_verify` (multi-token verify) detect MoE and fall back to per-token decode when any block has `n_experts > 1`. Correct but not yet performance-tuned; a batched MoE prefill is a follow-up.

# ------------------------------------------------------------------------------------
# bitwise parity contract

Per claude_preflight rule 23: every kernel produces bitwise-identical output to its scalar reference before shipping. For the MoE path:

1. Top-1 expert index from the engine's router must match PyTorch's `top_i[:, 0]` for the same token, given the same router weights and hidden state.
2. Per-expert FFN output (post-`ffn_down` int32) must match the corresponding PyTorch expert's output bit-for-bit, since the kernel is the same INT8 VNNI matmul.
3. Block residual after the MoE FFN must match the PyTorch MEGA forward at every layer, on a representative checkpoint, with cosine distance < 0.01.

Validation is gated on the export pipeline writing a v11 binary from a QAT'd MEGA checkpoint.

# ------------------------------------------------------------------------------------
# what this contract does NOT cover

- Top-K > 1 weighted combine. Reserved for a follow-up; the loader refuses any v11 with `router_topk > 1`.
- Ternary expert weights. The kernel exists ([documentation/kernels/ternary.md](ternary.md)) and is wired for the non-MoE FFN path (`load_b_ternary`), but the forward path that calls it inside an MoE block is not yet wired. The loader refuses any v11 MoE binary with `quant_mode != INT8`.
- Batched MoE prefill (per-expert grouped matmul over the prefill rows that route to the same expert). Falls back to per-token decode today.
- Per-expert load-balance auxiliary loss. Trainer-side concern; the engine does not see it.
- Router weight per-row scales. v11 uses uniform per-tensor scale only.

# ------------------------------------------------------------------------------------
# update obligation

Adding, removing, or renaming any field above requires:

1. The implementation in `veritate_engine/v1/src/model.c` (`route_token_top1`, the FFN dispatch in `forward_decode`, and the v11 load path) is updated.
2. The block_t / model_t fields in `veritate_engine/v1/src/veritate.h` are updated.
3. The export pipeline in `veritate_mri/export.py` is updated to match the on-disk layout.
4. This file's tables and code samples are updated in the same commit.
5. A build note ships the format change per claude_preflight rule 43.
