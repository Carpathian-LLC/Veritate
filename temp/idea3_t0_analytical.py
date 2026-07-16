# IDEA 3 T0: analytical FLOP breakdown for the current shipped models.
# Zero GPU. Arithmetic only. Names the dominant per-step FLOP cost so T1/T2/T3
# lever choice is informed rather than guessed.

import json

# ------------------------------------------------------------------------------------
# Model configs (post-training)

MODELS = {
    "10m_dense": {
        "trunk": "dense", "hidden": 320, "layers": 8, "ffn": 1280, "heads": 8,
        "seq": 512, "batch": 64, "n_chunks": 4, "vocab": 256,
        # patched-specific
        "patch_size": None, "n_global_layers": None, "n_enc_dec_layers": None,
    },
    "80m_hybrid": {
        "trunk": "hybrid", "hidden": 768, "layers": 12, "ffn": 3072, "heads": 12,
        "seq": 1024, "batch": 16, "n_chunks": 2, "vocab": 256,
        # patched-specific
        "patch_size": 4, "n_global_layers": 12, "n_enc_dec_layers": 4,
    },
}


# ------------------------------------------------------------------------------------
# FLOP formulas

def flops_attention_dense(B, T, H, heads, layers):
    """Dense self-attention: QKV proj + attn scores + AV + out proj, per layer."""
    head_dim = H // heads
    qkv = 3 * 2 * B * T * H * H           # QKV projections
    scores = 2 * B * heads * T * T * head_dim
    av = 2 * B * heads * T * T * head_dim
    proj = 2 * B * T * H * H              # output projection
    return layers * (qkv + scores + av + proj)


def flops_attention_patched(B, T, H, heads, patch_size, n_global_layers, n_enc_dec_layers):
    """Patched trunk: enc/dec run at T bytes (local), globals run at T/patch_size."""
    head_dim = H // heads
    # Encoder + decoder local attention over T bytes (small window)
    local_qkv = 3 * 2 * B * T * H * H
    local_scores = 2 * B * heads * T * T * head_dim
    local_av = 2 * B * heads * T * T * head_dim
    local_proj = 2 * B * T * H * H
    local_per_layer = local_qkv + local_scores + local_av + local_proj

    # Global blocks at T/patch_size positions
    P = T // patch_size
    g_qkv = 3 * 2 * B * P * H * H
    g_scores = 2 * B * heads * P * P * head_dim
    g_av = 2 * B * heads * P * P * head_dim
    g_proj = 2 * B * P * H * H
    global_per_layer = g_qkv + g_scores + g_av + g_proj

    return n_enc_dec_layers * local_per_layer + n_global_layers * global_per_layer


def flops_ffn(B, T, H, ffn, layers):
    """SwiGLU-style FFN: 3 projections (gate, up, down) per layer."""
    return layers * (3 * 2 * B * T * H * ffn)


def flops_ffn_patched(B, T, H, ffn, patch_size, n_global_layers, n_enc_dec_layers):
    """Patched: enc/dec FFN at T, global FFN at T/patch_size."""
    P = T // patch_size
    return (n_enc_dec_layers * (3 * 2 * B * T * H * ffn)
            + n_global_layers  * (3 * 2 * B * P * H * ffn))


def flops_embed(B, T, H, vocab):
    """Embedding lookup + tied LM head: 2 * B * T * H * vocab for the head."""
    return 2 * B * T * H * vocab


def flops_optim_step(n_params):
    """AdamW update (2 reads, 2 writes on m/v, one weight update): ~10x n_params FLOPs."""
    return 10 * n_params


def flops_gla_scan(B, T, H, layers):
    """GLA (gated linear attention) recurrent scan per global position: constant per step,
    but summed over positions. ~5x H^2 per position (Q, K, V, G, gate)."""
    return layers * (5 * B * T * H * H)


# ------------------------------------------------------------------------------------
# Estimate helpers

def params_transformer(hidden, layers, ffn, vocab, patch_layers_extra=0):
    per_block_attn = 4 * hidden * hidden
    per_block_ffn  = 3 * hidden * ffn
    per_block = per_block_attn + per_block_ffn
    embed = vocab * hidden + hidden * vocab
    return layers * per_block + embed + patch_layers_extra


def analyze(name, cfg):
    B = cfg["batch"]
    T = cfg["seq"]
    H = cfg["hidden"]
    ffn = cfg["ffn"]
    heads = cfg["heads"]
    layers = cfg["layers"]
    vocab = cfg["vocab"]
    n_chunks = cfg["n_chunks"]
    # Total tokens processed per step:
    tokens_per_step = B * T * n_chunks

    if cfg["trunk"] == "dense":
        n_params = params_transformer(H, layers, ffn, vocab)
        attn = flops_attention_dense(B, T, H, heads, layers)
        ffn_f = flops_ffn(B, T, H, ffn, layers)
        gla   = 0
    else:
        # Patched hybrid: enc + dec local + global recurrent
        n_global = cfg["n_global_layers"]
        n_ed     = cfg["n_enc_dec_layers"]
        n_params = params_transformer(H, n_global + n_ed, ffn, vocab)
        attn = flops_attention_patched(B, T, H, heads, cfg["patch_size"], n_global, n_ed)
        ffn_f = flops_ffn_patched(B, T, H, ffn, cfg["patch_size"], n_global, n_ed)
        gla   = flops_gla_scan(B, T // cfg["patch_size"], H, n_global)  # global path only

    embed = flops_embed(B, T, H, vocab)
    fwd = attn + ffn_f + gla + embed
    # backward is ~2x forward
    bwd = 2 * fwd
    step_flops = (fwd + bwd) * n_chunks
    optim = flops_optim_step(n_params)
    total = step_flops + optim

    def pct(x): return 100.0 * x / total

    return {
        "name": name,
        "trunk": cfg["trunk"],
        "n_params": n_params,
        "tokens_per_step": tokens_per_step,
        "fwd_flops": fwd,
        "bwd_flops": bwd,
        "step_flops_all_chunks": step_flops,
        "optim_flops": optim,
        "total_flops_per_step": total,
        "share_attention_pct":  pct((attn) * n_chunks * 3),
        "share_ffn_pct":        pct((ffn_f) * n_chunks * 3),
        "share_gla_pct":        pct((gla) * n_chunks * 3) if gla else 0.0,
        "share_embed_pct":      pct((embed) * n_chunks * 3),
        "share_optim_pct":      pct(optim),
        "flops_per_token":      total / tokens_per_step,
    }


# ------------------------------------------------------------------------------------

print("=" * 70)
print("IDEA 3 T0: analytical FLOP breakdown (arithmetic only, no GPU)")
print("=" * 70)
for name, cfg in MODELS.items():
    r = analyze(name, cfg)
    print(f"\n--- {name} ({r['trunk']}) ---")
    print(f"  params:            {r['n_params']:>15,}")
    print(f"  tokens/step:       {r['tokens_per_step']:>15,}")
    print(f"  fwd FLOPs:         {r['fwd_flops']:>15,.0f}")
    print(f"  bwd FLOPs:         {r['bwd_flops']:>15,.0f}")
    print(f"  step FLOPs (all):  {r['step_flops_all_chunks']:>15,.0f}")
    print(f"  optim FLOPs:       {r['optim_flops']:>15,.0f}")
    print(f"  TOTAL per step:    {r['total_flops_per_step']:>15,.0f}")
    print(f"  FLOPs/token:       {r['flops_per_token']:>15,.0f}   (~6*params rule of thumb: {6*r['n_params']:,})")
    print(f"  share attention:   {r['share_attention_pct']:>6.1f}%")
    print(f"  share FFN:         {r['share_ffn_pct']:>6.1f}%")
    print(f"  share GLA scan:    {r['share_gla_pct']:>6.1f}%")
    print(f"  share embed/head:  {r['share_embed_pct']:>6.1f}%")
    print(f"  share optim:       {r['share_optim_pct']:>6.1f}%")

print(f"\n{'='*70}")
print("LEVER SELECTION (per ideas.md IDEA 3):")
print("  T1 hybrid_moe : cuts FFN cost via expert sparsity. Fund IF FFN dominant.")
print("  T2 adaptive patching : cuts global-block positions. Fund IF attention dominant.")
print("  T3 mixture-of-depths : skips layers on easy tokens. Fund IF layer count is the wall.")
