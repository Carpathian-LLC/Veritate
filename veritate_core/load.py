# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Single factory that maps a checkpoint state_dict + cfg to a model instance.
# - The only place in the codebase allowed to branch on state-dict shape per
#   preflight rule 11a. Inference/decode code never sniffs variants; it asks
#   this factory once and then calls the contract methods on the returned model.
# veritate_core/load.py
# ------------------------------------------------------------------------------------
# Imports:


# ------------------------------------------------------------------------------------
# Constants

POS_EMB_KEY     = "pos_emb.weight"
TOK_EMB_KEY     = "tok_emb.weight"
MTP_PREFIX      = "mtp.transforms."
BLOCK_PREFIX    = "blocks."
DEFAULT_HEADS_DIVISOR = 64
ROPE_BASE_DEFAULT      = 10000.0
TRUNK_DENSE      = "dense"
TRUNK_RECURRENT  = "recurrent"
TRUNK_PATCHED    = "patched"
TRUNK_HYBRID     = "hybrid"
TRUNK_HYBRID_MOE = "hybrid_moe"
TRUNK_LOOPED     = "looped"


# ------------------------------------------------------------------------------------
# Functions


def shape_from_state_dict(sd, cfg):
    """Infer shape (vocab, hidden, layers, ffn, heads, seq) from a state_dict + cfg."""
    vocab, hidden = sd[TOK_EMB_KEY].shape
    if POS_EMB_KEY in sd:
        seq = sd[POS_EMB_KEY].shape[0]
    else:
        seq = int(cfg.get("seq") or 0)
        if seq <= 0:
            raise RuntimeError(
                "No pos_emb.weight in checkpoint and no seq in cfg/args. "
                "RoPE-based checkpoints must record `seq` in training_args."
            )
    layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith(BLOCK_PREFIX))
    ffn_per_layer = [sd[f"blocks.{L}.ff.up.weight"].shape[0] for L in range(layers)]
    ffn = ffn_per_layer[0] if all(f == ffn_per_layer[0] for f in ffn_per_layer) else ffn_per_layer
    heads = int(cfg.get("heads") or 0)
    if heads <= 0 or hidden % heads != 0:
        target = max(1, hidden // DEFAULT_HEADS_DIVISOR)
        for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                        key=lambda d: (abs(d - target), -d)):
            heads = h
            break
    return {"vocab": vocab, "hidden": hidden, "layers": layers,
            "ffn": ffn, "heads": heads, "seq": seq}


def _load_variant_trunk(sd, cfg, trunk, shape):
    """Research-trunk branch (trunk recorded in training_args). Patched/hybrid
    global-block count = total blocks minus the fixed local enc/dec blocks."""
    from veritate_core.model_recurrent import STATE_RULE_DEFAULT
    state_rule = str(cfg.get("state_rule") or STATE_RULE_DEFAULT)
    activation = cfg.get("activation") or _act_default()
    common = {"vocab": shape["vocab"], "hidden": shape["hidden"], "ffn": shape["ffn"],
                  "heads": shape["heads"], "seq": shape["seq"], "activation": activation}
    if trunk == TRUNK_RECURRENT:
        from veritate_core.model_recurrent import VeritateRecurrent
        model = VeritateRecurrent(layers=shape["layers"], state_rule=state_rule, **common)
    elif trunk in (TRUNK_PATCHED, TRUNK_HYBRID, TRUNK_HYBRID_MOE, TRUNK_LOOPED):
        from veritate_core.model_patched import (
            GLOBAL_FFN_MOE,
            GLOBAL_MIXER_RECURRENT,
            LOOP_MAX,
            LOOP_UNIQUE_DIV,
            N_LOCAL_DEC,
            N_LOCAL_ENC,
            VeritatePatched,
        )
        glob = shape["layers"] - N_LOCAL_ENC - N_LOCAL_DEC
        kwargs = {}
        if trunk in (TRUNK_HYBRID, TRUNK_HYBRID_MOE):
            kwargs["global_mixer"] = GLOBAL_MIXER_RECURRENT
            kwargs["state_rule"] = state_rule
        if trunk == TRUNK_HYBRID_MOE:
            kwargs["global_ffn"] = GLOBAL_FFN_MOE
        if trunk == TRUNK_LOOPED:
            kwargs["global_loops"] = int(cfg.get("global_loops") or LOOP_MAX)
            glob = glob * LOOP_UNIQUE_DIV  # constructor halves unique blocks when looping
        model = VeritatePatched(layers=glob, **common, **kwargs)
    else:
        raise RuntimeError(
            f"trunk '{trunk}' has no inference load branch (checkpoint-eval only)."
        )
    model.load_state_dict(sd, strict=True)
    return model


def _act_default():
    from veritate_core.model import ACT_DEFAULT
    return ACT_DEFAULT


def load_from_state_dict(sd, cfg, strict_canonical=True):
    """Construct the right Veritate model variant for the given state_dict and
    load it. Returns the constructed model with state_dict applied.

    The only allowed branch on state-dict shape lives here. Callers never name
    a model class.
    """
    if TOK_EMB_KEY not in sd:
        raise RuntimeError(
            "state_dict has no tok_emb.weight; not a Veritate checkpoint."
        )
    if any(k.startswith(MTP_PREFIX) for k in sd):
        # user-data compat: checkpoints from the retired veritate_800m / veritate_85m
        # trainers carry multi-byte heads; their model classes left with trainers/ on
        # 2026-08-18. They still export (training/export.py, v12) and serve on the C engine.
        raise RuntimeError(
            "this checkpoint carries multi-byte prediction heads (mtp.transforms.*) from the "
            "retired veritate_800m / veritate_85m trainers and cannot be loaded in PyTorch; "
            "export it to a .bin and serve it on the C engine instead")
    shape = shape_from_state_dict(sd, cfg)
    trunk = str((cfg or {}).get("trunk") or TRUNK_DENSE)
    if trunk != TRUNK_DENSE:
        return _load_variant_trunk(sd, cfg, trunk, shape)
    if POS_EMB_KEY not in sd:
        from veritate_core.model_rope import VeritateRoPE
        rope_base = float(cfg.get("rope_base") or ROPE_BASE_DEFAULT)
        model = VeritateRoPE(
            vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
            ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
            rope_base=rope_base,
        )
        model.load_state_dict(sd, strict=False)
        return model

    from veritate_core.model import ACT_DEFAULT, Veritate
    activation = cfg.get("activation") or ACT_DEFAULT
    model = Veritate(**shape, activation=activation)
    model.load_state_dict(sd, strict=strict_canonical)
    return model
