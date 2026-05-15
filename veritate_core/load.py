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

import importlib.util
import os

# ------------------------------------------------------------------------------------
# Constants

POS_EMB_KEY     = "pos_emb.weight"
TOK_EMB_KEY     = "tok_emb.weight"
MTP_PREFIX      = "mtp.transforms."
BLOCK_PREFIX    = "blocks."
DEFAULT_HEADS_DIVISOR = 64
N_PREDICT_DEFAULT_800M = 4
N_PREDICT_DEFAULT_85M  = 2
ROPE_BASE_DEFAULT      = 10000.0
TRAINERS_SUBDIR = "trainers"
TRAINER_800M_DIR = "veritate_800m"
TRAINER_85M_DIR  = "veritate_85m"
PLUGIN_MODULE_FILENAME = "trainer.py"

_TRAINER_CLASS_CACHE = {}

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


def _import_trainer_model(trainer_dirname, class_name):
    key = (trainer_dirname, class_name)
    cached = _TRAINER_CLASS_CACHE.get(key)
    if cached is not None:
        return cached
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    plugin_path = os.path.join(repo_root, TRAINERS_SUBDIR, trainer_dirname, PLUGIN_MODULE_FILENAME)
    mod_name = f"veritate_trainer_{trainer_dirname}"
    spec = importlib.util.spec_from_file_location(mod_name, plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    _TRAINER_CLASS_CACHE[key] = cls
    return cls


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
    shape = shape_from_state_dict(sd, cfg)
    has_pos_emb = POS_EMB_KEY in sd
    has_mtp = any(k.startswith(MTP_PREFIX) for k in sd)

    if not has_pos_emb and has_mtp:
        Veritate800M = _import_trainer_model(TRAINER_800M_DIR, "Veritate800M")
        n_predict = int(cfg.get("n_predict") or N_PREDICT_DEFAULT_800M)
        rope_base = float(cfg.get("rope_base") or ROPE_BASE_DEFAULT)
        model = Veritate800M(
            vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
            ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
            n_predict=n_predict, rope_base=rope_base,
        )
        model.load_state_dict(sd, strict=False)
        return model

    if has_pos_emb and has_mtp:
        Veritate85M = _import_trainer_model(TRAINER_85M_DIR, "Veritate85M")
        n_predict = int(cfg.get("n_predict") or N_PREDICT_DEFAULT_85M)
        model = Veritate85M(
            vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
            ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
            n_predict=n_predict,
        )
        model.load_state_dict(sd, strict=True)
        return model

    if not has_pos_emb and not has_mtp:
        from veritate_core.model_rope import VeritateRoPE
        rope_base = float(cfg.get("rope_base") or ROPE_BASE_DEFAULT)
        model = VeritateRoPE(
            vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
            ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
            rope_base=rope_base,
        )
        model.load_state_dict(sd, strict=False)
        return model

    from veritate_core.model import Veritate
    model = Veritate(**shape)
    model.load_state_dict(sd, strict=strict_canonical)
    return model
