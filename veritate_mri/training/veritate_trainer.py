# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - THE trainer. There is exactly one, and it ships with the platform: the separate
#   Veritate-Trainers repo was retired 2026-07-29 (it existed to distribute 19
#   per-size trainers, which were deleted 2026-07-27 for being byte-identical apart
#   from a PLUGIN_ID string). This module owns the training loop, the LR schedule,
#   the data loader, the chunked forward, the resume logic, the QAT and 8-bit-Adam
#   plumbing, and the config.json bootstrap.
# - Every size, one trainer. Shapes and tuned defaults are DATA, in
#   veritate_mri/data/trainer_sizes.json. Adding or retuning a size means editing
#   that JSON, never touching this file. A literal tunable in here is a bug.
# - This file IS the entry point, so it puts the repo root on sys.path itself
#   rather than relying on a caller to have done it.
# veritate_mri/training/veritate_trainer.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time

# veritate_mri/training -> veritate_mri -> repo root, so `veritate_core` imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# _HERE is on sys.path (set above), so this resolves whether the trainer is run
# as a script or imported as a module.
import grow as grow_mod
import numpy as np
import torch

from veritate_core.plugin import bench, hardware, multicorpus, paths, save
from veritate_core.plugin import model as _model_mod
from veritate_core.plugin import qat as qat_helpers

# Optional: shared optimizer builder (muon) and patched-trunk variant.
# Feature-detect so this trainer still runs on an older platform.
try:
    from veritate_core.plugin import optim as optim_helpers
except ImportError:
    optim_helpers = None
try:
    from veritate_core.plugin import model_patched as _model_patched_mod
    VeritatePatched = _model_patched_mod.VeritatePatched
except ImportError:
    _model_patched_mod = None
    VeritatePatched = None
try:
    from veritate_core.plugin import model_recurrent as _model_recurrent_mod
    VeritateRecurrent = _model_recurrent_mod.VeritateRecurrent
except ImportError:
    VeritateRecurrent = None
try:
    from veritate_core.plugin import model_memory as _model_memory_mod
    VeritateMemory = _model_memory_mod.VeritateMemory
except ImportError:
    VeritateMemory = None
try:
    from veritate_core.plugin import slm as slm_helpers
except ImportError:
    slm_helpers = None
# Optional: NVMe optimizer paging needs a platform new enough to expose the memory
# planner/executor offload APIs. Feature-detect so this trainer still runs on an
# older platform (it just falls back to the pre-paging behavior).
try:
    from veritate_core.plugin import mem_executor, mem_planner
    _MEM_PAGING = (hasattr(mem_executor, "make_optimizer")
                   and hasattr(mem_executor, "OFFLOAD_TIERS")
                   and hasattr(mem_planner, "plan_training_memory")
                   and hasattr(bench, "plan_result"))
except Exception:
    mem_planner = None
    mem_executor = None
    _MEM_PAGING = False

Veritate         = _model_mod.Veritate
VOCAB_BYTE_LEVEL = _model_mod.VOCAB_BYTE_LEVEL

append_train_row       = save.append_train_row
compose_name           = save.compose_name
hash_corpus            = save.hash_corpus
require_description    = save.require_description
resolve_corpus         = save.resolve_corpus
truncate_train_csv_at  = save.truncate_train_csv_at


# ------------------------------------------------------------------------------------
# Constants

SHAPE_FIELDS    = ("layers", "hidden", "ffn", "heads")
BASE_CKPT_PREFIX = "step_"
BASE_CKPT_SUFFIX = ".pt"

LR_SCHEDULES    = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")
PRECISIONS      = ("fp32", "bf16")

STATE_CARRIES      = ("off", "chunks")
STATE_CARRY_CHUNKS = "chunks"
STATE_CARRY_TRUNKS = ("hybrid", "hybrid_moe", "hybrid_monarch", "hybrid_pkm",
                      "hybrid_pkm_fire", "recurrent")

# Weights, grads, and Adam moments are all fp32 here (autocast keeps fp32 master
# weights even at bf16 precision), so the memory planner sizes its buckets in fp32.
PLAN_DTYPE      = "fp32"
PAGED_STATE_DIR = "optim_state"


# ------------------------------------------------------------------------------------
# Functions

SIZES_FILE          = "trainer_sizes.json"
SHARED_DEFAULTS_KEY = "shared_defaults"
DEFAULT_SIZE_KEY    = "default_size"
SIZES_KEY           = "sizes"
SHAPE_KEY           = "shape"
DEFAULTS_KEY        = "defaults"


def _sizes_path(here):
    """veritate_mri/data/trainer_sizes.json is the home, and it is the path
    readers/trainers.py resolves too, so the set the dashboard offers can never
    drift from the set this trainer accepts. The beside-the-trainer fallback is
    kept for older checkouts synced from the retired Veritate-Trainers repo,
    where the table shipped next to the trainer instead."""
    for cand in (os.path.join(_HERE, "..", "data", SIZES_FILE),
                 os.path.join(here, SIZES_FILE)):
        cand = os.path.normpath(cand)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "no " + SIZES_FILE + " found. Expected it at veritate_mri/data/" + SIZES_FILE
        + " (it owns every size and tuned default).")


def _load_manifest(here):
    """There is ONE trainer and no per-size plugin folders. Shapes and tuned
    defaults come from trainer_sizes.json, so nothing tuned is hardcoded here and
    adding a size means editing that JSON only. A manifest.json beside the trainer
    still wins, for anyone pinning a custom set without touching the shared table."""
    local = os.path.join(here, "manifest.json")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    with open(_sizes_path(here), encoding="utf-8") as f:
        doc = json.load(f)
    return {
        "name": "Veritate", "kind": "trainer", "bench": True,
        "flow": ["scratch", "continue"],
        "description": "Canonical Veritate trainer. Every size, one trainer.",
        "sizes": {k: v[SHAPE_KEY] for k, v in doc[SIZES_KEY].items()},
        "defaults": {**(doc.get(SHARED_DEFAULTS_KEY) or {}),
                     "size": doc.get(DEFAULT_SIZE_KEY) or ""},
        "size_defaults": {k: v.get(DEFAULTS_KEY) or {} for k, v in doc[SIZES_KEY].items()},
    }


def _size_presets(manifest):
    return {
        k: {field: v[field] for field in SHAPE_FIELDS}
        for k, v in (manifest.get("sizes") or {}).items()
    }


def apply_size_defaults(args, manifest, argv):
    """Layer this size's tuned defaults over the shared ones, without ever
    overriding something the caller set on the command line."""
    tuned = (manifest.get("size_defaults") or {}).get(getattr(args, "size", "")) or {}
    for k, v in tuned.items():
        if not hasattr(args, k):
            continue
        flag = "--" + k
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            continue
        cur = getattr(args, k)
        try:
            setattr(args, k, type(cur)(v) if cur is not None and not isinstance(cur, bool) else v)
        except (TypeError, ValueError):
            setattr(args, k, v)


RESERVED_STRING_FLAGS = ("name", "corpus", "description", "resume")
# Reserved bool flags that the dashboard sends for every plugin regardless of
# whether the manifest opts in. Declared unconditionally so a user toggling
# them on the form is always honored, even on plugins whose manifest defaults
# don't list them. Manifest values override the defaults below.
RESERVED_BOOL_FLAGS = {
    "use_act_ckpt":  False,
    "use_8bit_adam": False,
    "qat_enabled":   False,
}

# Knobs the dashboard's Core Plugins section may inject regardless of whether
# the trainer's manifest opts in. Declared unconditionally so a user toggling
# a Core Plugin always has its args parsed cleanly. Manifest values override.
RESERVED_STR_FLAGS = {
    "activation": "gelu",
    "optimizer": "adamw",
    "trunk": "dense",
    "slm_ref": "",
    "state_rule": "gla",
    "state_carry": "off",
    "loss_mask": "off",
    # How much of the hook suite to write at each checkpoint.
    #   full  — every dump. ~137s per checkpoint on a 200m trunk.
    #   light — skip save.HEAVY_DUMPS. ~9s, because what remains reads weights
    #           and activations instead of generating text.
    #   off   — checkpoint only, no hooks directory.
    # Default is full: an existing run's behaviour must not change because this
    # flag appeared. Pair `light` with --hooks_full_every to keep dense cheap
    # signal and still get the deep probes periodically.
    "hooks": "full",
    # Pin validation to one corpus regardless of the training mix. Empty keeps
    # the default, which follows the heaviest corpus in the mix: a run whose val
    # bin moves when its mix moves cannot be compared to the run before it.
    "val_bin": "",
}
RESERVED_FLOAT_FLAGS = {
    "l1_lambda": 0.0,
    "slm_keep": 0.6,
    # Consolidation stop rule. A sleep run has no useful fixed length: it must run
    # until it starts costing more than it buys, then stop. Measured on wren1_8
    # (2026-08-25, lr 2e-4 over 48 study chunks): held-out val improved through step
    # 10 (0.9782) and fell off a cliff by step 20 (1.1122), so a fixed step count
    # either stops short of the gain or runs straight through the damage. Non-zero
    # arms the rule: when a val reading exceeds the run's BEST by more than this
    # fraction, the run checkpoints and stops. 0 disables it, so an ordinary
    # pretrain run is unaffected.
    "stop_on_val_rise": 0.0,
}
RESERVED_INT_FLAGS = {
    # With --hooks light, promote every Nth checkpoint back to the full suite,
    # so the expensive probes still trend over the run without being paid for at
    # every checkpoint. 0 disables the promotion. Ignored when --hooks is full
    # (already everything) or off (deliberately nothing).
    "hooks_full_every": 0,
}


# Dashboard TRAINER_SCHEMA fields that reach argv but that this trainer may
# legitimately not implement for a given configuration: the MoE router knobs and
# quantization mode are rendered for every plugin regardless of trunk, and
# `recipe` is uiOnly (it expands into other flags client-side). Everything not
# listed here and not an accepted flag is an error — see parse_args.
SCHEMA_IGNORED_FLAGS = frozenset({
    "recipe",
    "quant_mode",
    "n_experts", "router_topk", "router_aux_loss", "router_aux_loss_coef",
    "inject_layer", "mtp_aux_weight", "n_predict",
    "freeze_base",
    # Consumed by the runner as an environment variable (VERITATE_MODEL_TYPE) and
    # stamped into config.json by save(); it reaches argv too, and the trainer has
    # nothing to do with it there.
    "model_type",
    # Rendered for every plugin but only meaningful to trunks this file does not
    # build: `variant`/`alpha`/`rope_base` belong to the attention variants,
    # `init_from` to the plugins that seed weights from a foreign checkpoint.
    "variant", "alpha", "rope_base", "init_from",
})

# Shape fields the dashboard renders. This trainer does NOT take shape from
# flags: a fresh run gets it from --size, a resume from the checkpoint weights
# (see shape_for_run). Ignoring them silently is the exact failure the unknown-
# flag check exists to prevent, so they are parsed and then ASSERTED against the
# resolved shape. 0 means "not supplied".
SHAPE_OVERRIDE_FLAGS = ("layers", "hidden", "ffn", "heads")

# Trunks built by VeritatePatched. They wrap the `layers` global blocks in
# N_LOCAL_ENC + N_LOCAL_DEC local ones, so their checkpoints hold more block
# entries than the layer count that produced them. Keep in step with the
# model_cls dispatch in run().
PATCHED_TRUNKS = frozenset({
    "patched", "hybrid", "hybrid_moe", "hybrid_monarch",
    "hybrid_pkm", "hybrid_pkm_fire", "looped",
})


def parse_args(manifest):
    ap = argparse.ArgumentParser(description=manifest.get("description", ""))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench_budget_s", type=float, default=bench.BENCH_BUDGET_S_DEFAULT,
                    help="wall-clock ceiling for the batch ramp; a slow box needs one")
    for k in RESERVED_STRING_FLAGS:
        ap.add_argument("--" + k, type=str, default="")
    defaults = manifest.get("defaults", {}) or {}
    for k, default_val in RESERVED_BOOL_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(manifest_val))
    for k, default_val in RESERVED_STR_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, type=str, default=str(manifest_val))
    for k, default_val in RESERVED_FLOAT_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, type=float, default=float(manifest_val))
    for k, default_val in RESERVED_INT_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, type=int, default=int(manifest_val))
    for k in SHAPE_OVERRIDE_FLAGS:
        if k not in defaults:
            ap.add_argument("--" + k, type=int, default=0)
    for k, v in defaults.items():
        if k in RESERVED_STRING_FLAGS or k in RESERVED_BOOL_FLAGS:
            continue
        if k in RESERVED_STR_FLAGS or k in RESERVED_FLOAT_FLAGS:
            continue
        if k in RESERVED_INT_FLAGS:
            continue
        if isinstance(v, bool):
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int,   default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str,   default=str(v))
    # The dashboard renders the full TRAINER_SCHEMA for every plugin, so schema
    # flags this trainer does not implement (quant_mode on a non-MoE trunk, the
    # MoE router knobs) legitimately arrive on argv. Those are ignorable.
    #
    # Anything ELSE on argv is a mistake — a typo, or a flag carried over from a
    # trainer that no longer exists — and silently dropping it changes training
    # behaviour with no error and no log line. Refuse to start instead.
    args, unused = ap.parse_known_args()
    unknown = []
    for tok in unused:
        if not tok.startswith("--"):
            continue                      # a value belonging to a dropped flag
        flag = tok[2:].split("=", 1)[0]
        base = flag[3:] if flag.startswith("no-") else flag   # BooleanOptionalAction
        if base not in SCHEMA_IGNORED_FLAGS:
            unknown.append(tok.split("=", 1)[0])
    if unknown:
        raise SystemExit(
            "refusing to start: unrecognized flag(s) " + ", ".join(sorted(set(unknown))) +
            "\n\nThis trainer does not implement them, and they are not dashboard schema "
            "flags it is allowed to ignore. Silently dropping them would change how this "
            "run trains with no error and no log line.\n"
            "Fix the flag, or add it to SCHEMA_IGNORED_FLAGS if it is a dashboard-only "
            "field this trainer deliberately does not use.")
    return args


def trunk_block_overhead(trunk):
    """Blocks a trunk adds beyond its `layers` argument.

    VeritatePatched builds N_LOCAL_ENC local blocks, then `layers` global blocks,
    then N_LOCAL_DEC local ones, so --size 200m (layers=16) produces a 20-entry
    blocks list. Counting entries and calling that the layer count builds a model
    four blocks too deep; with load_state_dict(strict=False) that loads silently
    and leaves the extra blocks random. Read the constants off the module so the
    two cannot drift apart.
    """
    if trunk in PATCHED_TRUNKS and _model_patched_mod is not None:
        return (getattr(_model_patched_mod, "N_LOCAL_ENC", 2)
                + getattr(_model_patched_mod, "N_LOCAL_DEC", 2))
    return 0


def shape_from_checkpoint(name, trunk="dense"):
    """Read the architecture out of a run's newest checkpoint weights.

    The weights are the only account of a model's shape that cannot drift, but
    they record the blocks a trunk BUILT, not the `layers` argument it was built
    from. For the patched trunks those differ by trunk_block_overhead(); the
    returned `layers` is always the constructor's number, so callers can hand it
    straight back to the model class.
    """
    ckpt = os.path.join(paths.checkpoints_dir(name),
                        "step_" + str(latest_checkpoint_step(name)) + ".pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    blocks = 1 + max((int(m.group(1)) for m in
                      (re.match(r"blocks\.(\d+)\.", k) for k in sd) if m), default=-1)
    if not blocks:
        raise RuntimeError("cannot read layer count from checkpoint: " + ckpt)
    layers = blocks - trunk_block_overhead(trunk)
    if layers < 1:
        raise RuntimeError("checkpoint has " + str(blocks) + " blocks, too few for trunk "
                           + str(trunk) + ": " + ckpt)
    hidden = int(sd["tok_emb.weight"].shape[1])
    # The up-projection's out-features IS the FFN width. Name it across the
    # variants this repo has shipped rather than assuming one trunk's spelling.
    ffn = next((int(v.shape[0]) for k, v in sd.items()
                if re.match(r"blocks\.0\.(?:ff|mlp)\.(?:up|fc1|w1|up_proj)\.weight$", k)), None)
    # qkv packs q, k and v into one matrix, so heads is not recoverable from the
    # weights alone; the preset's head count stands unless the caller knows better.
    out = {"layers": layers, "hidden": hidden,
           "params": sum(v.numel() for v in sd.values())}
    if ffn is not None:
        out["ffn"] = ffn
    return out


def check_shape_overrides(args, shape, argv):
    """Refuse a launch whose explicit --layers/--hidden/--ffn/--heads disagree.

    This trainer resolves shape from --size or from the resumed weights, so a
    shape flag on argv cannot change what gets built. Honoring it is impossible
    and ignoring it would build a different model than the operator asked for,
    which is how wren_base ended up with a config.json claiming 16 layers over
    20 layers of weights. Say no instead.
    """
    for field in SHAPE_OVERRIDE_FLAGS:
        asked = int(getattr(args, field, 0) or 0)
        flag = "--" + field
        if not asked or not any(a == flag or a.startswith(flag + "=") for a in argv):
            continue
        got = shape.get(field)
        if got is not None and asked != got:
            raise SystemExit(
                "refusing to start: " + flag + " " + str(asked) + " but this run's "
                + field + " is " + str(got) + ". shape comes from --size (fresh) or "
                "from the resumed checkpoint's weights; it cannot be set on the "
                "command line. Drop the flag, or pick a --size whose " + field
                + " matches.")


def shape_for_run(args, size_presets, argv=None):
    """Resume takes its shape from the checkpoint; a fresh run from the preset.

    Any field the checkpoint disagrees with the preset on is a bug in the preset
    or in config.json, never a reason to build the preset's model. Report both
    numbers and use the weights.
    """
    shape = dict(size_presets[args.size])
    if not getattr(args, "resume", ""):
        check_shape_overrides(args, shape, argv if argv is not None else sys.argv)
        return shape
    actual = shape_from_checkpoint(args.resume, getattr(args, "trunk", "dense") or "dense")
    for field, got in actual.items():
        if got is None:
            continue
        want = shape.get(field)
        # params is a consequence of the other dims, so it disagrees whenever they
        # do and says nothing on its own. Announce only the structural fields.
        if want is not None and want != got and field != "params":
            print("shape: --size " + str(args.size) + " says " + field + "=" + str(want)
                  + " but " + args.resume + "'s weights have " + field + "=" + str(got)
                  + "; using the weights", flush=True)
        shape[field] = got
    check_shape_overrides(args, shape, argv if argv is not None else sys.argv)
    return shape


HOOK_MODES = ("full", "light", "off")


def hook_plan(args, step):
    """Which dumps to SKIP at this checkpoint, and a label for the log line.

    The full hook suite costs ~137s on a 200m trunk, nearly all of it in the five
    save.HEAVY_DUMPS. On a run with dense checkpoints that overhead is charged
    once per checkpoint and it adds up: at ckpt_every 250 over 8000 steps it is
    over an hour of wall clock. `light` keeps the cheap probes (~9s) so the run
    still trends, and --hooks_full_every promotes every Nth checkpoint back to
    the full suite so the expensive ones keep a coarse trend line too.
    """
    mode = (getattr(args, "hooks", "full") or "full").strip().lower()
    if mode not in HOOK_MODES:
        raise SystemExit("refusing to start: --hooks must be one of "
                         + ", ".join(HOOK_MODES) + " (got " + mode + ")")
    if mode == "off":
        return set(save.ALL_DUMPS), "off"
    if mode == "full":
        return set(), "full"
    every = int(getattr(args, "hooks_full_every", 0) or 0)
    # Count promotions in checkpoints, not steps: `every 4` means every 4th
    # checkpoint regardless of how ckpt_every is set.
    if every > 0 and (step // max(1, args.ckpt_every)) % every == 0:
        return set(), "full"
    return set(save.HEAVY_DUMPS), "light"


def latest_checkpoint_step(name):
    ckpt_dir = paths.checkpoints_dir(name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("no checkpoints dir for: " + name)
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(BASE_CKPT_PREFIX) and fn.endswith(BASE_CKPT_SUFFIX):
            try:
                steps.append(int(fn[len(BASE_CKPT_PREFIX):-len(BASE_CKPT_SUFFIX)]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError("no step_*.pt under: " + ckpt_dir)
    return max(steps)


def apply_resume_overrides(args, argv):
    cfg_path = paths.config_path(args.resume)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError("no config.json for resume target: " + args.resume)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    ta = cfg.get("training_args") or {}
    for k, v in ta.items():
        if not hasattr(args, k):
            continue
        flag = "--" + k
        user_set = any(a == flag or a.startswith(flag + "=") for a in argv)
        if user_set:
            continue
        cur = getattr(args, k)
        if isinstance(cur, bool) and not isinstance(v, bool):
            continue
        try:
            setattr(args, k, type(cur)(v) if cur is not None else v)
        except (TypeError, ValueError):
            setattr(args, k, v)


def lr_at(step, total, warmup, base_lr, min_lr, schedule="cosine",
          wsd_decay_frac=0.1, wsd_decay_kind="sqrt"):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr + (min_lr - base_lr) * p
    if schedule == "wsd":
        decay_frac = max(1e-6, min(1.0, float(wsd_decay_frac)))
        stable_p = 1.0 - decay_frac
        if p <= stable_p:
            return base_lr
        q = (p - stable_p) / decay_frac
        q = min(max(q, 0.0), 1.0)
        if wsd_decay_kind == "linear":
            shape = 1.0 - q
        elif wsd_decay_kind == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * q))
        else:
            shape = 1.0 - math.sqrt(q)
        return min_lr + (base_lr - min_lr) * shape
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def make_data_loader(bin_path, total_chunk_len, batch_size, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    N = len(arr)
    rng = np.random.RandomState(seed)
    if total_chunk_len + 2 > N:
        raise ValueError("corpus too small for chunk length: " + str(N) + " < " + str(total_chunk_len + 2))

    def draw():
        starts = rng.randint(0, N - total_chunk_len - 1, size=batch_size, dtype=np.int64)
        toks = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        tgts = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        for b, s in enumerate(starts):
            toks[b] = arr[s:s + total_chunk_len]
            tgts[b] = arr[s + 1:s + 1 + total_chunk_len]
        return torch.from_numpy(toks), torch.from_numpy(tgts)

    return draw, N


# A corpus this dense in ChatML turn markers is a chat/SFT set, not free prose.
# Training one WITHOUT loss_mask=assistant computes loss over the user's turns too,
# so the model learns to continue the user rather than answer them. That silently
# produces a fluent model that ignores the question, so the trainer refuses to
# start until the caller decides explicitly.
TRAINING_KIND_CHAT      = "chat"   # stamped on config.json so the serve path knows
                                   # this model expects ChatML framing, instead of
                                   # guessing and mismatching what it was trained on
CHATML_DENSITY_SFT      = 0.20     # marker-bearing sample windows / windows read
CHATML_PROBE_WINDOWS    = 64
CHATML_PROBE_WINDOW_LEN = 8192

CHATML_ASSISTANT_OPEN = b"<|im_start|>assistant\n"
CHATML_IM_START       = b"<|im_start|>"
CHATML_IM_END         = b"<|im_end|>"


def val_rose_past_best(v, best_val, tolerance):
    """Consolidation stop rule: True when this val reading exceeds the run's BEST by
    more than `tolerance` (fractional). A consolidation run has no useful fixed
    length -- it must run while it helps and stop when it starts costing more than it
    buys. Zero tolerance disables the rule, leaving ordinary runs unaffected."""
    tolerance = float(tolerance or 0.0)
    if tolerance <= 0.0 or best_val in (None, float("inf")):
        return False
    return v > best_val * (1.0 + tolerance)


def _keep_assistant(row_bytes):
    """Mask of the bytes loss is computed on: assistant CONTENT plus its closing
    <|im_end|>, never the turn openings or the user's text. Scans marker POSITIONS
    via bytes.find rather than walking one byte at a time: the byte walk cost 43% of
    training throughput at batch 16 (10.4k tok/s against the bench's 18.1k), because
    it ran seq * n_chunks * batch Python iterations per step."""
    n = len(row_bytes)
    keep = np.zeros(n, dtype=bool)
    ao, ist, ie = CHATML_ASSISTANT_OPEN, CHATML_IM_START, CHATML_IM_END
    la, le = len(ao), len(ie)
    pos = row_bytes.find(ao)
    while pos != -1:
        start = pos + la
        end   = row_bytes.find(ie, start)
        nxt_t = row_bytes.find(ist, start)
        if end != -1 and (nxt_t == -1 or end < nxt_t):
            keep[start:end + le] = True
            resume = end + le
        elif nxt_t != -1:
            keep[start:nxt_t] = True
            resume = nxt_t
        else:
            keep[start:n] = True
            resume = n
        pos = row_bytes.find(ao, resume)
    return keep


def apply_role_mask(toks, tgts):
    tk = toks.cpu().numpy().astype(np.uint8)
    tg = tgts.clone()
    B, N = tk.shape
    for b in range(B):
        rb = tk[b].tobytes()
        if CHATML_IM_START not in rb:
            continue
        keep = _keep_assistant(rb)
        active = np.zeros(N, dtype=bool)
        active[:-1] = keep[1:]
        tg[b][~torch.from_numpy(active)] = -1
    return tg


def carry_seam_clip(clip):
    """Gradient hook for the state tensors carried across chunk boundaries.
    Backprop through the carry seam explodes on some rows (truncated-BPTT
    gradient explosion, data-dependent; wren1_3 2026-08-18): norm-clip the
    seam gradient to `clip`, and zero it if non-finite, so the retention
    learning signal keeps its direction and the step stays finite."""
    def hook(g):
        n = g.norm()
        if not torch.isfinite(n):
            return torch.zeros_like(g)
        if n > clip:
            return g * (clip / (n + 1e-6))
        return g
    return hook


def chunked_step(model, tokens, targets, seq, amp_dtype, *, backward=False, bptt_window=1,
                 device_type="cuda", l1_lambda=0.0, slm_ref=None, slm_keep=0.6,
                 state_carry="off", carry_grad_clip=0.0):
    total_len = tokens.shape[1]
    n_chunks = max(1, total_len // seq)
    K = max(1, int(bptt_window))
    loss_sum = 0.0
    n_valid  = 0
    window_losses = []
    capture_l1 = bool(getattr(model, "capture_l1", False)) and l1_lambda > 0.0
    # Memory trunks carry fast-weight state across the contiguous chunks of one
    # step (reset per step): the loss then rewards reading memory beyond the
    # attention window. Non-memory models lack forward_carry and take the
    # plain path.
    mem_carry = hasattr(model, "forward_carry")
    if mem_carry:
        model.reset_memory()
    # state_carry=chunks: recurrent global state threads left-to-right through the
    # step's contiguous chunks via forward_streaming (positions stay window-local).
    # Grads flow through the carry inside a bptt window; carry detached at each
    # backward. State resets per step (rows are unrelated stream offsets).
    rec_carry  = state_carry == STATE_CARRY_CHUNKS
    rec_states = None
    for cstart in range(0, total_len, seq):
        cend = min(cstart + seq, total_len)
        ct = tokens[:, cstart:cend]
        cg = targets[:, cstart:cend]
        if ct.size(1) < 2:
            break
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            if rec_carry:
                logits, rec_states = model.forward_streaming(ct, rec_states)
                if backward and carry_grad_clip > 0.0:
                    for st in rec_states:
                        for v in st.values():
                            if torch.is_tensor(v) and v.requires_grad:
                                v.register_hook(carry_seam_clip(carry_grad_clip))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), cg.reshape(-1), ignore_index=-1)
            else:
                logits, loss = model.forward_carry(ct, targets=cg) if mem_carry else model(ct, targets=cg)
            if slm_ref is not None:
                loss = slm_helpers.selective_loss(slm_ref, ct, cg, logits, keep_frac=slm_keep)
        if loss is None or not torch.isfinite(loss):
            continue
        if capture_l1:
            l1 = model.post_l1_sum()
            if l1 is not None:
                loss = loss + l1_lambda * l1
        if backward:
            aux = model.moe_aux_sum() if hasattr(model, "moe_aux_sum") else None
            if aux is not None:
                loss = loss + aux
        loss_sum += float(loss.detach().item())
        n_valid  += 1
        if backward:
            window_losses.append(loss)
            window_full = len(window_losses) >= K
            last_chunk  = (cstart + seq) >= total_len
            if window_full or last_chunk:
                (torch.stack(window_losses).sum() / n_chunks).backward()
                window_losses = []
                if mem_carry:
                    model.carry_memory()
                if rec_states is not None:
                    rec_states = [{k: v.detach() for k, v in st.items()} for st in rec_states]
    if n_valid == 0:
        return None
    return loss_sum / n_valid


def write_config(name, args, base_cfg, n_params, corpus_hash, plugin_id):
    cfg_path = paths.config_path(name)
    os.makedirs(paths.model_dir(name), exist_ok=True)
    ta = vars(args).copy()
    if corpus_hash:
        ta["corpus_sha256"] = corpus_hash.get("train_sha256")
        ta["corpus_bytes"]  = corpus_hash.get("train_bytes")
    shape = dict(base_cfg)
    shape["seq"]   = args.seq
    shape["vocab"] = VOCAB_BYTE_LEVEL
    qat_on = bool(getattr(args, "qat_enabled", False))
    cfg = {
        "name": name,
        "description": args.description,
        "kind": "trainer",
        "plugin": plugin_id,
        "vocab": VOCAB_BYTE_LEVEL,
        "shape": shape,
        "training":  ("qat" if qat_on else (args.training_kind or "")),
        "qat_source": (args.resume if (qat_on and args.resume) else ""),
        "training_args": ta,
        "n_params_total": n_params,
        "wrote_at": int(time.time()),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_resume_state(model, name, step, device, require_complete=False):
    """Load a checkpoint into `model`, reporting anything that did not land.

    strict=False is load-bearing here: QAT seeds a new model from a plain one and
    legitimately has observer/scale tensors the source lacks. The cost is that a
    genuinely wrong shape also loads without a word. wren_base was resumed into a
    trunk built four blocks too deep and 54,649,152 parameters stayed at their
    random init, with nothing in the log but a params count nobody was diffing.
    So: always say what was skipped, and on a plain resume refuse to continue.
    """
    ckpt = torch.load(paths.checkpoint_path(name, step), map_location=device, weights_only=False)
    sd = ckpt["model"]
    if any(k.startswith("base.") for k in sd):
        sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
    result = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        live = dict(model.state_dict())
        stranded = sum(int(live[k].numel()) for k in missing if k in live)
        print("load: " + str(len(missing)) + " tensor(s) not in the checkpoint ("
              + str(stranded) + " params left at init), "
              + str(len(unexpected)) + " tensor(s) in the checkpoint with no home",
              flush=True)
        for k in (missing[:6] + unexpected[:6]):
            print("      " + k, flush=True)
        if require_complete and missing:
            raise SystemExit(
                "refusing to continue: resuming " + name + " left " + str(stranded)
                + " parameters at their random init. The model being built does not "
                "match the checkpoint. Check --trunk and --size against the config "
                "this checkpoint was written with.")
    return ckpt.get("optimizer")


@torch.no_grad()
def evaluate(model, val_draw, n_iters, seq, amp_dtype, bptt_window, device_type="cuda",
             state_carry="off"):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(next(model.parameters()).device, non_blocking=True)
        tgts = tgts.to(next(model.parameters()).device, non_blocking=True)
        loss = chunked_step(model, toks, tgts, seq, amp_dtype,
                            bptt_window=bptt_window, device_type=device_type,
                            state_carry=state_carry)
        if loss is not None:
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else None


def build_optimizer(params, args, device, plan=None, state_dir=None, model=None):
    want_muon = getattr(args, "optimizer", "adamw") == "muon"
    if want_muon and (optim_helpers is None or model is None):
        print("optimizer=muon requested but unavailable on this platform; using AdamW", flush=True)
        want_muon = False
    if want_muon:
        print("optimizer: Muon (2D hidden weights) + AdamW (emb/norms/1D), "
              "orthogonalizing in "
              + str(optim_helpers.ns_dtype(device)).split(".")[-1], flush=True)
        return optim_helpers.build_muon(model, args, device)
    if _MEM_PAGING and plan is not None and plan.tier in mem_executor.OFFLOAD_TIERS:
        print("optimizer state paged to NVMe (" + plan.tier + "); resident optimizer "
              "memory ~0, step time is disk-bound", flush=True)
        return mem_executor.make_optimizer(
            params, plan,
            lr=args.base_lr, betas=(args.beta1, args.beta2), eps=1e-6,
            weight_decay=args.weight_decay, state_dir=state_dir)
    use_8bit = bool(getattr(args, "use_8bit_adam", False))
    if use_8bit and importlib.util.find_spec("bitsandbytes") is None:
        print("8-bit AdamW requested but bitsandbytes is not installed; "
              "falling back to torch AdamW (fp32 moments, ~2x optimizer memory)", flush=True)
        use_8bit = False
    if use_8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            params,
            lr=args.base_lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), eps=1e-6,
        )
    return torch.optim.AdamW(
        params,
        lr=args.base_lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=1e-6,
        fused=(device == "cuda"),
    )


def chatml_density(train_path):
    """Fraction of sampled windows that contain a ChatML turn marker. Sampled
    rather than scanned: a pretrain corpus can be gigabytes."""
    try:
        size = os.path.getsize(train_path)
    except OSError:
        return 0.0
    if size <= 0:
        return 0.0
    span = min(CHATML_PROBE_WINDOW_LEN, size)
    hits = 0
    with open(train_path, "rb") as f:
        for i in range(CHATML_PROBE_WINDOWS):
            off = (size - span) * i // max(1, CHATML_PROBE_WINDOWS - 1)
            f.seek(off)
            if CHATML_IM_START in f.read(span):
                hits += 1
    return hits / CHATML_PROBE_WINDOWS


def resolve_val_path(corpus_mix, override=""):
    """Return (val_path, warning). resolve_and_weight sorts weight-descending, so
    validation follows the HEAVIEST corpus in the mix, not the first one written.
    A mix whose top corpus ships no _val.bin trains blind for its whole duration,
    which is silent unless it is said out loud.

    `override` pins validation to one corpus regardless of the mix. A run whose
    val bin moves when its mix moves cannot be compared to the run before it,
    which is what sleep consolidation needs to detect a model degrading over
    successive runs on its own output."""
    if override:
        from readers import corpus as corpus_reader
        pinned = corpus_reader.resolve_paths(override)[1]
        if pinned and os.path.isfile(pinned):
            return pinned, ""
        return None, (f"WARNING: val_bin '{override}' resolves to no _val.bin on this box, so"
                      " this run will report NO validation loss. Build it or clear the override.")
    val_path = corpus_mix[0][1] if corpus_mix else None
    if val_path and os.path.isfile(val_path):
        return val_path, ""
    return None, ("WARNING: the heaviest corpus in this mix ships no _val.bin, so this"
                  " run will report NO validation loss for its entire duration. Build a"
                  " _val.bin for it, or give the top weight to a corpus that has one.")


def require_loss_mask_decision(args, corpus_mix, argv):
    """Refuse to train a ChatML-dense corpus until loss_mask is set explicitly.

    Forgetting it is silent: the run looks healthy, the loss falls, and the model
    comes out fluent but unable to answer a question. A warning is too easy to
    miss in a long log, so this is an error the caller has to answer."""
    if any(a == "--loss_mask" or a.startswith("--loss_mask=") for a in argv):
        return
    if getattr(args, "loss_mask", "off") != "off":
        return
    for _stem, _val, train_path, _w in _iter_mix_paths(corpus_mix):
        density = chatml_density(train_path)
        if density >= CHATML_DENSITY_SFT:
            raise SystemExit(
                "refusing to start: corpus '" + str(args.corpus) + "' is "
                + str(int(density * 100)) + "% ChatML turn markers, which makes it a "
                "chat/SFT corpus, but --loss_mask was never set.\n"
                "  Without it, loss is computed over the USER's turns too and the model "
                "learns to continue the user instead of answering.\n"
                "  Pass --loss_mask assistant   (what you almost certainly want for SFT)\n"
                "  or   --loss_mask off         (to train on every byte on purpose)")


def _iter_mix_paths(corpus_mix):
    """Yield (stem, val_path, train_path, weight) for a resolved multicorpus mix,
    tolerating the tuple shapes the resolver has used across versions."""
    for entry in corpus_mix or []:
        if not isinstance(entry, (tuple, list)):
            continue
        paths_in = [x for x in entry if isinstance(x, str) and x.endswith(".bin")]
        train_path = None
        for cand in paths_in:
            if cand.endswith("_train.bin"):
                train_path = cand
                break
        if train_path is None and paths_in:
            train_path = paths_in[-1]
        if train_path:
            yield (None, None, train_path, None)


def detect_training_kind(corpus_mix):
    """What framing did this model learn? Stamped on config.json so serving reads
    it instead of assuming. A model trained on plain prose and then served ChatML
    produces confident nonsense, which is exactly how chat_200m broke."""
    for _stem, _val, train_path, _w in _iter_mix_paths(corpus_mix):
        if chatml_density(train_path) >= CHATML_DENSITY_SFT:
            return TRAINING_KIND_CHAT
    return ""


def run(plugin_id, here):
    manifest = _load_manifest(here)
    size_presets = _size_presets(manifest)
    if not size_presets:
        raise ValueError("manifest.sizes missing or empty for plugin: " + plugin_id)

    args = parse_args(manifest)
    apply_size_defaults(args, manifest, sys.argv)
    resume_mode = bool(args.resume)
    qat_enabled = bool(getattr(args, "qat_enabled", False))
    qat_source  = args.resume if (resume_mode and qat_enabled) else None
    if resume_mode:
        apply_resume_overrides(args, sys.argv)
        if qat_enabled:
            args.qat_enabled = True
    if not args.bench:
        require_description(args.description)

    if args.size not in size_presets:
        raise ValueError("unknown size: " + str(args.size) + " (valid: " + ", ".join(size_presets) + ")")
    if args.precision not in PRECISIONS:
        raise ValueError("unknown precision: " + str(args.precision))
    if args.lr_schedule not in LR_SCHEDULES:
        raise ValueError("unknown lr_schedule: " + str(args.lr_schedule))
    if args.lr_schedule == "wsd":
        kind = getattr(args, "wsd_decay_kind", "sqrt")
        if kind not in WSD_DECAY_KINDS:
            raise ValueError("unknown wsd_decay_kind: " + str(kind)
                             + " (valid: " + ", ".join(WSD_DECAY_KINDS) + ")")
        frac = float(getattr(args, "wsd_decay_frac", 0.1))
        if not (0.0 < frac <= 1.0):
            raise ValueError("wsd_decay_frac must be in (0, 1], got " + str(frac))

    if qat_source is not None:
        name = qat_source + "_qat"
        print("QAT continue: source=" + qat_source + " new model=" + name, flush=True)
    elif resume_mode:
        name = args.resume
    elif getattr(args, "name", "").strip():
        # Dashboard scratch path: user-supplied slug, size suffix appended by
        # save.compose_name (matches the native trainer convention).
        slug = compose_name(args.name, args.size)
        name = slug + "_qat" if qat_enabled else slug
    else:
        v = args.version
        if v.endswith("_qat"):
            v = v[:-4]
        elif v.endswith("qat"):
            v = v[:-3]
        version_tag = (v + "_qat") if qat_enabled else v
        name = compose_name(args.corpus, args.size, args.precision, version_tag)
    print("model name: " + name, flush=True)

    _corpus_mix = None
    val_path    = None
    args.training_kind = ""
    if not args.bench:
        _corpus_mix = multicorpus.resolve_and_weight(args.corpus, resolve_corpus)
        print("corpus mix:   " + multicorpus.format_mix_summary(_corpus_mix), flush=True)
        val_path, val_warning = resolve_val_path(_corpus_mix, getattr(args, "val_bin", "") or "")
        print(("corpus val:   " + val_path) if val_path else val_warning, flush=True)
        require_loss_mask_decision(args, _corpus_mix, sys.argv)
        args.training_kind = detect_training_kind(_corpus_mix)
        if args.training_kind:
            print("training kind: " + args.training_kind
                  + " (config.json records the framing this model expects)", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = hardware.pick_device()
    print("device: " + device, flush=True)
    amp_dtype = hardware.resolve_precision(args.precision, device)

    shape = shape_for_run(args, size_presets)

    # Plan the memory ladder from the size preset BEFORE building the model: a model
    # whose weights+grads exceed the budget cannot be built (the allocation OOMs/
    # SIGKILLs), so refuse with the floor numbers instead of hanging. bench plans at
    # batch 1 (the ramp explores batch); a real run plans at its configured batch.
    mem_plan = None
    if _MEM_PAGING and shape.get("params"):
        plan_batch = 1 if args.bench else args.batch_size
        mem_plan = mem_planner.plan_training_memory(
            shape["params"], shape["hidden"], shape["layers"], shape["ffn"],
            plan_batch, args.seq, PLAN_DTYPE)
        print("mem plan: " + mem_planner.format_plan(mem_plan), flush=True)
        if not mem_plan.fits:
            if args.bench:
                print("BENCH_RESULT " + json.dumps(bench.plan_result(mem_plan, device, args.seq)), flush=True)
            elif mem_plan.params_bytes + mem_plan.grads_bytes > mem_plan.budget_bytes:
                print("INFEASIBLE: weights+grads alone exceed the memory budget; paging "
                      "the optimizer cannot help. Use a smaller size or a larger-memory "
                      "machine.", flush=True)
            else:
                print("INFEASIBLE at this batch/seq: activations push the run over budget "
                      "even with checkpointing. Reduce batch_size or seq.", flush=True)
            return

    activation = getattr(args, "activation", "gelu") or "gelu"
    l1_lambda  = float(getattr(args, "l1_lambda", 0.0) or 0.0)
    trunk      = getattr(args, "trunk", "dense") or "dense"
    model_cls  = Veritate
    model_kwargs = {}
    if trunk == "patched":
        if VeritatePatched is None:
            raise ValueError("trunk=patched requested but platform lacks model_patched")
        model_cls = VeritatePatched
    elif trunk == "hybrid":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent"}
    elif trunk == "hybrid_moe":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid_moe requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent", "global_ffn": "moe"}
    elif trunk == "hybrid_monarch":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid_monarch requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent", "global_ffn": "monarch"}
    elif trunk == "hybrid_pkm":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid_pkm requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent", "global_ffn": "pkm"}
    elif trunk == "hybrid_pkm_fire":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid_pkm_fire requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent", "global_ffn": "pkm_fire"}
    elif trunk == "looped":
        if VeritatePatched is None:
            raise ValueError("trunk=looped requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_loops": 4}
    elif trunk == "recurrent":
        if VeritateRecurrent is None:
            raise ValueError("trunk=recurrent requested but platform lacks model_recurrent")
        model_cls = VeritateRecurrent
    elif trunk == "memory":
        if VeritateMemory is None:
            raise ValueError("trunk=memory requested but platform lacks model_memory")
        model_cls = VeritateMemory
    state_rule = getattr(args, "state_rule", "gla") or "gla"
    if trunk in STATE_CARRY_TRUNKS and state_rule != "gla":
        model_kwargs["state_rule"] = state_rule
    state_carry = getattr(args, "state_carry", "off") or "off"
    if state_carry not in STATE_CARRIES:
        raise ValueError("unknown state_carry: " + str(state_carry)
                         + " (valid: " + ", ".join(STATE_CARRIES) + ")")
    if state_carry == STATE_CARRY_CHUNKS and trunk not in STATE_CARRY_TRUNKS:
        raise ValueError("state_carry=chunks requires trunk in: " + ", ".join(STATE_CARRY_TRUNKS))
    veritate_model = model_cls(
        vocab=VOCAB_BYTE_LEVEL,
        hidden=shape["hidden"], layers=shape["layers"],
        ffn=shape["ffn"], heads=shape["heads"], seq=args.seq,
        activation=activation,
        capture_l1=(l1_lambda > 0.0),
        **model_kwargs,
    )
    print(f"activation: {activation}  l1_lambda: {l1_lambda}  trunk: {trunk}"
          f"  state_carry: {state_carry}", flush=True)
    if qat_enabled:
        qat_helpers.set_qat(veritate_model, True)
        print("QAT: enabled (fake-quant matmuls + embeddings + RMSNorm + residual adds)", flush=True)

    if _MEM_PAGING and mem_plan is not None and mem_plan.tier in mem_executor.CHECKPOINT_TIERS:
        mem_executor.enable_grad_checkpoint(veritate_model)
        print("activation checkpointing: ENABLED (mem plan tier " + mem_plan.tier + ")", flush=True)
    elif getattr(args, "use_act_ckpt", False):
        print("activation checkpointing: ENABLED", flush=True)
        for blk in veritate_model.blocks:
            # Kwargs pass through so state_carry=chunks (state=, return_state=)
            # composes with activation checkpointing.
            blk.forward = (lambda fwd: lambda x, **kw: torch.utils.checkpoint.checkpoint(
                fwd, x, use_reentrant=False, **kw))(blk.forward)

    veritate_model.to(device)
    slm_ref = None
    _slm_name = (getattr(args, "slm_ref", "") or "").strip()
    if _slm_name:
        if slm_helpers is None:
            raise ValueError("slm_ref requested but platform lacks slm helper")
        slm_ref = slm_helpers.load_reference(paths.model_dir(_slm_name), device)
        print(f"SLM: selective loss ON, reference={_slm_name}, keep_frac={getattr(args, 'slm_keep', 0.6)}", flush=True)
    n_params = sum(p.numel() for p in veritate_model.parameters())
    # print what the run actually computes in, not what was asked for: a device
    # without bf16 acceleration silently trains fp32, and the requested string
    # sent an investigation after the wrong dtype (2026-08-24)
    print("device: " + device + "  precision: "
          + (args.precision if amp_dtype is not None else "fp32 (autocast off)"), flush=True)
    print("params: " + str(n_params), flush=True)
    print("shape:  hidden=" + str(shape["hidden"]) + " layers=" + str(shape["layers"])
          + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + " seq=" + str(args.seq), flush=True)

    if args.bench:
        if _MEM_PAGING:
            result = bench.run(veritate_model, device, args.seq, VOCAB_BYTE_LEVEL,
                               on_progress=lambda s: print("bench: " + s, flush=True),
                               plan=mem_plan, amp_dtype=amp_dtype,
                               n_chunks=args.n_chunks, bptt_window=args.bptt_window,
                               optimizer=args.optimizer, budget_s=args.bench_budget_s)
        else:
            result = bench.run(veritate_model, device, args.seq, VOCAB_BYTE_LEVEL,
                               on_progress=lambda s: print("bench: " + s, flush=True),
                               amp_dtype=amp_dtype,
                               n_chunks=args.n_chunks, bptt_window=args.bptt_window,
                               optimizer=args.optimizer, budget_s=args.bench_budget_s)
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        return

    resume_step = 0
    resume_opt_state = None
    if qat_source is not None:
        src_step = latest_checkpoint_step(qat_source)
        print("QAT load: " + qat_source + " step " + str(src_step) + "  -> new model " + name, flush=True)
        load_resume_state(veritate_model, qat_source, src_step, device)
        write_config(name, args, shape, n_params, corpus_hash=None, plugin_id=plugin_id)
        print("wrote: " + paths.config_path(name), flush=True)
    elif resume_mode:
        resume_step = latest_checkpoint_step(name)
        print("resume: " + name + "  from step " + str(resume_step), flush=True)
        resume_opt_state = load_resume_state(veritate_model, name, resume_step, device,
                                             require_complete=True)
        dropped = truncate_train_csv_at(name, resume_step)
        if dropped:
            print("train.csv: dropped " + str(dropped) + " stale rows past step " + str(resume_step), flush=True)
    else:
        print("hashing corpus (one-time, ~5-10s for 2GB)...", flush=True)
        corpus_hash = hash_corpus(args.corpus)
        print("corpus sha256: " + corpus_hash.get("train_sha256", "?")[:16]
              + "...  bytes=" + str(corpus_hash.get("train_bytes")), flush=True)
        write_config(name, args, shape, n_params, corpus_hash, plugin_id=plugin_id)
        print("wrote: " + paths.config_path(name), flush=True)

    total_chunk_len = args.seq * args.n_chunks
    train_draw, train_n = multicorpus.make_mixed_loader(_corpus_mix, args.batch_size, total_chunk_len, args.seed)
    val_draw = None
    if val_path:
        val_draw, _ = make_data_loader(val_path, total_chunk_len, args.batch_size, args.seed + 1)
    print("train corpus bytes: " + str(train_n) + "  per-step chunk: " + str(total_chunk_len)
          + "  batch: " + str(args.batch_size), flush=True)

    if _MEM_PAGING:
        opt = build_optimizer(veritate_model.parameters(), args, device, plan=mem_plan,
                              state_dir=os.path.join(paths.model_dir(name), PAGED_STATE_DIR),
                              model=veritate_model)
    else:
        opt = build_optimizer(veritate_model.parameters(), args, device, model=veritate_model)
    if resume_opt_state is not None:
        try:
            opt.load_state_dict(resume_opt_state)
            print("optimizer state restored", flush=True)
        except Exception as e:
            print("optimizer state restore skipped: " + str(e), flush=True)

    grow_to_ffn = int(getattr(args, "grow_to_ffn", 0) or 0)
    val_hist    = []
    best_val, best_val_step, stop_now = float("inf"), 0, False
    has_grown   = False
    t0 = time.time()
    last_log = t0
    last_log_step = resume_step
    start_step = resume_step + 1
    skipped_nonfinite = 0
    for step in range(start_step, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                   schedule=args.lr_schedule,
                   wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.1),
                   wsd_decay_kind=getattr(args, "wsd_decay_kind", "sqrt"))
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        if getattr(args, "loss_mask", "off") == "assistant":
            tgts = apply_role_mask(toks, tgts)
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        veritate_model.train()
        opt.zero_grad(set_to_none=True)
        loss = chunked_step(veritate_model, toks, tgts, args.seq, amp_dtype,
                            backward=True, bptt_window=args.bptt_window,
                            device_type=device, l1_lambda=l1_lambda,
                            slm_ref=slm_ref, slm_keep=float(getattr(args, "slm_keep", 0.6)),
                            state_carry=state_carry, carry_grad_clip=args.grad_clip)
        if loss is None:
            skipped_nonfinite += 1
            if skipped_nonfinite % args.log_every == 0 or skipped_nonfinite == 1:
                print(f"WARNING step {step}: non-finite loss, step skipped "
                      f"({skipped_nonfinite} skipped so far)", flush=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(veritate_model.parameters(), args.grad_clip)
        if not torch.isfinite(gn):
            # A finite loss does not guarantee finite gradients; stepping on a
            # non-finite gradient writes nan into the weights and every later
            # step silently fails (wren1_3, 2026-08-18). Skip and keep training.
            opt.zero_grad(set_to_none=True)
            skipped_nonfinite += 1
            if skipped_nonfinite % args.log_every == 0 or skipped_nonfinite == 1:
                print(f"WARNING step {step}: non-finite grad norm, step skipped "
                      f"({skipped_nonfinite} skipped so far)", flush=True)
            continue
        opt.step()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            elapsed = now - t0
            window_s = max(1e-6, now - last_log)
            window_steps = step - last_log_step
            tok_per_s = window_steps * args.batch_size * total_chunk_len / window_s
            print("step " + str(step) + "  loss " + format(loss, ".4f") + "  lr " + format(lr, ".2e")
                  + "  gn " + format(float(gn), ".3f") + "  tok/s " + format(tok_per_s, ".0f")
                  + "  elapsed " + format(elapsed, ".0f") + "s", flush=True)
            append_train_row(name, step, "train", float(loss),
                             lr=lr, grad_norm=float(gn),
                             tok_per_s=tok_per_s, wall_s=elapsed, seed=args.seed)
            last_log = now
            last_log_step = step

        if val_draw is not None and step % args.eval_every == 0:
            v = evaluate(veritate_model, val_draw, args.eval_iters, args.seq, amp_dtype,
                         args.bptt_window, device_type=device, state_carry=state_carry)
            if v is not None:
                print("step " + str(step) + "  val_loss " + format(v, ".4f"), flush=True)
                append_train_row(name, step, "val", v, lr=lr,
                                 wall_s=time.time() - t0, seed=args.seed)
                val_hist.append(v)
                if v < best_val:
                    best_val, best_val_step = v, step
                stop_rise = float(getattr(args, "stop_on_val_rise", 0.0) or 0.0)
                if val_rose_past_best(v, best_val, stop_rise):
                    print("STOP step " + str(step) + "  val " + format(v, ".4f")
                          + " rose past best " + format(best_val, ".4f")
                          + " (step " + str(best_val_step) + ") by more than "
                          + format(stop_rise * 100, ".1f") + "%; best weights are "
                          + "step_" + str(best_val_step) + ".pt", flush=True)
                    stop_now = True
                # Growth only nets compute if the cheap stage stops the moment its
                # curve flattens (successes.md 2026-07-25); a fixed step count
                # overspends the small shape and turns a 3.6x step win into a loss.
                if (grow_to_ffn and not has_grown
                        and grow_to_ffn > veritate_model.ffn
                        and grow_mod.val_has_flattened(val_hist)):
                    widened = grow_mod.widen_model(veritate_model, grow_to_ffn)
                    opt = build_optimizer(veritate_model.parameters(), args, device,
                                          model=veritate_model)
                    has_grown = True
                    args.ffn = grow_to_ffn
                    print("GROW step " + str(step) + "  ffn -> " + str(grow_to_ffn)
                          + "  layers " + str(len(widened))
                          + "  params " + str(sum(p.numel() for p in veritate_model.parameters()))
                          + "  (val flattened)", flush=True)

        if step % args.ckpt_every == 0 or step == args.total_steps or stop_now:
            ckpt_args = vars(args).copy()
            ckpt_args["vocab"]  = veritate_model.vocab
            ckpt_args["hidden"] = veritate_model.hidden
            ckpt_args["layers"] = veritate_model.layers
            ckpt_args["ffn"]    = veritate_model.ffn
            ckpt_args["heads"]  = veritate_model.heads
            ckpt_args["seq"]    = veritate_model.seq
            ckpt_args.setdefault("description", args.description)
            skip_dumps, hook_label = hook_plan(args, step)
            ckpt_path = save.save(veritate_model, name, step, optimizer=opt, args=ckpt_args,
                                  dump_set=skip_dumps)
            print("checkpoint + hooks(" + hook_label + "): " + ckpt_path, flush=True)

        if stop_now:
            break

    print("done.", flush=True)


# ------------------------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    run(plugin_id="veritate", here=os.path.dirname(os.path.abspath(__file__)))
