# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Size-adaptive training-memory planner. Given a model's parameter count and
#   shape plus the run's batch/seq/dtype, it sums the four memory buckets
#   (params, grads, optimizer state, activations), compares against this host's
#   unified-memory budget from hardware.unified_memory_bytes(), and returns the
#   minimal escalation tier that fits. Pure arithmetic: no torch, no device, no
#   allocation, so it runs at startup before the model is on device.
# - Apple Silicon has no separate VRAM; the GPU shares the unified RAM pool, so
#   "offload to host" frees nothing. The escalation ladder is therefore
#   checkpoint activations -> bf16 optimizer state -> page optimizer to NVMe ->
#   infeasible. Each rung cuts a real bucket; none pretends a second pool exists.
# veritate_core/plugin/mem_planner.py
# ------------------------------------------------------------------------------------
# Imports

from dataclasses import dataclass

from veritate_core.plugin import hardware

# ------------------------------------------------------------------------------------
# Constants

BYTES_FP32 = 4
BYTES_BF16 = 2

DTYPE_BYTES = {"fp32": BYTES_FP32, "bf16": BYTES_BF16, "fp16": BYTES_BF16}

# Adam keeps two fp32 moment buffers per parameter; mixed precision adds one
# fp32 master copy on top of the low-precision live weights.
ADAM_MOMENT_SLOTS  = 2
MASTER_COPY_BYTES  = BYTES_FP32

# Coarse activation model: each block retains, per token, tensors on the order of
# (hidden + ffn) elements for the backward pass. ACT_OVERHEAD folds in attention
# scratch, norm buffers, and autograd's retained intermediates. Both constants are
# calibrated against measured MPS forward retention (see documentation/platform/
# mem_planner.md) and biased to slightly over-predict: under-prediction OOMs.
# Checkpointing trades the per-block store for recompute, retaining only inputs.
ACT_OVERHEAD          = 13.5
CHECKPOINT_ACT_RETAIN = 0.06

# Fraction of unified memory a run may claim. The rest is OS, framework, and MPS
# allocator fragmentation headroom; MPS cannot hand out the full pool.
USABLE_FRACTION = 0.85

TIER_NONE       = "none"
TIER_CHECKPOINT = "checkpoint_activations"
TIER_LOWP_OPT   = "checkpoint+bf16_optimizer"
TIER_PAGE       = "checkpoint+page_optimizer_to_nvme"
TIER_INFEASIBLE = "infeasible_reduce_batch_or_seq"

# ------------------------------------------------------------------------------------
# Functions


@dataclass(frozen=True)
class MemoryPlan:
    tier: str
    fits: bool
    budget_bytes: int
    required_bytes: int
    params_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activations_bytes: int


def _optimizer_bytes(param_count, dtype, moment_bytes=BYTES_FP32):
    state = param_count * ADAM_MOMENT_SLOTS * moment_bytes
    if dtype != "fp32":
        state += param_count * MASTER_COPY_BYTES
    return state


def _activation_bytes(batch, seq, hidden, ffn, layers, dtype, retain=1.0):
    per_token = (hidden + ffn) * ACT_OVERHEAD
    return int(batch * seq * layers * per_token * DTYPE_BYTES[dtype] * retain)


def plan_training_memory(param_count, hidden, layers, ffn, batch, seq,
                         dtype="bf16", budget_bytes=None):
    """Return the minimal MemoryPlan that fits this run on the unified-memory host.
    `param_count` is sum of model parameter element counts (trainer passes
    sum(p.numel() for p in model.parameters())). `dtype` is the live-weight dtype.
    `budget_bytes` overrides the auto-detected unified-memory budget (tests)."""
    if budget_bytes is None:
        budget_bytes = int(hardware.unified_memory_bytes() * USABLE_FRACTION)

    params = param_count * DTYPE_BYTES[dtype]
    grads  = params

    opt_full = _optimizer_bytes(param_count, dtype)
    opt_bf16 = _optimizer_bytes(param_count, dtype, moment_bytes=BYTES_BF16)

    act_full = _activation_bytes(batch, seq, hidden, ffn, layers, dtype)
    act_ckpt = _activation_bytes(batch, seq, hidden, ffn, layers, dtype,
                                 retain=CHECKPOINT_ACT_RETAIN)

    rungs = (
        (TIER_NONE,       opt_full, act_full),
        (TIER_CHECKPOINT, opt_full, act_ckpt),
        (TIER_LOWP_OPT,   opt_bf16, act_ckpt),
        (TIER_PAGE,       0,        act_ckpt),
    )

    for tier, optimizer, activations in rungs:
        required = params + grads + optimizer + activations
        if required <= budget_bytes:
            return MemoryPlan(tier, True, budget_bytes, required,
                              params, grads, optimizer, activations)

    required = params + grads + act_ckpt
    return MemoryPlan(TIER_INFEASIBLE, False, budget_bytes, required,
                      params, grads, 0, act_ckpt)


def format_plan(plan):
    gb = 1024 ** 3
    return (f"mem_planner tier={plan.tier} fits={plan.fits} "
            f"need={plan.required_bytes / gb:.1f}GB budget={plan.budget_bytes / gb:.1f}GB "
            f"[params={plan.params_bytes / gb:.1f} grads={plan.grads_bytes / gb:.1f} "
            f"opt={plan.optimizer_bytes / gb:.1f} act={plan.activations_bytes / gb:.1f}]")
