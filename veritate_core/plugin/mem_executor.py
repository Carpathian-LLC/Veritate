# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Applies a MemoryPlan to a training run. Two halves: apply_plan owns the model-side
#   activation-checkpointing wiring (wraps each block's forward in torch.utils.checkpoint
#   when the tier calls for it; the block list is the model's `.blocks` contract), and
#   make_optimizer owns the optimizer-side choice. For the optimizer-offload tiers
#   (bf16_optimizer, page_optimizer_to_nvme) make_optimizer returns a PagedAdamW whose
#   moment buffers live on NVMe, dropping resident optimizer memory toward zero.
# - apply_plan reports `optimizer_offload`: a caller that checkpoints but then builds a
#   plain in-RAM optimizer for an offload tier would silently under-deliver and OOM, so
#   the flag tells the caller it MUST route the optimizer through make_optimizer.
# veritate_core/plugin/mem_executor.py
# ------------------------------------------------------------------------------------
# Imports

from dataclasses import dataclass

from veritate_core.plugin import mem_planner

# ------------------------------------------------------------------------------------
# Constants

CHECKPOINT_TIERS = frozenset((
    mem_planner.TIER_CHECKPOINT,
    mem_planner.TIER_LOWP_OPT,
    mem_planner.TIER_PAGE,
))

# Tiers whose memory saving comes from moving optimizer state off the resident pool.
# Both are delivered by paging the Adam moments to NVMe (a superset of the bf16-moment
# saving the planner models for the lighter rung), so one mechanism serves both.
OFFLOAD_TIERS = frozenset((
    mem_planner.TIER_LOWP_OPT,
    mem_planner.TIER_PAGE,
))

# ------------------------------------------------------------------------------------
# Functions


@dataclass(frozen=True)
class AppliedPlan:
    tier: str
    grad_checkpoint: bool
    optimizer_offload: bool
    unmet: tuple = ()   # retained for back-compat; every tier is now wired, so always ()


def _checkpointed(forward):
    import torch
    def run(x):
        return torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
    return run


def enable_grad_checkpoint(model):
    """Wrap each block's forward in activation checkpointing. Idempotent: a block
    already wrapped is left alone so repeated calls do not nest checkpoints."""
    for blk in model.blocks:
        if getattr(blk.forward, "_grad_checkpointed", False):
            continue
        wrapped = _checkpointed(blk.forward)
        wrapped._grad_checkpointed = True
        blk.forward = wrapped


def apply_plan(model, plan):
    """Apply the model-side interventions a MemoryPlan calls for (activation
    checkpointing). `optimizer_offload` tells the caller it must build the optimizer
    via make_optimizer for this tier rather than a plain in-RAM AdamW."""
    grad_checkpoint = plan.tier in CHECKPOINT_TIERS
    if grad_checkpoint:
        enable_grad_checkpoint(model)
    return AppliedPlan(plan.tier, grad_checkpoint, plan.tier in OFFLOAD_TIERS)


def make_optimizer(params, plan, *, lr, betas, eps, weight_decay, state_dir=None):
    """Return the optimizer the plan's tier requires: a NVMe-paged AdamW for the
    optimizer-offload tiers, a plain in-RAM AdamW otherwise. `state_dir` is where the
    paged moment files live (kept across resume); None gives a throwaway temp dir."""
    if plan.tier in OFFLOAD_TIERS:
        from veritate_core.plugin import paged_optimizer
        return paged_optimizer.PagedAdamW(params, lr=lr, betas=betas, eps=eps,
                                          weight_decay=weight_decay, state_dir=state_dir)
    import torch
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps,
                             weight_decay=weight_decay)
