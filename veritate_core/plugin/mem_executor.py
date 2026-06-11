# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Applies a MemoryPlan to a model. Owns the activation-checkpointing wiring that
#   trainers used to monkeypatch inline, so the decision is plan-driven and shared
#   instead of a per-trainer flag. Wraps each transformer block's forward in
#   torch.utils.checkpoint when the plan's tier calls for it; the block list is the
#   model's `.blocks` contract, common to every Veritate model class.
# - Optimizer-level rungs (bf16 moments, NVMe paging) are not wired here yet. When a
#   plan needs one, apply_plan returns it in `unmet` so the trainer surfaces the gap
#   loudly rather than silently under-delivering and OOMing.
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

# Tier -> intervention this executor does not yet perform. Reported in `unmet`.
UNWIRED_INTERVENTIONS = {
    mem_planner.TIER_LOWP_OPT: "bf16_optimizer_state",
    mem_planner.TIER_PAGE:     "page_optimizer_to_nvme",
}

# ------------------------------------------------------------------------------------
# Functions


@dataclass(frozen=True)
class AppliedPlan:
    tier: str
    grad_checkpoint: bool
    unmet: tuple


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
    """Apply the interventions a MemoryPlan calls for. Returns what was applied and
    which interventions the plan needs that this executor does not yet perform."""
    grad_checkpoint = plan.tier in CHECKPOINT_TIERS
    if grad_checkpoint:
        enable_grad_checkpoint(model)
    unmet = tuple(name for tier, name in UNWIRED_INTERVENTIONS.items()
                  if tier == plan.tier)
    return AppliedPlan(plan.tier, grad_checkpoint, unmet)
