# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mem_planner picks the minimal escalation tier that fits a run in unified
#   memory. These pin the escalation ladder boundaries with explicit budgets so
#   the contract is deterministic and host-independent.
# tests/plugin_contract/test_mem_planner.py
# ------------------------------------------------------------------------------------
# Imports

from veritate_core.plugin import mem_planner as mp

# ------------------------------------------------------------------------------------
# Constants

GB = 1024 ** 3
# 1B-class byte-level shape used across the boundary cases.
SHAPE = dict(param_count=1_000_000_000, hidden=2048, layers=24, ffn=8192,
             batch=8, seq=1024, dtype="bf16")

# ------------------------------------------------------------------------------------
# Functions


def test_planner_is_exported():
    """mem_planner exposes plan_training_memory on the plugin surface."""
    assert hasattr(mp, "plan_training_memory")


def test_huge_budget_needs_no_offload():
    """A budget far above the footprint stays on tier none."""
    plan = mp.plan_training_memory(budget_bytes=512 * GB, **SHAPE)
    assert plan.tier == mp.TIER_NONE
    assert plan.fits


def test_tight_budget_escalates_to_checkpoint():
    """A budget just under the full footprint engages activation checkpointing."""
    full = mp.plan_training_memory(budget_bytes=10 ** 18, **SHAPE)
    just_under = full.required_bytes - 1
    plan = mp.plan_training_memory(budget_bytes=just_under, **SHAPE)
    assert plan.tier == mp.TIER_CHECKPOINT
    assert plan.fits
    assert plan.required_bytes <= just_under


def test_smaller_budget_pages_optimizer():
    """Below the bf16-optimizer footprint, the optimizer pages to NVMe (0 in RAM)."""
    plan = mp.plan_training_memory(budget_bytes=8 * GB, **SHAPE)
    assert plan.tier == mp.TIER_PAGE
    assert plan.optimizer_bytes == 0
    assert plan.fits


def test_impossible_budget_is_infeasible():
    """A budget below params+grads alone is reported infeasible, not fits."""
    plan = mp.plan_training_memory(budget_bytes=1 * GB, **SHAPE)
    assert plan.tier == mp.TIER_INFEASIBLE
    assert not plan.fits


def test_tier_monotonic_in_budget():
    """Shrinking the budget never moves the plan to a cheaper tier."""
    order = [mp.TIER_NONE, mp.TIER_CHECKPOINT, mp.TIER_LOWP_OPT,
             mp.TIER_PAGE, mp.TIER_INFEASIBLE]
    prev = -1
    for gb in (512, 64, 24, 12, 4, 1):
        plan = mp.plan_training_memory(budget_bytes=gb * GB, **SHAPE)
        idx = order.index(plan.tier)
        assert idx >= prev, f"{gb}GB regressed to {plan.tier}"
        prev = idx


def test_size_adaptive_same_budget_different_model():
    """Same host budget: a small model fits clean where a large one must offload."""
    budget = 64 * GB
    small = mp.plan_training_memory(budget_bytes=budget, param_count=200_000_000,
                                    hidden=1024, layers=12, ffn=4096,
                                    batch=8, seq=1024, dtype="bf16")
    large = mp.plan_training_memory(budget_bytes=budget, param_count=13_000_000_000,
                                    hidden=4096, layers=40, ffn=16384,
                                    batch=8, seq=1024, dtype="bf16")
    assert small.tier == mp.TIER_NONE
    assert large.tier != mp.TIER_NONE
