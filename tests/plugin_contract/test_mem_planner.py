# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - preflight rule 34c: the training-memory budget derives from the detected host
#   memory pool and anything printed names the specific pool (CPU/physical RAM,
#   CUDA/VRAM, MPS/unified memory), never a bare "RAM". Covers the escalation
#   ladder, the budget derivation, and the bench memory-kind labeller.
# - pure arithmetic: no torch device, no allocation, no probe of the real host.
# tests/plugin_contract/test_mem_planner.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest

from veritate_core.plugin import bench, mem_planner

# ------------------------------------------------------------------------------------
# Constants

MB = 1024 ** 2
GB = 1024 ** 3

SHAPE = {"param_count": 100_000_000, "hidden": 1024, "layers": 12, "ffn": 4096,
         "batch": 4, "seq": 512}

HOST_MEMORY_BYTES = 64 * GB
HUGE_BUDGET       = 1024 * GB
TINY_BUDGET       = 64 * MB   # below params+grads alone: no rung can fit

# rule 34c: each device kind names its own pool, and none of them is a bare "RAM".
DEVICE_MEMORY_LABELS = {"cuda": "VRAM", "mps": "unified memory", "cpu": "physical RAM"}
GENERIC_LABEL = "RAM"

FORMAT_LABEL_REASON = ("rule 34c gap: mem_planner.format_plan prints budget=<n>GB with no "
                       "memory-kind label, so a CUDA host's line is indistinguishable from "
                       "an MPS host's")

# ------------------------------------------------------------------------------------
# Functions

def _plan(**over):
    args = dict(SHAPE)
    args.update(over)
    return mem_planner.plan_training_memory(**args)


@pytest.fixture
def fixed_host(monkeypatch):
    """Pin the detected host memory so the budget is derived, not probed."""
    monkeypatch.setattr(mem_planner.hardware, "unified_memory_bytes", lambda: HOST_MEMORY_BYTES)


def test_budget_derives_from_the_detected_host_memory(fixed_host):
    """The plan's budget is the detected host pool scaled by USABLE_FRACTION."""
    assert _plan().budget_bytes == int(HOST_MEMORY_BYTES * mem_planner.USABLE_FRACTION)


def test_budget_override_replaces_the_detected_pool(fixed_host):
    """An explicit budget_bytes bypasses host detection entirely."""
    assert _plan(budget_bytes=TINY_BUDGET).budget_bytes == TINY_BUDGET


def test_a_roomy_budget_needs_no_escalation():
    """A run that fits outright plans the no-escalation tier."""
    assert _plan(budget_bytes=HUGE_BUDGET).tier == mem_planner.TIER_NONE


def test_a_roomy_budget_is_marked_fitting():
    """A run that fits outright is reported as fitting."""
    assert _plan(budget_bytes=HUGE_BUDGET).fits is True


def test_an_impossible_budget_is_marked_infeasible():
    """A run that cannot fit any rung plans the infeasible tier."""
    assert _plan(budget_bytes=TINY_BUDGET).tier == mem_planner.TIER_INFEASIBLE


def test_an_impossible_budget_is_not_marked_fitting():
    """A run that cannot fit any rung is not reported as fitting."""
    assert _plan(budget_bytes=TINY_BUDGET).fits is False


def test_checkpointing_is_the_first_rung_taken():
    """A run over budget only on activations escalates to checkpointing first."""
    roomy = _plan(budget_bytes=HUGE_BUDGET)
    just_short = roomy.required_bytes - 1
    assert _plan(budget_bytes=just_short).tier == mem_planner.TIER_CHECKPOINT


def test_checkpointing_shrinks_the_activation_bucket():
    """The checkpoint rung retains a fraction of the full activation bucket."""
    roomy = _plan(budget_bytes=HUGE_BUDGET)
    ckpt = _plan(budget_bytes=roomy.required_bytes - 1)
    assert ckpt.activations_bytes < roomy.activations_bytes


def test_paging_the_optimizer_zeroes_the_optimizer_bucket():
    """The paged rung reports zero optimizer bytes in the budget."""
    roomy = _plan(budget_bytes=HUGE_BUDGET)
    ckpt = _plan(budget_bytes=roomy.required_bytes - 1)
    paged = _plan(budget_bytes=ckpt.required_bytes - ckpt.optimizer_bytes)
    assert paged.optimizer_bytes == 0


def test_required_bytes_is_the_sum_of_the_four_buckets():
    """A plan's required bytes is exactly params plus grads plus optimizer plus activations."""
    p = _plan(budget_bytes=HUGE_BUDGET)
    assert p.required_bytes == \
        p.params_bytes + p.grads_bytes + p.optimizer_bytes + p.activations_bytes


def test_bf16_halves_the_parameter_bucket():
    """The live-weight dtype drives the parameter bucket size."""
    fp32 = _plan(budget_bytes=HUGE_BUDGET, dtype="fp32").params_bytes
    assert _plan(budget_bytes=HUGE_BUDGET, dtype="bf16").params_bytes * 2 == fp32


@pytest.mark.parametrize("device,label", sorted(DEVICE_MEMORY_LABELS.items()))
def test_memory_kind_names_the_pool_for_device(device, label):
    """Each device kind labels the memory pool the budget actually measures."""
    assert bench._memory_kind(device)[0] == label


@pytest.mark.parametrize("device", sorted(DEVICE_MEMORY_LABELS))
def test_memory_kind_is_never_a_bare_ram_label(device):
    """No device kind reports the generic RAM label rule 34c bans."""
    assert bench._memory_kind(device)[0] != GENERIC_LABEL


def test_memory_kind_labels_are_distinct():
    """The three device kinds report three distinct memory-pool labels."""
    assert len({bench._memory_kind(d)[0] for d in DEVICE_MEMORY_LABELS}) == \
        len(DEVICE_MEMORY_LABELS)


@pytest.mark.xfail(strict=True, reason=FORMAT_LABEL_REASON)
def test_format_plan_names_the_memory_kind():
    """The formatted plan line names which memory pool the budget measures."""
    line = mem_planner.format_plan(_plan(budget_bytes=HUGE_BUDGET))
    assert any(label in line for label in DEVICE_MEMORY_LABELS.values())
