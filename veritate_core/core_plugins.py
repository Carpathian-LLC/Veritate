# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Catalog of "Core Plugins": dashboard-selectable knobs that mutate trainer
#   args before the trainer process is spawned. Each entry declares the args it
#   injects (or omits) and which group it belongs to. Plugins in the same group
#   are mutually exclusive; plugins in different groups compose freely.
# - The dashboard fetches this list via /core_plugins and renders one box per
#   entry under the "Core Plugins" section of the Training tab. On run, the
#   selected ids resolve to a flat dict of trainer args via `args_for_selection`.
# - Adding a new core plugin: append a dict to `REGISTRY`. No code change in
#   the dashboard, no code change in trainers, as long as the trainer already
#   accepts the underlying args via its manifest.defaults block.
# veritate_core/core_plugins.py
# ------------------------------------------------------------------------------------
# Imports


# ------------------------------------------------------------------------------------
# Constants

GROUP_ACTIVATION = "activation"
GROUP_REGULARIZER = "regularizer"

# Each plugin:
#   id           stable string key; never changes
#   label        short display name
#   description  one-line explainer; renders below the title in the dashboard box
#   group        plugins in the same group are mutually exclusive
#   default      whether the plugin is on by default on a fresh form
#   args         dict of trainer args to inject when selected
#   applies_to   list of trainer manifest.flow values this plugin shows up on;
#                empty list = all flows
REGISTRY = [
    {
        "id":          "gelu_baseline",
        "label":       "GeLU baseline",
        "description": "Smooth GeLU activation in every FFN. Dense activations, "
                       "no sparsity penalty. The default Veritate recipe.",
        "group":       GROUP_ACTIVATION,
        "default":     True,
        "args":        {"activation": "gelu"},
        "applies_to":  [],
    },
    {
        "id":          "relu_sparse",
        "label":       "ReLU activations",
        "description": "Switches every FFN to ReLU. Roughly half of post-activation "
                       "units sit at exactly zero, opening the door to sparsity-aware "
                       "inference kernels later.",
        "group":       GROUP_ACTIVATION,
        "default":     False,
        "args":        {"activation": "relu"},
        "applies_to":  [],
    },
    {
        "id":          "silu_swish",
        "label":       "SiLU (swish)",
        "description": "Smooth alternative to GeLU. Slightly different optimization "
                       "dynamics, no sparsity. Cheap to try as a baseline variant.",
        "group":       GROUP_ACTIVATION,
        "default":     False,
        "args":        {"activation": "silu"},
        "applies_to":  [],
    },
    {
        "id":          "l1_sparsity_light",
        "label":       "L1 sparsity (light)",
        "description": "Adds a small l1_lambda=1e-3 penalty on mean(|post-activation|). "
                       "Pairs with ReLU. Light setting: pushes a fraction of units "
                       "toward zero without crushing capacity.",
        "group":       GROUP_REGULARIZER,
        "default":     False,
        "args":        {"l1_lambda": 1e-3},
        "applies_to":  [],
    },
    {
        "id":          "l1_sparsity_strong",
        "label":       "L1 sparsity (strong)",
        "description": "l1_lambda=1e-2 — aggressive sparsity bias. Loss often climbs "
                       "noticeably before it descends. Pairs with ReLU only.",
        "group":       GROUP_REGULARIZER,
        "default":     False,
        "args":        {"l1_lambda": 1e-2},
        "applies_to":  [],
    },
    {
        "id":          "neuron_balance",
        "label":       "Neuron balance (100% use)",
        "description": "Adds a load-balance penalty (l1_lambda=1e-2, reg_mode=balance) "
                       "that pushes every FFN unit to carry equal load. The opposite of "
                       "L1 sparsity: drives toward zero dead neurons. Pairs with ReLU.",
        "group":       GROUP_REGULARIZER,
        "default":     False,
        "args":        {"l1_lambda": 1e-2, "reg_mode": "balance"},
        "applies_to":  [],
    },
    {
        "id":          "act_checkpoint",
        "label":       "Activation checkpointing",
        "description": "Recomputes the forward pass during backward instead of "
                       "storing activations. ~30% slower per step, ~50% less "
                       "activation VRAM. Pick when memory is tight.",
        "group":       "memory",
        "default":     False,
        "args":        {"use_act_ckpt": True},
        "applies_to":  [],
    },
    {
        "id":          "adam8bit",
        "label":       "8-bit AdamW",
        "description": "Optimizer state in INT8 instead of FP32 (bitsandbytes). "
                       "Saves ~6 bytes/param of VRAM. Required to fit 1B+ MoE on "
                       "12 GB. Tiny convergence cost.",
        "group":       "optimizer",
        "default":     False,
        "args":        {"use_8bit_adam": True},
        "applies_to":  [],
    },
]


# ------------------------------------------------------------------------------------
# Functions

def all_plugins(flow=None):
    """Return the list of core plugins, filtered to a manifest.flow when given."""
    if flow is None:
        return list(REGISTRY)
    return [p for p in REGISTRY if not p["applies_to"] or flow in p["applies_to"]]


def conflicts(selected_ids):
    """Return a list of (id_a, id_b) pairs where both ids belong to the same
    group. Empty list = the selection is valid."""
    by_id = {p["id"]: p for p in REGISTRY}
    seen_group = {}
    out = []
    for pid in selected_ids:
        p = by_id.get(pid)
        if p is None:
            continue
        g = p["group"]
        if g in seen_group:
            out.append((seen_group[g], pid))
        else:
            seen_group[g] = pid
    return out


def args_for_selection(selected_ids):
    """Resolve a set of selected plugin ids to a flat dict of trainer args.
    Later entries in REGISTRY override earlier ones when the same key recurs."""
    by_id = {p["id"]: p for p in REGISTRY}
    out = {}
    for pid in selected_ids:
        p = by_id.get(pid)
        if p is None:
            continue
        for k, v in p["args"].items():
            out[k] = v
    return out
