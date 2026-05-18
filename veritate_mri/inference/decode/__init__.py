# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Decode-path utilities used by the MRI inference backends. Each module is
#   model-agnostic and works against any model class that implements the
#   cross-model contract: project_byte0, ensure_context, run_blocks, embed,
#   supports_mtp_decode. Per preflight rule 11a, decoders never branch on
#   model variant.
# - kv_cache.py     : O(1) per-step decode via cached K/V.
# - mtp_decode.py   : MTP-head-aware decoding.
# - constrained.py  : Output-shape constraints (JSON / vocab / stop-pattern).
# - constraints.py  : Constraint primitives.
# - exit_head.py    : Tiny per-layer exit classifier for adaptive depth.
# - kangaroo.py     : Self-speculative draft (target's first L/4 + adapter).
# - eagle3.py       : EAGLE-3 draft head consuming target's tap-layer hiddens.
# veritate_mri/decode/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from .kv_cache    import KVCachedDecoder
from .mtp_decode  import MTPDecoder
from .constrained import ConstrainedDecoder, ConstrainedStats
from .constraints import (
    Constraint,
    JSONConstraint,
    VocabConstraint,
    StopOnConstraint,
    CombineConstraint,
)

# ------------------------------------------------------------------------------------
# Constants

__all__ = [
    "KVCachedDecoder",
    "MTPDecoder",
    "ConstrainedDecoder",
    "ConstrainedStats",
    "Constraint",
    "JSONConstraint",
    "VocabConstraint",
    "StopOnConstraint",
    "CombineConstraint",
    "byte0_projector",
]


# ------------------------------------------------------------------------------------
# Functions

def byte0_projector(model):
    """Return the model's project_byte0 method.

    Every Veritate model exposes project_byte0(residual) per preflight rule 11a.
    This wrapper exists for callers that want a callable handle; new model
    variants must implement project_byte0 on the model class itself, never
    here.
    """
    project_fn = getattr(model, "project_byte0", None)
    if not callable(project_fn):
        raise TypeError(
            f"model {type(model).__name__} does not implement project_byte0(residual); "
            "every Veritate model class must define this contract method (rule 11a)."
        )
    return project_fn
