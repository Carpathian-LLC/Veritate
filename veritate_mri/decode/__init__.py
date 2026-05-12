# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Decode-path utilities used by the MRI inference backends. Each module is
#   model-agnostic and can be applied to any byte-level model variant the
#   Brain backend has already loaded.
# - byte0_projector() is the SINGLE PLACE where output-projection dispatch
#   happens. It returns a callable that maps a residual tensor to byte-0
#   logits, regardless of which model variant (canonical Veritate,
#   VeritateRoPE, Veritate800M, future variants). Inference modules accept
#   the callable as input; they never branch on model attributes
#   themselves. Per preflight rule 11a, the model knows what it is; the
#   consumer does not.
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
    """Return a callable `(residual) -> logits` that produces byte-0 logits
    from a model's intermediate or final residual.

    The model's contract:
      - If `model.project_byte0` exists and is callable, USE THAT. (preferred)
        Model classes that want full control over their output projection
        define this method. They know their own architecture; consumers
        do not.
      - Otherwise, fall back to a structural inspection ONCE here. This
        is the only place in the decode package that inspects model
        attributes. If you need to add a new variant, extend THIS function
        and nothing else.

    The callable accepts a residual tensor of arbitrary leading shape
    (typically [..., hidden]) and returns the same shape with the last
    dim replaced by vocab.
    """
    project_fn = getattr(model, "project_byte0", None)
    if callable(project_fn):
        return project_fn

    # Fallback: structural dispatch. KEEP THIS THE ONLY PLACE that does it.
    has_n_out  = hasattr(model, "n_out")
    has_lm     = hasattr(model, "lm_head")
    has_mtp    = hasattr(model, "mtp") and hasattr(model.mtp, "lm_head")
    if not has_n_out or not has_lm:
        raise TypeError(
            f"byte0_projector: model {type(model).__name__} exposes neither "
            f"project_byte0() nor the (n_out, lm_head) pair. Either add "
            f"project_byte0(residual) to the model class, or extend this "
            f"function with a new branch."
        )

    if has_mtp and hasattr(model.mtp, "norms") and hasattr(model.mtp, "transforms"):
        # MTP variant (e.g., Veritate800M): byte-0 routes through MTP head-0.
        def _project_mtp(residual):
            x = model.n_out(residual)
            h0 = model.mtp.norms[0](model.mtp.transforms[0](x))
            return model.mtp.lm_head(h0)
        return _project_mtp

    # Canonical Veritate / VeritateRoPE: tied lm_head directly.
    def _project_plain(residual):
        x = model.n_out(residual)
        return model.lm_head(x)
    return _project_plain
