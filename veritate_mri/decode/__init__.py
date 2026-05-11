# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Decode-path utilities used by the MRI inference backends. Each module is
#   model-agnostic and can be applied to any Veritate / Veritate800M instance
#   the Brain backend has already loaded.
# - kv_cache.py     : O(1) per-step decode via cached K/V. Byte-exact identical
#                     output to the un-cached path. Composes with MTP / speculative.
# - mtp_decode.py   : MTP-head-aware decoding for Veritate800M (head0_only /
#                     accept_all / verify). Speculative verify is lossless and
#                     byte-exact equivalent to head0-only decode.
# - constrained.py  : Output-shape constraints (JSON / vocab / stop-pattern).
#                     Pure logit masking, applies on top of any decode mode.
# - constraints.py  : Constraint primitives used by constrained.py.
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
]
