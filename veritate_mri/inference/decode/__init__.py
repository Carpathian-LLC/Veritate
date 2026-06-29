# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Decode-path utilities used by the MRI inference backends. Per preflight
#   rule 11a, decoders never branch on model variant; they call the canonical
#   contract methods.
# - kv_cache.py    : O(1) per-step decode via cached K/V.
# - mtp_decode.py  : MTP-head-aware decoding.
# - constraints.py : Constraint primitives consumed by Brain.stream(constraint=...).
# veritate_mri/inference/decode/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from .kv_cache   import KVCachedDecoder
from .mtp_decode import MTPDecoder
from .constraints import (
    Constraint,
    JSONConstraint,
    VocabConstraint,
    StopOnConstraint,
)

# ------------------------------------------------------------------------------------
# Constants

__all__ = [
    "KVCachedDecoder",
    "MTPDecoder",
    "Constraint",
    "JSONConstraint",
    "VocabConstraint",
    "StopOnConstraint",
]
