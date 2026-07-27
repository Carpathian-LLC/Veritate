# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Addressable external-memory retrieval tier: an on-disk leaf store keyed by
#   byte-native model embeddings, flat cosine retrieval over it, and an IVF
#   drill-down for sub-linear search at scale.
# veritate_core/memory/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from veritate_core.memory.hindex import HIndex
from veritate_core.memory.reader import encode_query, retrieve, search
from veritate_core.memory.store import MemStore, build, embed, load

# ------------------------------------------------------------------------------------
# Constants

__all__ = ["HIndex", "MemStore", "build", "embed", "encode_query", "load", "retrieve", "search"]
