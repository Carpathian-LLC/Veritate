# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MultiMind plugin entry. Re-exports MultiMindPlugin so callers say
#   `from veritate_core.multimind import MultiMindPlugin`.
# veritate_core/multimind/__init__.py
# ------------------------------------------------------------------------------------

from .plugin import MultiMindPlugin

__all__ = ["MultiMindPlugin"]
