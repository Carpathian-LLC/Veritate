# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The platform-side entry point that plugins call into. The full surface is
#   specified in documentation/plugins/contract.md.
# - This module is the only thing plugins are allowed to import from outside
#   their own bundle. Internals of veritate_mri/ are not part of the contract
#   and must not be reached into directly.
# veritate/plugin/__init__.py
# ------------------------------------------------------------------------------------

import os
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.normpath(os.path.join(_HERE, "..", ".."))
_VERITATE_MRI = os.path.join(_REPO_ROOT, "veritate_mri")
if _VERITATE_MRI not in sys.path:
    sys.path.insert(0, _VERITATE_MRI)

import save                       # noqa: E402  veritate_mri/save.py
from readers import paths         # noqa: E402  veritate_mri/readers/paths.py
from veritate import model        # noqa: E402  veritate/model.py
from veritate import qat          # noqa: E402  veritate/qat.py

__all__ = ["save", "paths", "model", "qat"]
