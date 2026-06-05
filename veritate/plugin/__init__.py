# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Backwards-compat shim for `veritate.plugin.*`. Trainer scripts written
#   against the old layout import:
#       from veritate.plugin import save, paths, qat as qat_helpers
#       from veritate.plugin import model as _veritate_model
#   This shim maps each of those names to its current canonical location:
#       save  -> veritate_mri/training/save.py
#       paths -> veritate_mri/readers/paths.py
#       model -> veritate_core/model.py
#       qat   -> veritate_core/qat.py
# - The parent `veritate/__init__.py` already puts repo root and veritate_mri
#   on sys.path, so the re-exports below work even when imported by a script
#   that hasn't done the path setup itself.
# veritate/plugin/__init__.py
# ------------------------------------------------------------------------------------

import sys as _sys

from training      import save              # noqa: E402, F401
from readers       import paths             # noqa: E402, F401
from veritate_core import model             # noqa: E402, F401
from veritate_core import qat               # noqa: E402, F401

# Make `from veritate.plugin.save import ...` (rare) also resolve.
_sys.modules[__name__ + ".save"]  = save
_sys.modules[__name__ + ".paths"] = paths
_sys.modules[__name__ + ".model"] = model
_sys.modules[__name__ + ".qat"]   = qat
