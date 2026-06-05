# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Backwards-compatibility shim. The package was renamed to `veritate_core`
#   (and the runtime modules moved to `veritate_mri/{training,readers}`), but
#   trainer scripts still ship `from veritate.model import ...` /
#   `from veritate.plugin import save, paths, qat as ...` from the old layout.
#   This thin shim re-exports the new locations under the old names so those
#   trainers keep working without per-file edits.
# - Self-locating: ensures the repo root and veritate_mri are on sys.path before
#   re-exports so the shim works regardless of caller setup.
# - Resolves the file-vs-package conflict with `veritate.py` (the launcher) by
#   being a real package — Python's finder prefers a regular package over a
#   same-name module in the same directory. The launcher is still invoked as
#   `python veritate.py` (script execution, not import), so the shim doesn't
#   shadow it in practice.
# veritate/__init__.py
# ------------------------------------------------------------------------------------

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.normpath(_os.path.join(_HERE, ".."))
_MRI  = _os.path.join(_REPO, "veritate_mri")

if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
if _MRI  not in _sys.path: _sys.path.insert(0, _MRI)

from veritate_core import model as _vc_model    # noqa: E402
from veritate_core import qat   as _vc_qat      # noqa: E402

# Alias submodules so `from veritate.model import X` and `from veritate import
# qat as vqat` resolve to the same module objects as their veritate_core peers.
_sys.modules[__name__ + ".model"] = _vc_model
_sys.modules[__name__ + ".qat"]   = _vc_qat

# Also bind as attributes on the package so `from veritate import qat` works.
model = _vc_model
qat   = _vc_qat
