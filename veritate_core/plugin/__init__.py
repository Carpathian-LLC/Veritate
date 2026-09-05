# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The platform-side entry point that plugins call into. The full surface is
#   specified in documentation.md (training, plugin surface).
# - This module is the only thing plugins are allowed to import from outside
#   their own bundle. Internals of veritate_mri/ are not part of the contract
#   and must not be reached into directly.
# veritate_core/plugin/__init__.py
# ------------------------------------------------------------------------------------

import os
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.normpath(os.path.join(_HERE, "..", ".."))
_VERITATE_MRI = os.path.join(_REPO_ROOT, "veritate_mri")
if _VERITATE_MRI not in sys.path:
    sys.path.insert(0, _VERITATE_MRI)

from readers import paths
from training import save

from veritate_core import (
    model,
    model_memory,
    model_patched,
    model_recurrent,
    qat,
)
from veritate_core.plugin import (
    bench,
    hardware,
    image_codec,
    image_decode,
    image_grid,
    mem_executor,
    mem_planner,
    multicorpus,
    oom_recovery,
    optim,
    slm,
)


def get_teacher_client(provider_override=None, model_override=None):
    from runtime import settings as settings_mod

    from veritate_mri.teacher import Client, get_provider, resolve_api_key
    cfg = settings_mod.get()
    provider_id = provider_override or cfg.get("teacher_provider", "")
    if not provider_id:
        return None
    get_provider(provider_id)
    model = model_override or cfg.get("teacher_model", "") or None
    base_url = cfg.get("teacher_base_url", "") or None
    api_key = resolve_api_key(provider_id, cfg.get("teacher_api_key", "") or None)
    return Client(provider_id, model=model, base_url=base_url, api_key=api_key)


__all__ = [
    "bench",
    "get_teacher_client",
    "hardware",
    "mem_executor",
    "mem_planner",
    "model",
    "paths",
    "qat",
    "save",
]
