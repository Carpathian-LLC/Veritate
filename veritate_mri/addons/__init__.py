# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - inference-time addon registry. each addon under veritate_mri/addons/<name>/
#   ships a manifest.json plus addon.py exposing an Addon class with the
#   contract: reset(), observe(byte_int), bias_logits(logits) -> logits.
# - addons are stackable. compose with Chain. selection happens at the API
#   boundary: caller passes a list of addon names + per-addon param overrides;
#   the registry instantiates each, the chain pipes bias_logits through them in
#   list order before sampling.
# - addons never train, never touch the checkpoint, never see weights. they
#   read raw logits and the rolling byte buffer they maintain themselves.
# veritate_mri/addons/__init__.py
# ------------------------------------------------------------------------------------
# Imports

import importlib.util
import json
import os

# ------------------------------------------------------------------------------------
# Constants

HERE              = os.path.dirname(os.path.abspath(__file__))
MANIFEST_NAME     = "manifest.json"
ADDON_FILENAME    = "addon.py"
ADDON_CLASS_NAME  = "Addon"


# ------------------------------------------------------------------------------------
# Functions

def addons_dir():
    return HERE


def list_addons():
    """returns [{name, manifest}], one entry per discovered addon directory."""
    out = []
    for fn in sorted(os.listdir(HERE)):
        d = os.path.join(HERE, fn)
        if not os.path.isdir(d):
            continue
        mp = os.path.join(d, MANIFEST_NAME)
        ap = os.path.join(d, ADDON_FILENAME)
        if not (os.path.isfile(mp) and os.path.isfile(ap)):
            continue
        with open(mp, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        out.append({"id": fn, "manifest": manifest})
    return out


def _load_class(addon_id):
    d = os.path.join(HERE, addon_id)
    ap = os.path.join(d, ADDON_FILENAME)
    if not os.path.isfile(ap):
        raise FileNotFoundError("no addon.py for: " + addon_id)
    mod_name = "veritate_mri_addon_" + addon_id
    spec = importlib.util.spec_from_file_location(mod_name, ap)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, ADDON_CLASS_NAME, None)
    if cls is None:
        raise AttributeError(addon_id + "/addon.py missing class " + ADDON_CLASS_NAME)
    return cls


def instantiate(addon_id, params=None):
    """build one addon by id; params override manifest defaults."""
    cls = _load_class(addon_id)
    d = os.path.join(HERE, addon_id)
    with open(os.path.join(d, MANIFEST_NAME), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    defaults = {}
    for k, spec in (manifest.get("params") or {}).items():
        if isinstance(spec, dict) and "default" in spec:
            defaults[k] = spec["default"]
    if params:
        defaults.update({k: v for k, v in params.items() if v is not None})
    return cls(**defaults)


def build_chain(selection):
    """selection: list of either "name" or {"id": str, "params": dict}."""
    chain = Chain()
    for s in selection or []:
        if isinstance(s, str):
            chain.add(instantiate(s))
        elif isinstance(s, dict):
            chain.add(instantiate(s["id"], s.get("params")))
    return chain


# ------------------------------------------------------------------------------------
# Chain: stacks addons. logits flow through bias_logits in order.

class Chain:
    def __init__(self):
        self.addons = []

    def add(self, addon):
        self.addons.append(addon)
        return self

    def reset(self):
        for a in self.addons:
            a.reset()

    def observe(self, byte_int):
        for a in self.addons:
            a.observe(byte_int)

    def observe_bytes(self, bs):
        for b in bs:
            self.observe(b)

    def bias_logits(self, logits):
        out = logits
        for a in self.addons:
            out = a.bias_logits(out)
        return out

    def __len__(self):
        return len(self.addons)

    def __bool__(self):
        return len(self.addons) > 0
