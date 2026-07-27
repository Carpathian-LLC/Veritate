# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - preflight rule 11a: inference/decode never branches on the model variant, so
#   every model class veritate_core can construct exposes the same contract
#   (embed / ensure_context / run_blocks / project_byte0 + the shape attributes
#   KVCachedDecoder reads). Variants are discovered from the package, so a NEW
#   variant that skips the contract fails here instead of at decode time.
# - CPU-only: every model is built at a toy shape, nothing is moved to a device.
# tests/core/test_model_variant_contract.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib
import inspect
import pkgutil

import pytest
import torch

import veritate_core
from veritate_core import load

# ------------------------------------------------------------------------------------
# Constants

# Honored by every variant today.
UNIVERSAL_CONTRACT = ("embed", "project_byte0", "forward")
# The extra surface KVCachedDecoder calls blindly (inference/decode/kv_cache.py).
KV_DECODE_CONTRACT = ("ensure_context", "run_blocks", "kv_cache_patch_attn")
CONTRACT_ATTRS     = ("hidden", "layers", "heads", "seq")

KV_GAP_REASON = ("rule 11a gap: VeritateMemory / VeritatePatched / VeritateRecurrent do not "
                 "implement the KV-decode contract, so KVCachedDecoder raises AttributeError "
                 "on them at decode time")
CLASS_PREFIX     = "Veritate"

TOY = {"vocab": 256, "hidden": 16, "layers": 2, "ffn": 32, "heads": 2, "seq": 32}
SEED  = 0
BATCH = 1
TOKENS = 8

# ------------------------------------------------------------------------------------
# Functions

def _variant_classes():
    """Every Veritate* model class in veritate_core, keyed by name."""
    found = {}
    for mod in pkgutil.iter_modules(veritate_core.__path__):
        if not mod.name.startswith("model"):
            continue
        module = importlib.import_module(f"{veritate_core.__name__}.{mod.name}")
        for name, obj in vars(module).items():
            if (name.startswith(CLASS_PREFIX) and inspect.isclass(obj)
                    and issubclass(obj, torch.nn.Module) and obj.__module__ == module.__name__):
                found[name] = obj
    return found


VARIANTS = _variant_classes()
VARIANT_PARAM = pytest.mark.parametrize("cls", list(VARIANTS.values()), ids=list(VARIANTS))


def _build(cls):
    torch.manual_seed(SEED)
    params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in TOY.items() if k in params}
    return cls(**kwargs)


def test_variant_registry_is_not_empty():
    """veritate_core exposes at least one Veritate model variant to test."""
    assert VARIANTS != {}


def test_canonical_model_is_in_the_registry():
    """The canonical dense Veritate trunk is one of the discovered variants."""
    assert "Veritate" in VARIANTS


@VARIANT_PARAM
@pytest.mark.parametrize("method", UNIVERSAL_CONTRACT)
def test_variant_declares_universal_contract_method(cls, method):
    """Every model variant declares the shared decode contract method."""
    assert callable(getattr(cls, method, None))


@pytest.mark.xfail(strict=True, reason=KV_GAP_REASON)
@pytest.mark.parametrize("method", KV_DECODE_CONTRACT)
def test_every_variant_declares_the_kv_decode_contract(method):
    """Every model variant declares the KV-decode contract KVCachedDecoder calls blindly."""
    assert [n for n, c in VARIANTS.items() if not callable(getattr(c, method, None))] == []


@VARIANT_PARAM
@pytest.mark.parametrize("attr", CONTRACT_ATTRS)
def test_variant_instance_exposes_shape_attribute(cls, attr):
    """Every built model variant exposes the shape attribute the KV decoder reads."""
    assert isinstance(getattr(_build(cls), attr), int)


@VARIANT_PARAM
def test_variant_project_byte0_returns_vocab_logits(cls):
    """project_byte0 maps a trunk residual to next-byte logits for every variant."""
    model = _build(cls)
    logits = model.project_byte0(torch.zeros(BATCH, TOKENS, model.hidden))
    assert logits.shape[-1] == TOY["vocab"]


@VARIANT_PARAM
def test_variant_project_byte0_preserves_the_token_axis(cls):
    """project_byte0 emits one logit row per residual position for every variant."""
    model = _build(cls)
    logits = model.project_byte0(torch.zeros(BATCH, TOKENS, model.hidden))
    assert logits.shape[:2] == (BATCH, TOKENS)


def test_shape_from_state_dict_reads_a_dense_checkpoint():
    """load.shape_from_state_dict recovers the toy shape from a dense state_dict."""
    model = _build(VARIANTS["Veritate"])
    shape = load.shape_from_state_dict(model.state_dict(), {"heads": TOY["heads"]})
    assert {k: shape[k] for k in ("vocab", "hidden", "layers", "seq")} == \
        {k: TOY[k] for k in ("vocab", "hidden", "layers", "seq")}


def test_shape_from_state_dict_rejects_a_ropeless_seq():
    """A checkpoint with no pos_emb and no seq in cfg raises instead of guessing a window."""
    model = _build(VARIANTS["Veritate"])
    sd = {k: v for k, v in model.state_dict().items() if k != load.POS_EMB_KEY}
    with pytest.raises(RuntimeError, match="seq"):
        load.shape_from_state_dict(sd, {})
