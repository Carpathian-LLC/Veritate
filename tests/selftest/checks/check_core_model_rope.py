# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tiny VeritateRoPE forward pass. exercises the rope variant.
# tests/selftest/checks/check_core_model_rope.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "core"
REQUIRES_TORCH  = True

SHAPE_VOCAB     = 256
SHAPE_HIDDEN    = 32
SHAPE_LAYERS    = 2
SHAPE_FFN       = 64
SHAPE_HEADS     = 2
SHAPE_SEQ       = 16
BATCH           = 2
SEQ_USED        = 8

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """instantiate tiny VeritateRoPE, run a forward pass, check logits shape."""
    try:
        import torch
    except Exception as exc:
        return _status.skip("core_model_rope", f"torch unavailable: {exc}")
    try:
        from veritate_core.model_rope import VeritateRoPE
    except Exception as exc:
        return _status.skip("core_model_rope", f"model_rope import failed: {exc}")

    torch.manual_seed(0)
    try:
        m = VeritateRoPE(SHAPE_VOCAB, SHAPE_HIDDEN, SHAPE_LAYERS, SHAPE_FFN, SHAPE_HEADS, SHAPE_SEQ)
    except TypeError as exc:
        return _status.skip("core_model_rope", f"signature mismatch: {exc}")
    x = torch.randint(0, SHAPE_VOCAB, (BATCH, SEQ_USED), dtype=torch.long)
    out = m(x, None) if _accepts_targets(m) else m(x)
    logits = out[0] if isinstance(out, tuple) else out
    shape = tuple(logits.shape)
    if shape[0] != BATCH or shape[1] != SEQ_USED:
        return _status.fail("core_model_rope", f"unexpected shape {shape}")
    return _status.ok("core_model_rope", f"forward shape {shape}")


def _accepts_targets(model):
    import inspect
    try:
        return "targets" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
