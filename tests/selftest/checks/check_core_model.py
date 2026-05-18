# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tiny Veritate forward pass. catches model regressions without needing a real
#   checkpoint or GPU. CPU only.
# tests/selftest/checks/check_core_model.py
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
    """instantiate tiny Veritate, run a forward pass, check logits shape."""
    try:
        import torch
    except Exception as exc:
        return _status.skip("core_model", f"torch unavailable: {exc}")

    from veritate_core.model import Veritate, VOCAB_BYTE_LEVEL
    assert VOCAB_BYTE_LEVEL == SHAPE_VOCAB

    torch.manual_seed(0)
    m = Veritate(SHAPE_VOCAB, SHAPE_HIDDEN, SHAPE_LAYERS, SHAPE_FFN, SHAPE_HEADS, SHAPE_SEQ)
    x = torch.randint(0, SHAPE_VOCAB, (BATCH, SEQ_USED), dtype=torch.long)
    logits, loss = m(x, None) if _accepts_targets(m) else (m(x), None)
    shape = tuple(logits.shape)
    expected = (BATCH, SEQ_USED, SHAPE_VOCAB)
    if shape != expected:
        return _status.fail("core_model", f"logits shape {shape} != {expected}")
    return _status.ok("core_model", f"forward shape {shape}", {"params": _param_count(m)})


def _accepts_targets(model):
    """Veritate.forward signature accepts (tokens, targets) per the contract."""
    import inspect
    try:
        sig = inspect.signature(model.forward)
        return "targets" in sig.parameters
    except (TypeError, ValueError):
        return False


def _param_count(model):
    return sum(p.numel() for p in model.parameters())
