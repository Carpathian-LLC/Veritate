# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - user-data compat at the loader: a checkpoint with multi-byte heads (the retired
#   veritate_800m / 85m trainers, whose classes left with trainers/) is refused with a
#   message that names the path that still serves it, instead of an import crash; the
#   RoPE-only and canonical layouts still load.
# tests/core/test_load_retired_checkpoints.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core import load
from veritate_core.model import VOCAB_BYTE_LEVEL, Veritate
from veritate_core.model_rope import VeritateRoPE

# ------------------------------------------------------------------------------------
# Constants

SHAPE = {"hidden": 16, "layers": 1, "ffn": 32, "heads": 2, "seq": 8}

# ------------------------------------------------------------------------------------
# Functions


def test_a_multi_byte_head_checkpoint_is_refused_with_the_serving_path_named():
    sd = VeritateRoPE(vocab=VOCAB_BYTE_LEVEL, **SHAPE).state_dict()
    sd["mtp.transforms.0.weight"] = torch.zeros(SHAPE["hidden"], SHAPE["hidden"])
    with pytest.raises(RuntimeError, match=r"multi-byte prediction heads.*C engine"):
        load.load_from_state_dict(sd, {}, strict_canonical=False)


def test_rope_only_and_canonical_checkpoints_still_load():
    torch.manual_seed(0)
    rope = VeritateRoPE(vocab=VOCAB_BYTE_LEVEL, **SHAPE)
    got = load.load_from_state_dict(rope.state_dict(), {"seq": SHAPE["seq"]}, strict_canonical=False)
    assert isinstance(got, VeritateRoPE) and got.seq == SHAPE["seq"]
    plain = Veritate(vocab=VOCAB_BYTE_LEVEL, **SHAPE)
    got = load.load_from_state_dict(plain.state_dict(), {"heads": SHAPE["heads"]}, strict_canonical=True)
    assert isinstance(got, Veritate)
    toks = torch.randint(0, VOCAB_BYTE_LEVEL, (1, 8))
    assert torch.equal(got.eval()(toks)[0], plain.eval()(toks)[0])
