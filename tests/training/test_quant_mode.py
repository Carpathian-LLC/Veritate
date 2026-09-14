# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers the trainer's quant_mode flag: the Training tab offers int8 / int4 / ternary
#   under QAT, and until 2026-09-08 the trainer listed the flag as ignorable and never
#   applied it, so every QAT run rounded to int8 whatever was picked. Pins that the flag
#   parses, that a bad value is refused up front, and that qat.set_quant_mode reaches
#   every quantized layer.
# tests/training/test_quant_mode.py
# ------------------------------------------------------------------------------------
# Imports:
# ------------------------------------------------------------------------------------
import sys

import pytest
from training import veritate_trainer as vt

from veritate_core import qat
from veritate_core.model import QuantLinear, Veritate

# ------------------------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------------------------


def test_quant_mode_is_a_reserved_flag_not_an_ignored_one(monkeypatch):
    """--quant_mode parses on every launch, defaults to int8, and is no longer dropped."""
    assert "quant_mode" not in vt.SCHEMA_IGNORED_FLAGS
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py", "--quant_mode", "ternary"])
    assert vt.parse_args({"description": "t", "defaults": {}}).quant_mode == "ternary"
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py"])
    assert vt.parse_args({"description": "t", "defaults": {}}).quant_mode == qat.QUANT_MODE_INT8


@pytest.mark.parametrize("mode", qat.QUANT_MODES)
def test_set_quant_mode_reaches_every_quantized_layer(mode):
    """Under QAT the picked rounding applies to all QuantLinear layers, not only the first."""
    m = Veritate(vocab=256, hidden=16, layers=2, ffn=32, heads=2, seq=8)
    qat.set_qat(m, True)
    qat.set_quant_mode(m, mode)
    layers = [x for x in m.modules() if isinstance(x, QuantLinear)]
    assert layers and all(x.quant_mode == mode for x in layers)


def test_an_unknown_quant_mode_is_refused():
    """A typo in the mode is a launch error, not a silent int8 run."""
    with pytest.raises(ValueError, match="unknown quant_mode"):
        qat.set_quant_mode(Veritate(vocab=256, hidden=16, layers=1, ffn=32, heads=2, seq=8), "int3")
