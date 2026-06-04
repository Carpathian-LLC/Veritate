# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - load_from_state_dict must rebuild a canonical model with the activation it was
#   trained with. A ReLU checkpoint loaded as GELU produces wrong outputs.
# tests/plugin_contract/test_load_activation.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.model import Veritate
from veritate_core.load import load_from_state_dict


# ------------------------------------------------------------------------------------
# Functions

def _net(activation):
    return Veritate(vocab=256, hidden=32, layers=2, ffn=48, heads=4, seq=16,
                    activation=activation)


def test_load_respects_relu_activation():
    """A ReLU checkpoint round-trips as ReLU, not the GELU default."""
    net = _net("relu")
    m = load_from_state_dict(net.state_dict(), {"activation": "relu", "heads": 4}, strict_canonical=True)
    assert m.activation == "relu"


def test_load_defaults_to_gelu_when_cfg_silent():
    """Missing activation in cfg falls back to the GELU default (back-compat)."""
    net = _net("gelu")
    m = load_from_state_dict(net.state_dict(), {"heads": 4}, strict_canonical=True)
    assert m.activation == "gelu"


def test_loaded_relu_matches_source_forward():
    """ReLU checkpoint reproduces the source model's logits after load."""
    torch.manual_seed(0)
    net = _net("relu")
    net.eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        ref, _ = net(x)
    m = load_from_state_dict(net.state_dict(), {"activation": "relu", "heads": 4}, strict_canonical=True)
    m.eval()
    with torch.no_grad():
        got, _ = m(x)
    assert torch.allclose(ref, got, atol=1e-6)
