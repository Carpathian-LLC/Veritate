# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - measure_activity must score each layer with the model's own activation, not a
#   hardcoded GELU. A ReLU model must not route through F.gelu.
# tests/mri/test_pruning_activation.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritate_core import model as vmodel
from training import pruning


# ------------------------------------------------------------------------------------
# Functions

def _tiny_corpus(tmp_path):
    bin_path = os.path.join(str(tmp_path), "corpus.bin")
    np.random.RandomState(0).randint(0, 256, size=512, dtype=np.uint8).tofile(bin_path)
    return bin_path


def test_measure_activity_uses_model_activation(tmp_path, monkeypatch):
    """measure_activity on a ReLU model never calls F.gelu."""
    import torch.nn.functional as F
    def _boom(*a, **k):
        raise AssertionError("gelu used on a relu model")
    monkeypatch.setattr(F, "gelu", _boom)

    net = vmodel.Veritate(vocab=256, hidden=32, layers=2, ffn=48, heads=4, seq=32,
                          activation="relu")
    net.eval()
    report = pruning.measure_activity(net, _tiny_corpus(tmp_path), n_samples=2, seq_len=32)
    assert len(report["per_layer"]) == 2
    for e in report["per_layer"]:
        assert 0 <= e["alive"] <= e["total"] == 48
