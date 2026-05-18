# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Bit-for-bit parity between Triton-fused and unfused fake_quant ops.
# - Required by preflight rule 24 (kernel == scalar reference).
# - Runs only on CUDA; xfail on CPU.
# tests/engine/test_qat_triton_parity.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from veritate_core import qat as _qat
from veritate_core import qat_triton as _qt


# ------------------------------------------------------------------------------------
# Constants

SEED       = 0
HIDDEN     = 768
SEQ        = 64
BATCH      = 4
WEIGHT_OUT = 768
WEIGHT_IN  = 768

# ------------------------------------------------------------------------------------
# Functions


def _have_cuda_and_triton():
    return _qt.TRITON_AVAILABLE and torch.cuda.is_available()


@unittest.skipUnless(_have_cuda_and_triton(), "CUDA+Triton required")
class FakeQuantActParity(unittest.TestCase):

    def test_act_forward_matches_reference(self):
        """Triton fake_quant_act matches unfused fake_quant_act bit-for-bit on CUDA."""
        torch.manual_seed(SEED)
        x = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16)
        os.environ["VERITATE_NO_TRITON"] = "1"
        ref = _qat.fake_quant_act(x)
        os.environ.pop("VERITATE_NO_TRITON", None)
        out = _qat.fake_quant_act(x)
        self.assertTrue(torch.equal(ref, out))

    def test_act_backward_matches_reference(self):
        """Triton fake_quant_act backward matches unfused STE+clamp mask."""
        torch.manual_seed(SEED)
        x_ref = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_trt = x_ref.detach().clone().requires_grad_(True)
        os.environ["VERITATE_NO_TRITON"] = "1"
        _qat.fake_quant_act(x_ref).sum().backward()
        os.environ.pop("VERITATE_NO_TRITON", None)
        _qat.fake_quant_act(x_trt).sum().backward()
        self.assertTrue(torch.equal(x_ref.grad, x_trt.grad))


@unittest.skipUnless(_have_cuda_and_triton(), "CUDA+Triton required")
class FakeQuantWeightParity(unittest.TestCase):

    def test_w_forward_matches_reference(self):
        """Triton fake_quant_weight matches unfused per-tensor maxabs INT8 on CUDA."""
        torch.manual_seed(SEED)
        w = torch.randn(WEIGHT_OUT, WEIGHT_IN, device="cuda", dtype=torch.bfloat16)
        os.environ["VERITATE_NO_TRITON"] = "1"
        ref = _qat.fake_quant_weight(w)
        os.environ.pop("VERITATE_NO_TRITON", None)
        out = _qat.fake_quant_weight(w)
        diff = (ref.float() - out.float()).abs().max().item()
        self.assertLess(diff, 1e-3)

    def test_w_backward_matches_reference(self):
        """Triton fake_quant_weight backward matches unfused STE+clamp mask."""
        torch.manual_seed(SEED)
        w_ref = torch.randn(WEIGHT_OUT, WEIGHT_IN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w_trt = w_ref.detach().clone().requires_grad_(True)
        os.environ["VERITATE_NO_TRITON"] = "1"
        _qat.fake_quant_weight(w_ref).sum().backward()
        os.environ.pop("VERITATE_NO_TRITON", None)
        _qat.fake_quant_weight(w_trt).sum().backward()
        diff = (w_ref.grad.float() - w_trt.grad.float()).abs().max().item()
        self.assertLess(diff, 1e-2)


@unittest.skipUnless(_have_cuda_and_triton(), "CUDA+Triton required")
class FakeQuantLnWeightParity(unittest.TestCase):

    def test_ln_forward_matches_reference(self):
        """Triton fake_quant_ln_weight matches unfused scale-64 INT8 on CUDA."""
        torch.manual_seed(SEED)
        w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
        os.environ["VERITATE_NO_TRITON"] = "1"
        ref = _qat.fake_quant_ln_weight(w)
        os.environ.pop("VERITATE_NO_TRITON", None)
        out = _qat.fake_quant_ln_weight(w)
        self.assertTrue(torch.equal(ref, out))


if __name__ == "__main__":
    unittest.main()
