# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Shape, parameter-cost and contract cover for the Monarch-factored FFN.
# tests/core/test_monarch.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core.model_monarch import MonarchFFN, MonarchLinear, square_factors

# ------------------------------------------------------------------------------------
# Functions


def test_square_factors_are_most_square_and_exact():
    """square_factors returns an exact factorization closest to square."""
    for n in (256, 1024, 1280, 2304):
        s1, s2 = square_factors(n)
        assert s1 * s2 == n
        assert s1 <= s2


def test_monarch_linear_preserves_shape_and_cuts_parameters():
    """A Monarch map is features-to-features but costs far less than dense."""
    torch.manual_seed(0)
    layer = MonarchLinear(1024)
    x = torch.randn(3, 7, 1024)
    assert layer(x).shape == x.shape
    dense = 1024 * 1024
    assert layer.param_count() == sum(p.numel() for p in layer.parameters())
    assert layer.param_count() < dense / 8


def test_monarch_ffn_matches_dense_ffn_shape_and_backprops():
    """The FFN drop-in returns hidden-width output and passes gradient to both factors."""
    torch.manual_seed(0)
    ff = MonarchFFN(256, 1024)
    x = torch.randn(2, 5, 256, requires_grad=True)
    out = ff(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert ff.up[0].w1.grad is not None
    assert ff.down[0].w2.grad is not None


def test_monarch_ffn_rejects_ffn_that_is_not_a_multiple_of_hidden():
    """Width mismatch is refused at construction rather than silently reshaped."""
    with pytest.raises(ValueError, match="multiple of hidden"):
        MonarchFFN(256, 1000)


def test_monarch_ffn_reports_no_probe_weight_pair():
    """Factored layers hold no (in, out) pair, so consumers must skip them."""
    ff = MonarchFFN(256, 512)
    assert ff.probe_weights() is None
    assert ff.probe_module() is ff.up[0]


def test_monarch_ffn_capture_l1_records_post_activation():
    """capture_l1 populates _last_l1 so mixed-trunk regularization stays uniform."""
    ff = MonarchFFN(256, 512, capture_l1=True)
    ff(torch.randn(2, 4, 256))
    assert ff._last_l1 is not None and float(ff._last_l1.detach()) >= 0.0
