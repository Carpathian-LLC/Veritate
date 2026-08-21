# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The proof for checkpoint growth (IDEA 21): a grown checkpoint must compute the
#   SAME function as its source (identical logits on any byte input, fp32 CPU) at
#   the larger shape, along every axis separately and all together, and every
#   parameter of the grown model, new ones included, must receive a finite
#   gradient. heads-only growth at fixed hidden is architecturally impossible
#   (head_dim = hidden/heads would shrink), which the tool must refuse.
# tests/training/test_grow_function_preserving.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "veritate_mri"), os.path.join(REPO, "veritate_mri", "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grow

from veritate_core.model_patched import VeritatePatched

# ------------------------------------------------------------------------------------
# Constants

SEQ, HID, LAYERS, FFN, HEADS = 128, 64, 2, 128, 4
# fp32 CPU: the mapping is exact in real arithmetic; the sqrt write/read scales
# round at ~1e-7 relative per op, so observed deltas sit around 1e-6 on logits
# of order 1. 1e-5 is the spec bound with an order of magnitude of headroom.
TOL = 1e-5

# ------------------------------------------------------------------------------------
# Functions


def _build(layers=LAYERS, hidden=HID, ffn=FFN, heads=HEADS, seq=SEQ, seed=3):
    torch.manual_seed(seed)
    return VeritatePatched(vocab=256, hidden=hidden, layers=layers, ffn=ffn,
                           heads=heads, seq=seq, global_mixer="recurrent").eval()


def _ckpt(tmp_path, model):
    path = str(tmp_path / "step_0.pt")
    torch.save({"model": model.state_dict(), "step": 0,
                "args": {"hidden": HID, "layers": LAYERS, "ffn": FFN, "heads": HEADS}},
               path)
    return path


def _grown(tmp_path, src_path, **target):
    out = str(tmp_path / "grown.pt")
    grow.grow_checkpoint(src_path, out, **target)
    shape = {"layers": LAYERS, "hidden": HID, "ffn": FFN, "heads": HEADS, "seq": SEQ}
    shape.update(target)
    model = _build(**shape)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval(), ckpt


def _inputs(batch=2):
    torch.manual_seed(11)
    return torch.randint(0, 256, (batch, SEQ))


def _text_inputs(length=SEQ):
    """Text-like bytes with a bounded boundary count (spaces every 8 bytes, so
    boundaries stay under the SOURCE slot budget SEQ/4). Random bytes are ~half
    boundaries and overflow the source's slots, where the source model silently
    truncates - outside its representable domain, seq growth legitimately
    differs (previously dropped slots come alive)."""
    row = (b"abcdefg " * (length // 8 + 1))[:length]
    return torch.tensor(list(row), dtype=torch.long).unsqueeze(0).repeat(2, 1)


def _max_delta(src, grown, x):
    with torch.no_grad():
        a, _ = src(x)
        b, _ = grown(x)
    return float((a - b).abs().max())


def _params(model):
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def _assert_axis(tmp_path, **target):
    src = _build()
    grown, _ = _grown(tmp_path, _ckpt(tmp_path, src), **target)
    delta = _max_delta(src, grown, _inputs())
    assert delta < TOL, f"axis {target}: max|dlogits| = {delta:.3e}"
    assert _params(grown) > _params(src)


def test_hidden_growth_preserves_logits(tmp_path):
    """hidden 64 -> 96 at fixed heads (head_dim 16 -> 24): logits unchanged."""
    _assert_axis(tmp_path, hidden=96)


def test_head_growth_preserves_logits(tmp_path):
    """heads 4 -> 8 at fixed head_dim (hidden 64 -> 128, new head slices with
    zeroed output columns): logits unchanged."""
    _assert_axis(tmp_path, hidden=128, heads=8)


def test_ffn_growth_preserves_logits(tmp_path):
    """ffn 128 -> 192 by duplicate-up / split-down: logits unchanged."""
    _assert_axis(tmp_path, ffn=192)


def test_depth_growth_preserves_logits(tmp_path):
    """+1 identity-initialized global block: logits bit-identical (both residual
    writes of the new block are exactly zero)."""
    src = _build()
    grown, _ = _grown(tmp_path, _ckpt(tmp_path, src), layers=LAYERS + 1)
    delta = _max_delta(src, grown, _inputs())
    assert delta == 0.0, f"axis layers: max|dlogits| = {delta:.3e}"
    assert _params(grown) > _params(src)


def test_all_axes_growth_preserves_logits(tmp_path):
    """hidden 64->96, heads 4->6, ffn 128->192, layers 2->3 together: logits
    unchanged and parameter count strictly larger."""
    _assert_axis(tmp_path, hidden=96, heads=6, ffn=192, layers=LAYERS + 1)


def test_grown_model_trains_with_finite_grads(tmp_path):
    """A backward pass through the fully grown model yields a finite gradient for
    every parameter, new ones included (trainability, not just preservation)."""
    src = _build()
    grown, _ = _grown(tmp_path, _ckpt(tmp_path, src),
                      hidden=96, heads=6, ffn=192, layers=LAYERS + 1)
    grown.train()
    x = _inputs()
    torch.manual_seed(12)
    targets = torch.randint(0, 256, x.shape)
    _, loss = grown(x, targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in grown.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_grown_checkpoint_drops_optimizer_and_stamps_args(tmp_path):
    """The grown checkpoint carries no optimizer state and its embedded args
    agree with the grown weights."""
    src = _build()
    path = _ckpt(tmp_path, src)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["optimizer"] = {"state": {}}
    torch.save(ckpt, path)
    _, grown_ckpt = _grown(tmp_path, path, hidden=96)
    assert "optimizer" not in grown_ckpt
    assert grown_ckpt["args"]["hidden"] == 96
    assert grown_ckpt["args"]["layers"] == LAYERS


def test_head_only_growth_is_refused(tmp_path):
    """heads growth at fixed hidden would shrink head_dim and cannot preserve the
    function; the tool refuses instead of guessing."""
    src = _build()
    with pytest.raises(ValueError, match="head"):
        grow.grow_checkpoint(_ckpt(tmp_path, src), str(tmp_path / "bad.pt"), heads=8)


def test_cli_grows_via_main(tmp_path):
    """The python -m training.grow CLI path (including --seq) grows a
    checkpoint end to end."""
    src = _build()
    out = str(tmp_path / "cli.pt")
    grow.main([_ckpt(tmp_path, src), "--ffn", "192", "--seq", "256", "--out", out])
    model = _build(ffn=192, seq=256)
    model.load_state_dict(torch.load(out, map_location="cpu",
                                     weights_only=False)["model"], strict=True)
    assert _max_delta(src, model.eval(), _text_inputs()) < TOL


def test_seq_growth_preserves_logits_on_old_domain(tmp_path):
    """seq 128 -> 256 alone: logits on old-domain inputs (length <= old seq,
    boundary count within the source's slot budget) are bit-identical - only
    the position tables gained rows, and the extra slot rows equal the zero
    padding the source path already computed."""
    src = _build()
    grown, ckpt = _grown(tmp_path, _ckpt(tmp_path, src), seq=256)
    assert ckpt["model"]["pos_emb.weight"].shape[0] == 256
    assert ckpt["model"]["slot_pos_emb.weight"].shape[0] == 64
    # Existing rows exact; new rows copy the last learned row.
    assert torch.equal(ckpt["model"]["pos_emb.weight"][:SEQ],
                       src.state_dict()["pos_emb.weight"])
    assert torch.equal(ckpt["model"]["pos_emb.weight"][SEQ:],
                       src.state_dict()["pos_emb.weight"][-1:].repeat(256 - SEQ, 1))
    delta = _max_delta(src, grown, _text_inputs())
    assert delta == 0.0, f"axis seq: max|dlogits| = {delta:.3e}"
    assert _params(grown) > _params(src)


def test_grown_seq_runs_longer_context(tmp_path):
    """The seq-grown model runs forward+backward at a length beyond the source
    seq with finite loss and finite gradients everywhere (new capacity trains)."""
    src = _build()
    grown, _ = _grown(tmp_path, _ckpt(tmp_path, src), seq=256)
    grown.train()
    torch.manual_seed(13)
    x = torch.randint(0, 256, (2, 256))
    targets = torch.randint(0, 256, (2, 256))
    _, loss = grown(x, targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in grown.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_all_axes_with_seq_preserve_old_domain(tmp_path):
    """Every axis at once including seq: behavior on old-domain inputs is
    preserved within the width-growth rounding tolerance."""
    src = _build()
    grown, _ = _grown(tmp_path, _ckpt(tmp_path, src),
                      hidden=96, heads=6, ffn=192, layers=LAYERS + 1, seq=256)
    delta = _max_delta(src, grown, _text_inputs())
    assert delta < TOL, f"all axes + seq: max|dlogits| = {delta:.3e}"
    assert _params(grown) > _params(src)


def test_seq_shrink_and_stride_are_refused(tmp_path):
    """seq below the source or off the patched trunk's slot stride is refused."""
    src = _build()
    path = _ckpt(tmp_path, src)
    with pytest.raises(ValueError, match="growth only"):
        grow.grow_checkpoint(path, str(tmp_path / "bad.pt"), seq=64)
    with pytest.raises(ValueError, match="stride"):
        grow.grow_checkpoint(path, str(tmp_path / "bad.pt"), seq=130)
