# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IDEA 24's architectural rule, pinned: no tensor whose extent is the output
#   resolution may be materialized. The tiled arms must hold a fixed working set as
#   the frame grows; the conv_full control must scale with area, since it exists to
#   be the thing that fails.
# - Random weights only. No corpus, no checkpoint, no device beyond CPU.
# tests/plugin_contract/test_image_decode.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core.plugin import image_decode

# ------------------------------------------------------------------------------------
# Constants

SMALL = (120, 160)
BIG   = (240, 320)   # 4x the area of SMALL, still divisible by patch, grid_div and the conv stack
TILED = ("coord", "patch")

# ------------------------------------------------------------------------------------
# Functions


def _peaks(height, width):
    report = image_decode.bench(height, width, reps=1, warmup=0)
    return {arm: v["peak_activation_bytes"] for arm, v in report["arms"].items()}


def test_tiled_arms_hold_their_working_set_while_the_control_scales_with_area():
    """Quadrupling the frame leaves the tiled arms' peak tensor unchanged and grows the
    conv control's in proportion to pixels."""
    small = _peaks(*SMALL)
    big   = _peaks(*BIG)
    for arm in TILED:
        assert big[arm] == small[arm]
    assert big["conv_full"] >= small["conv_full"] * 3.5


def test_patch_arm_costs_an_order_of_magnitude_fewer_flops_than_the_coord_arm():
    """The dictionary path does no multiplies, so only its residual is billed."""
    arms = image_decode.bench(*BIG, reps=1, warmup=0)["arms"]
    assert arms["coord"]["gflop"] >= arms["patch"]["gflop"] * 10


def test_every_arm_decodes_a_full_uint8_frame():
    """Each decoder returns the whole frame as bytes, not a float feature map."""
    height, width = SMALL
    torch.manual_seed(0)
    latent = torch.randn((1, image_decode.DEFAULT_LATENT_CH,
                          height // image_decode.DEFAULT_GRID_DIV,
                          width // image_decode.DEFAULT_GRID_DIV))
    codes = torch.randint(0, image_decode.VOCAB_BYTE_LEVEL,
                          (height // image_decode.DEFAULT_PATCH, width // image_decode.DEFAULT_PATCH))
    frames = [
        image_decode.CoordDecoder(image_decode.DEFAULT_LATENT_CH, image_decode.DEFAULT_MLP_WIDTH,
                                  image_decode.DEFAULT_TILE).render(latent, height, width),
        image_decode.BytePatchDecoder(image_decode.DEFAULT_PATCH, image_decode.DEFAULT_CODE_EMB,
                                      image_decode.DEFAULT_PATCH_HIDDEN,
                                      image_decode.DEFAULT_BAND).render(codes),
        image_decode.ConvDecoder(image_decode.DEFAULT_LATENT_CH,
                                 image_decode.DEFAULT_CONV_CH).render(latent, height, width),
    ]
    for frame in frames:
        assert frame.shape == (height, width, image_decode.RGB)
        assert frame.dtype == torch.uint8


def test_bench_refuses_a_frame_the_patch_grid_cannot_tile():
    """Geometry that does not divide is a caller error, not a silently cropped frame."""
    with pytest.raises(ValueError, match="does not divide"):
        image_decode.bench(121, 160, reps=1, warmup=0)


def test_bench_refuses_an_unknown_arm():
    """An arm name typo names the valid set rather than decoding nothing."""
    with pytest.raises(ValueError, match="unknown arm"):
        image_decode.bench(*SMALL, arms=("difusion",), reps=1, warmup=0)


def test_one_seed_decodes_identical_pixels():
    """Decoding is a pure function of weights and codes, so F0 reruns are comparable."""
    height, width = SMALL
    codes = torch.randint(0, image_decode.VOCAB_BYTE_LEVEL,
                          (height // image_decode.DEFAULT_PATCH, width // image_decode.DEFAULT_PATCH))
    frames = []
    for _ in range(2):
        torch.manual_seed(7)
        decoder = image_decode.BytePatchDecoder(image_decode.DEFAULT_PATCH, image_decode.DEFAULT_CODE_EMB,
                                                image_decode.DEFAULT_PATCH_HIDDEN, image_decode.DEFAULT_BAND)
        frames.append(decoder.render(codes))
    assert torch.equal(frames[0], frames[1])
