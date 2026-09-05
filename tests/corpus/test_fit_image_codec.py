# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - F1's driver. What is pinned here is what makes a fit over a real photo library
#   affordable and repeatable: pictures are decoded once, a growing set re-decodes only
#   its new pictures, and an image keeps its side of the train/val split for the life of
#   the set. A positional split silently moves pictures between sides as the library
#   grows, which quietly invalidates every comparison across fits.
# tests/corpus/test_fit_image_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

import numpy as np
import pytest
from PIL import Image
from tools import fit_image_codec

# ------------------------------------------------------------------------------------
# Constants

H = W = 40
PATCH  = 20
PLANES = 2

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Private image, cache and codec roots."""
    images = str(tmp_path / "images")
    cache  = str(tmp_path / "cache")
    codecs = str(tmp_path / "codecs")
    monkeypatch.setattr(fit_image_codec.paths, "image_set_dir",
                        lambda name: os.path.join(images, name))
    monkeypatch.setattr(fit_image_codec.paths, "image_cache_path",
                        lambda name, h, w: os.path.join(cache, f"{name}_{h}x{w}.u8"))
    monkeypatch.setattr(fit_image_codec.paths, "image_cache_index_path",
                        lambda name, h, w: os.path.join(cache, f"{name}_{h}x{w}.index.json"))
    monkeypatch.setattr(fit_image_codec.paths, "codec_path",
                        lambda name: os.path.join(codecs, f"{name}.codec.pt"))
    os.makedirs(os.path.join(images, "set"), exist_ok=True)
    return tmp_path


def _add(home, name, color):
    Image.new("RGB", (H * 2, H * 2), color).save(
        os.path.join(str(home / "images"), "set", name))


def _cache_names(home):
    with open(os.path.join(str(home / "cache"), f"set_{H}x{W}.index.json"),
              encoding="utf-8") as handle:
        return json.load(handle)["names"]


def test_the_cache_holds_one_frame_per_picture_at_the_asked_geometry(home):
    for i in range(3):
        _add(home, f"{i:08x}_pic.png", (10 * i, 20, 30))
    path, names = fit_image_codec.build_cache("set", H, W, verbose=False)
    assert len(names) == 3
    frames = np.memmap(path, dtype=np.uint8, mode="r", shape=(3, H, W, 3))
    assert frames.shape == (3, H, W, 3)
    assert int(frames[0, 0, 0, 1]) == 20


def test_adding_pictures_decodes_only_the_new_ones(home):
    """A library grows; re-decoding all of it every time is the cost this avoids."""
    _add(home, "00000001_a.png", (5, 5, 5))
    fit_image_codec.build_cache("set", H, W, verbose=False)
    decoded = []
    real = fit_image_codec._decode
    fit_image_codec._decode = lambda args: (decoded.append(args[0]), real(args))[1]
    try:
        _add(home, "00000002_b.png", (9, 9, 9))
        _, names = fit_image_codec.build_cache("set", H, W, verbose=False)
    finally:
        fit_image_codec._decode = real
    assert len(names) == 2
    assert len(decoded) == 1
    assert decoded[0].endswith("00000002_b.png")


def test_a_reused_frame_keeps_its_pixels(home):
    """Copying a cached frame forward must not shift or corrupt it."""
    _add(home, "00000001_a.png", (7, 77, 177))
    path, _ = fit_image_codec.build_cache("set", H, W, verbose=False)
    before = np.array(np.memmap(path, dtype=np.uint8, mode="r", shape=(1, H, W, 3))[0])
    _add(home, "00000002_b.png", (1, 2, 3))
    path, names = fit_image_codec.build_cache("set", H, W, verbose=False)
    after = np.memmap(path, dtype=np.uint8, mode="r", shape=(len(names), H, W, 3))
    assert np.array_equal(before, after[names.index("00000001_a.png")])


def test_a_picture_keeps_its_side_of_the_split_as_the_set_grows(home):
    """The split follows the content hash, not the position, so a fit and a corpus
    built weeks apart hold out the same pictures."""
    names = [f"{i:08x}_p.png" for i in range(120)]
    _train, val = fit_image_codec._split(names)
    assert len(val) > 0
    grown = sorted(names + [f"{i:08x}_q.png" for i in range(500, 560)])
    _train2, val2 = fit_image_codec._split(grown)
    was_val = {names[i] for i in val}
    now_val = {grown[i] for i in val2}
    assert was_val <= now_val


def test_val_is_never_empty(home):
    """A tiny set must still report a held-out number rather than dividing by zero."""
    _train, val = fit_image_codec._split(["ffffffff_a.png"])
    assert len(val) == 1


def test_fitting_lowers_held_out_error_and_saves_a_loadable_codec(home):
    """The whole point of F1: a codec that reconstructs, saved where a corpus can
    find it."""
    for i in range(24):
        _add(home, f"{i:08x}_pic.png", (8 * i, 255 - 8 * i, 60))
    rep = fit_image_codec.fit("set", "smoke", height=H, width=W, planes=PLANES,
                              patch=PATCH, latent_dim=8, dec_hidden=32, epochs=3,
                              batch_size=8, lr=5e-3, device="cpu", verbose=False)
    assert rep["image_code_bytes"] == PLANES * (H // PATCH) * (W // PATCH)
    assert rep["history"][-1]["l1"] < rep["history"][0]["l1"]
    from veritate_core.plugin import image_codec
    reloaded = image_codec.load(rep["path"])
    assert reloaded.code_bytes(H, W) == rep["image_code_bytes"]


def test_an_output_scale_caches_and_scores_the_pictures_at_that_size(home):
    """out_scale=2 fits the decoder on the pictures at twice the frame: the cache holds
    2H x 2W frames, the saved codec carries the scale and decodes to it, and the code
    length is still the frame's."""
    for i in range(6):
        _add(home, f"{i:08x}_p.png", (20 * i, 100, 200 - 20 * i))
    rep = fit_image_codec.fit("set", "big", height=H, width=W, planes=PLANES, patch=PATCH,
                              epochs=1, batch_size=2, device="cpu", verbose=False, out_scale=2)
    assert rep["out_scale"] == 2 and (rep["out_height"], rep["out_width"]) == (2 * H, 2 * W)
    assert rep["image_code_bytes"] == PLANES * (H // PATCH) * (W // PATCH)
    assert os.path.isfile(os.path.join(str(home / "cache"), f"set_{2 * H}x{2 * W}.u8"))
    import torch

    codec = fit_image_codec.image_codec.load(fit_image_codec.paths.codec_path("big"))
    assert codec.out_scale == 2
    codes = codec.encode(torch.rand(1, 3, H, W))[0]
    assert codec.decode(codes).shape == (2 * H, 2 * W, 3)


def test_resuming_onto_a_different_geometry_is_refused(home):
    """A corpus is unreadable without the codec that wrote it, so geometry cannot drift."""
    for i in range(8):
        _add(home, f"{i:08x}_pic.png", (i, i, i))
    fit_image_codec.fit("set", "smoke", height=H, width=W, planes=PLANES, patch=PATCH,
                        latent_dim=8, dec_hidden=32, epochs=1, batch_size=4,
                        device="cpu", verbose=False)
    with pytest.raises(ValueError, match="geometry"):
        fit_image_codec.fit("set", "smoke", height=H, width=W, planes=PLANES + 1,
                            patch=PATCH, latent_dim=8, dec_hidden=32, epochs=1,
                            batch_size=4, device="cpu", resume=True, verbose=False)


def test_a_cache_that_would_not_fit_on_the_disk_is_refused_before_it_starts(home, monkeypatch):
    """1920x1080 over a phone library is 869 GB. That is an error message, not a full
    disk an hour later; and nothing is left behind."""
    import collections
    import shutil as _shutil
    _add(home, "00000001_a.png", (5, 5, 5))
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: usage(100, 90, 1000))
    with pytest.raises(ValueError, match=r"would need .* GB but only"):
        fit_image_codec.build_cache("set", H, W, verbose=False)
    assert not os.listdir(str(home / "cache"))


def test_the_codec_is_fitted_on_a_capped_sample_of_the_set(home):
    """A codec does not improve past ~10k pictures; the sample keeps a 140k-picture
    library from costing a 40 GB cache before the model sees a step."""
    for i in range(6):
        _add(home, f"{i:08x}_pic.png", (10 * i, 20, 30))
    rep = fit_image_codec.fit("set", "capped", height=H, width=W, planes=PLANES, patch=PATCH,
                              latent_dim=8, dec_hidden=32, epochs=1, batch_size=2,
                              device="cpu", limit=4, verbose=False)
    assert rep["images"] == 4
    assert len(_cache_names(home)) == 4


def test_exif_orientation_is_applied_so_portrait_shots_are_upright(home):
    """Phone JPEGs are stored sideways with an orientation tag; the decode must honour
    it or every portrait picture trains rotated."""
    from PIL import Image as _Image
    img = _Image.new("RGB", (H * 2, H * 4), (0, 0, 0))
    for y in range(H * 2):                       # top half red, bottom half blue
        for x in range(H * 2):
            img.putpixel((x, y), (255, 0, 0))
    for y in range(H * 2, H * 4):
        for x in range(H * 2):
            img.putpixel((x, y), (0, 0, 255))
    exif = _Image.Exif()
    exif[0x0112] = 6                             # rotate 90 CW on display
    path = os.path.join(str(home / "images"), "set", "00000003_p.jpg")
    img.save(path, exif=exif, quality=95)
    frame = fit_image_codec._decode((path, H, W))
    assert frame.shape == (H, W, 3)
    # after a 90 CW rotation the red half is on the RIGHT and the blue half on the left
    assert int(frame[H // 2, W - 2, 0]) > 200 and int(frame[H // 2, 1, 2]) > 200


def test_a_stop_request_during_decode_leaves_no_half_cache(home):
    from veritate_core.plugin.image_progress import StopRequested
    for i in range(3):
        _add(home, f"{i:08x}_pic.png", (10 * i, 20, 30))
    fit_image_codec.PROGRESS_EVERY, saved = 1, fit_image_codec.PROGRESS_EVERY
    try:
        with pytest.raises(StopRequested):
            fit_image_codec.build_cache("set", H, W, verbose=False, should_stop=lambda: True)
    finally:
        fit_image_codec.PROGRESS_EVERY = saved
    assert not [f for f in os.listdir(str(home / "cache")) if f.endswith(".tmp")]
    assert not os.path.isfile(os.path.join(str(home / "cache"), f"set_{H}x{W}.u8"))
