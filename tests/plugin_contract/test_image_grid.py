# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The masked-grid objective. Pins the property the whole lever rests on: every draw
#   is record-aligned, so loss lands only on image bytes and never on a half image.
#   A uniform draw (what the next-byte loader does) would cut records apart.
# tests/plugin_contract/test_image_grid.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
import pytest
import torch

from veritate_core.plugin import image_grid

# ------------------------------------------------------------------------------------
# Constants

CODE_BYTES = 12
MASK_BYTE  = 255
SEQ        = 32
BATCH      = 4
RECORDS    = 40

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.RandomState(0)
    path = tmp_path / "img_train.bin"
    with open(path, "wb") as handle:
        for i in range(RECORDS):
            codes = bytes(rng.randint(0, MASK_BYTE, size=CODE_BYTES, dtype=np.uint8).tolist())
            handle.write(b"caption %d" % i + codes + image_grid.RECORD_SEP)
    return str(path)


def test_every_record_boundary_is_found(corpus):
    """The separator scan is what makes alignment possible at all."""
    assert image_grid.code_block_ends(corpus).size == RECORDS


def test_loss_lands_only_on_image_bytes(corpus):
    """Windows end at a code block, so supervised positions are always the image."""
    draw, _ = image_grid.make_record_loader(corpus, SEQ, BATCH, CODE_BYTES, MASK_BYTE, seed=0)
    _tokens, targets = draw()
    supervised = (targets != image_grid.IGNORE_INDEX).nonzero()[:, 1]
    assert int(supervised.min()) >= SEQ - CODE_BYTES


def test_masked_positions_carry_the_mask_byte(corpus):
    """The model must see a value that cannot be a real code where it has to predict."""
    draw, _ = image_grid.make_record_loader(corpus, SEQ, BATCH, CODE_BYTES, MASK_BYTE, seed=0)
    tokens, targets = draw()
    masked = targets != image_grid.IGNORE_INDEX
    assert bool((tokens[masked] == MASK_BYTE).all())
    assert masked.any(dim=1).all()


def test_a_seed_replays_the_same_batch(corpus):
    """Two runs at one seed train on identical data."""
    batches = [image_grid.make_record_loader(corpus, SEQ, BATCH, CODE_BYTES, MASK_BYTE, seed=3)[0]()
               for _ in range(2)]
    assert torch.equal(batches[0][0], batches[1][0])
    assert torch.equal(batches[0][1], batches[1][1])


def test_an_image_wider_than_the_window_is_refused(corpus):
    """Training on a window that cannot hold a whole image is silently wrong, so it raises."""
    with pytest.raises(ValueError, match="exceeds seq"):
        image_grid.make_record_loader(corpus, CODE_BYTES - 1, BATCH, CODE_BYTES, MASK_BYTE, seed=0)


def test_the_mask_schedule_stays_a_ratio():
    """The cosine schedule must span easy and hard fills without leaving [0, 1]."""
    ratios = image_grid.cosine_mask_ratio(np.random.RandomState(0), 512)
    assert ratios.min() >= 0.0 and ratios.max() <= 1.0
    assert ratios.min() < 0.2 and ratios.max() > 0.8
