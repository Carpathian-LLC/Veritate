# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The masked-grid objective: what the trainer runs instead of next-byte prediction
#   when a corpus holds encoded images. An image has no causal order, so predicting it
#   left to right wastes the only structural freedom it has. Training a bidirectional
#   model to fill masked positions lets generation run in a handful of parallel
#   refinement passes instead of one forward per byte, which is the difference between
#   a compute-bound and a bandwidth-bound decode on a CPU.
# - Draws are RECORD-ALIGNED, not uniformly random. Every window ends exactly where a
#   record's code block ends, so the image occupies a fixed, known slice of the window
#   and whatever caption bytes fit precede it. A uniform draw would cut images in half.
# - Masking follows the cosine schedule the parallel-decode construction uses: a ratio
#   near 1 teaches the model to start from nothing, a ratio near 0 teaches it to finish
#   an almost complete image, and generation walks that schedule backwards.
# - Loss lands only on masked code positions. Everything else is -1, which the model's
#   cross_entropy already ignores.
# veritate_core/plugin/image_grid.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import numpy as np
import torch

# ------------------------------------------------------------------------------------
# Constants

RECORD_SEP   = b"<|endoftext|>"
IGNORE_INDEX = -1
MIN_MASKED   = 1

# ------------------------------------------------------------------------------------
# Functions


def code_block_ends(bin_path):
    """Offsets where each record's code block ends, i.e. where its separator starts."""
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    head = np.flatnonzero(arr[:len(arr) - len(RECORD_SEP) + 1] == RECORD_SEP[0])
    for i, byte in enumerate(RECORD_SEP[1:], start=1):
        if head.size == 0:
            break
        head = head[arr[head + i] == byte]
    return head


def cosine_mask_ratio(rng, count):
    """MaskGIT's schedule: uniform in the angle, so the run sees every difficulty."""
    return np.cos(rng.uniform(0.0, 1.0, size=count) * (math.pi / 2.0))


def make_record_loader(bin_path, seq, batch_size, code_bytes, mask_byte, seed):
    """Record-aligned masked draws. Returns (draw, usable_records).

    Every window is `seq` bytes ending at a code block's last byte, so positions
    [seq - code_bytes, seq) are always the image and everything before is context."""
    if code_bytes > seq:
        raise ValueError("image_code_bytes " + str(code_bytes) + " exceeds seq " + str(seq)
                         + "; the objective cannot see a whole image")
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    ends = code_block_ends(bin_path)
    ends = ends[ends >= seq]
    if ends.size == 0:
        raise ValueError("no whole record fits seq " + str(seq) + " in " + str(bin_path))
    rng = np.random.RandomState(seed)
    first_code = seq - code_bytes

    def draw():
        picks = ends[rng.randint(0, ends.size, size=batch_size)]
        window = np.empty((batch_size, seq), dtype=np.int64)
        for b, end in enumerate(picks):
            window[b] = arr[end - seq:end]
        tokens = torch.from_numpy(window)
        targets = torch.full_like(tokens, IGNORE_INDEX)
        ratios = cosine_mask_ratio(rng, batch_size)
        for b, ratio in enumerate(ratios):
            n = max(MIN_MASKED, round(ratio * code_bytes))
            where = rng.choice(code_bytes, size=n, replace=False) + first_code
            targets[b, where] = tokens[b, where]
            tokens[b, where] = mask_byte
        return tokens, targets

    return draw, int(ends.size)


def masked_step(model, tokens, targets, amp_dtype, device_type, backward=False):
    """One masked-fill step. The model's own cross_entropy ignores the -1 positions, so
    the objective needs no second loss path. Returns None on a non-finite loss, which is
    the contract the trainer's step skipping already expects."""
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
        _, loss = model(tokens, targets)
    if not torch.isfinite(loss):
        return None
    if backward:
        loss.backward()
    return loss.detach()
