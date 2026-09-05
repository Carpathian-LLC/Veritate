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
# - A record closer to the start of its bin than `seq` bytes has less history than the
#   window; its window is LEFT-PADDED with PAD_BYTE so the picture still trains. The
#   alternative -- dropping it -- silently loses the first pictures of every corpus and
#   makes a small val bin unusable outright.
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
# Fills the front of a window whose record has less history than seq. Context only:
# the image occupies a fixed slice at the end of the window and is never padded.
PAD_BYTE     = 0

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


def make_record_loader(bin_path, seq, batch_size, code_bytes, mask_byte, seed, caption_dropout=0.0):
    """Record-aligned masked draws. Returns (draw, usable_records).

    Every window is `seq` bytes ending at a code block's last byte, so positions
    [seq - code_bytes, seq) are always the image and everything before is context. The
    context is the record's OWN bytes only -- the separator that opens it and its caption;
    whatever the bin holds before that (the previous picture's code tail) is replaced with
    PAD_BYTE, which is exactly the window generation builds (image_sample.build_window), so
    the model trains on what it will be given. A record with less than `seq` bytes of
    history is left-padded the same way, so every record in the bin is a training example.
    `caption_dropout` is the share of draws whose whole context is padded: the model learns
    to draw from nothing too, which is what a caption-free generation asks for and what
    classifier-free guidance needs."""
    if code_bytes > seq:
        raise ValueError("image_code_bytes " + str(code_bytes) + " exceeds seq " + str(seq)
                         + "; the objective cannot see a whole image")
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    all_ends = code_block_ends(bin_path)
    ends = all_ends[all_ends >= code_bytes]
    if ends.size == 0:
        raise ValueError("no image records in " + str(bin_path))
    rng = np.random.RandomState(seed)
    first_code = seq - code_bytes
    p_drop = float(caption_dropout or 0.0)

    def draw():
        picks = ends[rng.randint(0, ends.size, size=batch_size)]
        drops = rng.uniform(size=batch_size) < p_drop if p_drop > 0 else np.zeros(batch_size, dtype=bool)
        window = np.full((batch_size, seq), PAD_BYTE, dtype=np.int64)
        for b, end in enumerate(picks):
            end = int(end)
            i = int(np.searchsorted(all_ends, end))
            # the previous record's separator opens this one; nothing before it is ours
            own_start = int(all_ends[i - 1]) if i > 0 else 0
            start = max(own_start, end - seq)
            if drops[b]:
                start = end - code_bytes
            window[b, seq - (end - start):] = arr[start:end]
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


def masked_step(model, tokens, targets, amp_dtype, device_type, backward=False, scale=1.0, scaler=None):
    """One masked-fill step. The model's own cross_entropy ignores the -1 positions, so
    the objective needs no second loss path. Returns None on a non-finite loss, which is
    the contract the trainer's step skipping already expects. `scale` multiplies the loss
    before backward (1/accum under gradient accumulation); a `scaler` (torch.amp.GradScaler,
    fp16 autocast) scales it again so small gradients survive the half format. The
    returned loss is unscaled either way."""
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
        _, loss = model(tokens, targets)
    if not torch.isfinite(loss):
        return None
    if backward:
        scaled = loss * scale if scale != 1.0 else loss
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
    return loss.detach()
