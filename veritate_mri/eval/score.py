# ------------------------------------------------------------------------------------
# veritate_mri/eval/score.py
# ------------------------------------------------------------------------------------
# Byte-level sequence scorer (production copy used by the MRI dashboard's deep-eval
# panel; mirrors experiments/v2/eval_harness/score.py).
#
# `score_sequence(model, prompt_bytes, completion_bytes)` returns the mean per-byte
# log-likelihood of `completion_bytes` conditioned on `prompt_bytes`. Used by every
# multiple-choice suite (MMLU, HellaSwag, etc.) to rank candidate answers.
#
# Algorithm:
#   1. Concatenate prompt + completion -> ids of length P + C.
#   2. Run the model on the full sequence.
#      Logit at position t predicts byte t+1.
#   3. Take logits at positions P-1 .. P+C-2  (these predict bytes P .. P+C-1).
#   4. log-softmax, gather at the target byte indices, sum.
#   5. Divide by C (mean per-byte log-likelihood, units: nats/byte).
#
# Length-normalization is critical: longer candidates would otherwise be penalized.
# This matches the standard `acc_norm` metric in lm-eval-harness.
#
# Inputs are raw `bytes`; each byte is a vocab id in [0, 256).
# Outputs are Python floats (nats per byte; higher = more likely under the model).
# ------------------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F


def _bytes_to_ids(b: bytes) -> torch.Tensor:
    return torch.tensor(list(b), dtype=torch.long)


def _model_device(model) -> torch.device:
    """Best-effort device sniff. Falls back to CPU if the model has no params."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def score_sequence(model, prompt_bytes: bytes, completion_bytes: bytes,
                   device: str | torch.device | None = None) -> float:
    """Per-byte log-likelihood of `completion_bytes` given `prompt_bytes`.

    Returns a float in nats/byte (more positive = more likely). For a uniform
    256-byte distribution, the floor is -ln(256) ~= -5.545 nats/byte.

    `device` defaults to wherever the model's parameters already live, so the
    same call works for CPU and MPS without the caller having to track it.
    """
    if not isinstance(prompt_bytes, (bytes, bytearray)):
        raise TypeError(f"prompt_bytes must be bytes, got {type(prompt_bytes)}")
    if not isinstance(completion_bytes, (bytes, bytearray)):
        raise TypeError(f"completion_bytes must be bytes, got {type(completion_bytes)}")
    if len(completion_bytes) == 0:
        raise ValueError("completion_bytes must be non-empty")

    P = len(prompt_bytes)
    C = len(completion_bytes)

    # If the prompt is empty we still need at least one context byte for the first
    # completion token to be predicted from. Convention: prepend a zero byte (NUL)
    # as a BOS-like anchor; the per-byte loss on the very first completion byte
    # will be conditioned on NUL — a fine baseline for a byte-level LM.
    if P == 0:
        prompt_bytes = b"\x00"
        P = 1

    if device is None:
        device = _model_device(model)

    ids = _bytes_to_ids(bytes(prompt_bytes) + bytes(completion_bytes)).unsqueeze(0)  # (1, P+C)
    ids = ids.to(device)

    # Respect the model's max sequence length. If we'd overflow, truncate from the
    # LEFT of the prompt (we always need to keep the full completion + at least
    # one byte of context).
    max_seq = getattr(model, "seq", None) or ids.size(1)
    if ids.size(1) > max_seq:
        keep_prompt = max_seq - C
        if keep_prompt < 1:
            raise ValueError(
                f"completion length {C} >= model.seq {max_seq}; can't score."
            )
        ids = ids[:, -max_seq:]
        P = keep_prompt  # new effective prompt length after truncation

    model.eval()
    with torch.no_grad():
        out = model(ids)
        # Veritate models return (logits, loss); also accept a bare tensor.
        logits = out[0] if isinstance(out, (tuple, list)) else out  # (1, T, V)

    # logits[:, t] predicts ids[:, t+1].
    # The C completion bytes live at positions P .. P+C-1 in `ids`, so they are
    # predicted by logits at positions P-1 .. P+C-2.
    pred_logits = logits[0, P - 1 : P - 1 + C, :]                # (C, V)
    targets     = ids[0, P : P + C]                              # (C,)
    log_probs   = F.log_softmax(pred_logits.float(), dim=-1)     # (C, V)
    gathered    = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (C,)
    return float(gathered.sum().item() / C)
