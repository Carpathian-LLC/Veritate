# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Inference-side abstention backstop. The primary IDK signal comes from
#   in-pretrain dosing of `idk_abstention` pairs (developer_documentation/training/
#   settings_index.md perfect-chat recipe). This gate is the safety net: when the
#   model tries to answer confidently but its own top-1 probability across the
#   first N decoded bytes is low, the response is replaced with a fixed template.
# - Two paths:
#   1. Streaming: AbstentionGate() consumes (byte, top1_prob) events. After
#      warmup_bytes, it locks a decision. Callers check gate.decision() and either
#      emit the buffered original bytes (proceed) or emit the template (abstain).
#   2. Post-hoc: apply_post_hoc(response_bytes, top1_probs) makes the same call
#      offline, for callers that generate then decide (non-streaming servers).
# - Not wired into any backend yet. Wire-in point: veritate_mri/inference/backends/
#   pytorch.py:stream() (per-byte top-1 already computed) once a trained 10M or
#   80M exists and threshold has been tuned on held-out prompts.
# - The template is the SAME phrase used in the SFT idk_abstention family so a
#   trained model's native IDK output and the gate's replacement look identical.
# veritate_mri/inference/decode/abstention.py
# ------------------------------------------------------------------------------------
# Imports

import os
from collections.abc import Iterable
from dataclasses import dataclass

# ------------------------------------------------------------------------------------
# Constants

ABSTAIN_TEMPLATE = b"I don't know that answer."

# Default threshold: mean top-1 probability across the first `warmup_bytes`
# decoded bytes. Below this, the model is guessing; the gate fires. Tune on
# 50 known-good + 50 known-bad prompts once a trained model exists.
DEFAULT_TOP1_MEAN_THRESHOLD = 0.20
DEFAULT_WARMUP_BYTES        = 12

# Post-hoc heuristic thresholds (used when no per-byte telemetry is available).
# A response shorter than this many characters that isn't already an IDK template
# is treated as a degenerate hallucination. Repetition ratio: unique bytes /
# total bytes; below this the response is in a repetition loop (rule 24d.pit).
POSTHOC_MIN_LENGTH_CHARS         = 4
POSTHOC_REPETITION_RATIO_FLOOR   = 0.20
POSTHOC_LOOKAHEAD_BYTES          = 32

DECISION_UNDECIDED = "undecided"
DECISION_PROCEED   = "proceed"
DECISION_ABSTAIN   = "abstain"


# ------------------------------------------------------------------------------------
# Config

@dataclass
class AbstentionConfig:
    top1_mean_threshold: float = DEFAULT_TOP1_MEAN_THRESHOLD
    warmup_bytes:        int   = DEFAULT_WARMUP_BYTES
    template:            bytes = ABSTAIN_TEMPLATE
    enabled:             bool  = True


# ------------------------------------------------------------------------------------
# Streaming gate

class AbstentionGate:
    """Per-response streaming gate. Feed each (byte, top1_prob) event via
    observe(); after warmup_bytes the gate's decision() is locked. Callers hold
    an internal buffer of the first warmup_bytes bytes until decision() != undecided,
    then either flush the buffer (proceed) or replace with config.template (abstain)."""

    def __init__(self, config: AbstentionConfig | None = None):
        self.config = config or AbstentionConfig()
        self._top1_probs: list[float] = []
        self._buffer:     bytearray  = bytearray()
        self._decision:   str        = DECISION_UNDECIDED

    def observe(self, byte: int, top1_prob: float) -> None:
        if not self.config.enabled or self._decision != DECISION_UNDECIDED:
            return
        self._buffer.append(byte & 0xFF)
        self._top1_probs.append(float(top1_prob))
        if len(self._top1_probs) >= self.config.warmup_bytes:
            mean = sum(self._top1_probs) / len(self._top1_probs)
            self._decision = (DECISION_ABSTAIN
                              if mean < self.config.top1_mean_threshold
                              else DECISION_PROCEED)

    def decision(self) -> str:
        return self._decision

    def buffered_bytes(self) -> bytes:
        return bytes(self._buffer)

    def abstention_bytes(self) -> bytes:
        return self.config.template

    def force_decision(self) -> None:
        """Called by the caller if generation ends before warmup completes. Picks
        proceed if any bytes made it through (a short natural response), abstain
        if not (empty output = degenerate)."""
        if self._decision != DECISION_UNDECIDED:
            return
        self._decision = DECISION_PROCEED if self._top1_probs else DECISION_ABSTAIN


# ------------------------------------------------------------------------------------
# Post-hoc backstop (no per-byte telemetry available)

def apply_post_hoc(response_bytes: bytes,
                   top1_probs: Iterable[float] | None = None,
                   config: AbstentionConfig | None = None) -> bytes:
    """Offline decision on a completed response. Prefers per-byte telemetry when
    the caller can supply it (top1_probs of the first warmup_bytes bytes);
    otherwise falls back to two heuristics: response length + repetition ratio
    on the first POSTHOC_LOOKAHEAD_BYTES bytes."""
    cfg = config or AbstentionConfig()
    if not cfg.enabled:
        return response_bytes

    if top1_probs is not None:
        probs = list(top1_probs)[: cfg.warmup_bytes]
        if probs:
            mean = sum(probs) / len(probs)
            if mean < cfg.top1_mean_threshold:
                return cfg.template
        return response_bytes

    trimmed = response_bytes.strip()
    if len(trimmed) < POSTHOC_MIN_LENGTH_CHARS:
        return cfg.template
    head = trimmed[:POSTHOC_LOOKAHEAD_BYTES]
    if head:
        ratio = len(set(head)) / len(head)
        if ratio < POSTHOC_REPETITION_RATIO_FLOOR:
            return cfg.template
    return response_bytes


# ------------------------------------------------------------------------------------
# Convenience

def make_gate(threshold: float = DEFAULT_TOP1_MEAN_THRESHOLD,
              warmup: int = DEFAULT_WARMUP_BYTES,
              enabled: bool = True) -> AbstentionGate:
    return AbstentionGate(AbstentionConfig(top1_mean_threshold=threshold,
                                           warmup_bytes=warmup,
                                           enabled=enabled))


# ------------------------------------------------------------------------------------
# Env-gated wrapper for the /v1/chat/completions local generation path.
# When VERITATE_ABSTENTION_GATE is truthy, wrap_response_text applies the post-hoc
# heuristic to a generated response and replaces degenerate output (repetition-loop
# only; short valid answers like "8." are preserved by disabling the length rule).

ENV_FLAG_NAME = "VERITATE_ABSTENTION_GATE"


def _env_flag_on() -> bool:
    v = os.environ.get(ENV_FLAG_NAME, "").strip().lower()
    return v in ("1", "true", "on", "yes")


def wrap_response_text(text: str) -> str:
    """Backstop for the local /v1/chat/completions path. Off by default; opt-in via
    VERITATE_ABSTENTION_GATE=on. Only fires on repetition-loop responses, never on
    short-but-valid answers (math replies, one-word acknowledgements). Meant to
    catch the residual ~10% of hallucinations the SFT dose missed on trained models."""
    if not _env_flag_on():
        return text
    if text is None:
        return text
    raw = text.encode("utf-8", errors="replace")
    trimmed = raw.strip()
    if not trimmed:
        return text
    head = trimmed[:POSTHOC_LOOKAHEAD_BYTES]
    if head:
        ratio = len(set(head)) / len(head)
        if ratio < POSTHOC_REPETITION_RATIO_FLOOR:
            return ABSTAIN_TEMPLATE.decode("utf-8")
    return text
