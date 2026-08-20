# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - In-context RAG injection: retrieve passages, put them in front of the user's
#   question inside the same turn, let attention decide what to use. The right
#   mechanism for byte-level models per the sprint-2 survey, and how published
#   1B-class SLM-RAG works.
# - This is the OPPOSITE of the F09 corpus-echo failure. F09 mixed an n-gram
#   retrieval distribution into the model's logits at alpha=1.5 and cost the 85M
#   +36% val NLL. That falsified logit fusion, not byte-level RAG.
# - Framing is ChatML because the chat SFT is. `loss_mask=assistant` grades only
#   the bytes after <|im_start|>assistant\n, so that marker is the sole signal
#   meaning "answer now, stop at <|im_end|>". Injecting a bare Context/Question/
#   Answer prefix instead drops the model into completion mode, where it continues
#   the document rather than replying, and the measured grounded rate does not
#   describe that path.
# - One passage by default. Multi-candidate injection is falsified at this scale
#   (failures.md: top-1 0.130, top-3 0.120, top-5 0.080) because a 200M reader
#   cannot disambiguate candidates. Raise k only behind a re-ranker collapsing k
#   to 1.
# - The LongLLMLingua-style compression hook is where a compressor model plugs in;
#   the published technique compresses retrieved context 4-6x at +21pt accuracy.
# veritate_mri/inference/agent/rag.py
# ------------------------------------------------------------------------------------
# Imports:

import re
from collections.abc import Callable

from .loop import IM_END, IM_START, ROLE_ASSISTANT, ROLE_USER

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_MAX_PASSAGES_B = 4096   # cap on concatenated context bytes
PASSAGE_SEPARATOR      = "\n\n"
TRUNCATION_MARK        = "... [truncated]"
TRUNCATION_FLOOR       = 32     # below this a truncated tail carries nothing; drop it
REPLY_RESERVE_B        = 256    # window bytes held back for the frame and the reply;
                                # 1024B window capped at 768B of passages measured
                                # end-to-end 0.270 vs 0.162 uncapped (2026-08-17)

USER_OPEN      = f"{IM_START}{ROLE_USER}\n"
ASSISTANT_OPEN = f"{IM_START}{ROLE_ASSISTANT}\n"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE           = re.compile(r"\S+")

# ------------------------------------------------------------------------------------
# Functions


def _passage_body(passages: list[str], max_bytes: int,
                  compressor: Callable[[str], str] | None) -> str:
    """Concatenated passage text capped at `max_bytes`. The passage that crosses the
    cap is truncated rather than dropped, so one oversized chunk still grounds."""
    chunks: list[str] = []
    total = 0
    for p in passages:
        if compressor is not None:
            p = compressor(p)
        cost = len(p.encode("utf-8"))
        if total + cost > max_bytes:
            remaining = max_bytes - total
            if remaining > TRUNCATION_FLOOR:
                chunks.append(p[:remaining] + TRUNCATION_MARK)
            break
        chunks.append(p)
        total += cost
    return PASSAGE_SEPARATOR.join(chunks)


def injection_budget(seq: int, prompt_bytes: int,
                     reserve: int = REPLY_RESERVE_B,
                     cap: int = DEFAULT_MAX_PASSAGES_B) -> int:
    """Passage-byte budget that keeps prompt + passages + reply inside a `seq`-byte
    window. Injecting past the window evicts the user-turn opener and the passage
    head before the model answers, which reads as a retrieval miss end-to-end.
    Zero (inject nothing) when the prompt alone fills the window."""
    return max(0, min(cap, seq - reserve - prompt_bytes))


def build_rag_prefix(prompt: str, passages: list[str],
                     max_bytes: int = DEFAULT_MAX_PASSAGES_B,
                     compressor: Callable[[str], str] | None = None) -> str:
    """Ground `prompt` by opening its user turn with the retrieved passages.

    `prompt` arrives either as a raw user message or already ChatML-framed by the
    client; both return one ChatML prompt ending at the assistant turn opening.
    Passages go INSIDE the final user turn, ahead of the question, which is the
    shape the grounded A/B measured. Retrieving nothing returns `prompt` untouched
    rather than an empty context block.
    """
    body = _passage_body(passages, max_bytes, compressor)
    if not body:
        return prompt
    cut = prompt.rfind(USER_OPEN)
    if cut >= 0:
        head = cut + len(USER_OPEN)
        return prompt[:head] + body + PASSAGE_SEPARATOR + prompt[head:]
    return USER_OPEN + body + PASSAGE_SEPARATOR + prompt + IM_END + "\n" + ASSISTANT_OPEN


def make_word_ppl_compressor(brain, keep_frac: float = 0.5, max_ctx_bytes: int = 1024):
    """Word-level perplexity compressor (I53 / S60). Uses a Brain-shaped
    backend as the byte-level scorer: per-byte NLL of the passage is
    aggregated to word boundaries by mean; words with the lowest mean
    NLL (most predictable from preceding bytes, least informative) are
    dropped. Single-space joiner between kept words.
    `keep_frac` in (0, 1] is the fraction of words to retain.
    `max_ctx_bytes` caps scoring cost; longer passages are truncated.
    Smoke (85M, S60): word-level @ keep=0.5 ~ 2x compression for
    +0.19 nll on held-out continuations; @ keep=0.25 ~ 4.3x for +0.27.
    Output is human-readable, unlike per-byte deletion."""
    import torch
    import torch.nn.functional as F
    _BYTE_WORD_RE = re.compile(rb"\S+|\s+")
    keep_frac = max(1e-3, min(1.0, float(keep_frac)))
    max_ctx_bytes = max(64, int(max_ctx_bytes))

    device = next(brain.model.parameters()).device

    @torch.no_grad()
    def _score_bytes(b: bytes):
        tokens = torch.tensor(list(b), dtype=torch.long,
                              device=device).unsqueeze(0)
        # Respect the model's seq cap so we never blow past pos_emb length.
        cap = getattr(brain.model, "seq", tokens.size(1))
        if tokens.size(1) > cap:
            tokens = tokens[:, -cap:]
        logits = brain.model(tokens)[0]
        logp = F.log_softmax(logits[0], dim=-1)
        nll = torch.zeros(tokens.size(1), device=tokens.device)
        targets = tokens[0]
        nll[1:] = -logp[:-1].gather(1, targets[1:].unsqueeze(1)).squeeze(1)
        if tokens.size(1) > 1:
            nll[0] = nll[1:].mean()
        return nll.tolist()

    def _compress(passage: str) -> str:
        if keep_frac >= 1.0 or not passage:
            return passage
        b = passage.encode("utf-8")
        if len(b) > max_ctx_bytes:
            b = b[:max_ctx_bytes]
        try:
            nlls = _score_bytes(b)
        except Exception as e:
            from runtime import logs as logmod
            logmod.warn("rag", f"compressor scoring failed; passing through: {type(e).__name__}: {e}")
            return passage
        spans = [(m.start(), m.end()) for m in _BYTE_WORD_RE.finditer(b)
                 if not m.group(0).isspace()]
        if not spans:
            return passage
        scores = []
        for (s, e) in spans:
            seg = nlls[s:e]
            scores.append(sum(seg) / max(1, len(seg)))
        keep_n = max(1, round(len(spans) * keep_frac))
        order = sorted(range(len(spans)), key=lambda i: scores[i], reverse=True)
        keep = set(order[:keep_n])
        out = []
        for i, (s, e) in enumerate(spans):
            if i in keep:
                out.append(b[s:e].decode("utf-8", errors="replace"))
        return " ".join(out) or passage

    return _compress


def crude_compressor(passage: str, ratio: float = 0.5) -> str:
    """Crude compression heuristic: keep only sentences with above-average
    information density (lots of nouns / numbers / proper-nouns). NOT a
    substitute for LongLLMLingua; this is a smoke baseline."""
    if not passage or ratio >= 1.0:
        return passage
    sentences = _SENTENCE_SPLIT_RE.split(passage)
    if len(sentences) <= 1:
        return passage[:int(len(passage) * ratio)]

    def info_score(s: str) -> float:
        words = _WORD_RE.findall(s)
        if not words:
            return 0.0
        caps = sum(1 for w in words if w[:1].isupper())
        digs = sum(1 for w in words if any(c.isdigit() for c in w))
        return (caps + 2 * digs) / len(words)

    scored = [(info_score(s), s) for s in sentences]
    scored.sort(reverse=True)
    keep = max(1, int(len(scored) * ratio))
    selected_set = {id(s) for _, s in scored[:keep]}
    out = [s for s in sentences if id(s) in selected_set]
    return " ".join(out) or sentences[0]
