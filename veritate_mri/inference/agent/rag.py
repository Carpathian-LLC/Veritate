# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - In-context RAG prefix injection. The right mechanism for byte-level
#   models per the sprint-2 survey: retrieve passages, inject as prompt
#   PREFIX, let the model's attention figure out what to use.
# - This is the OPPOSITE of the F09 corpus-echo failure. F09 mixed an
#   n-gram retrieval distribution with the model's logits at alpha=1.5;
#   it tanked the 85M's val NLL by +36%. Survey verdict: F09 was the
#   logit-fusion choice, not byte-level RAG itself. Prefix injection is
#   how every published 1B-class SLM-RAG actually works.
# - Composition:
#     1. Retriever (e.g., veritate_mri.agent.tools.retriever's BM25) returns
#        top-K passages for the user query.
#     2. We build a prefix:
#          "Context:\n[1] <passage1>\n[2] <passage2>\n...\nQuestion: <user>\nAnswer: "
#     3. Brain.stream(prefix, ...) generates the answer, optionally under
#        constrained decoding.
# - LongLLMLingua-style compression hook is a stub for now; the published
#   technique compresses retrieved context 4-6x at +21pt accuracy. When
#   we have a compressor model, plug it in here.
# veritate_mri/agent/rag.py
# ------------------------------------------------------------------------------------
# Imports:

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_TOP_K          = 3
DEFAULT_MAX_PASSAGES_B = 4096   # cap concatenated context bytes (~4 chunks at 1 kB)
DEFAULT_PREFIX_FORMAT  = "Context:\n{passages}\n\nQuestion: {query}\nAnswer: "
DEFAULT_PASSAGE_FORMAT = "[{i}] {chunk}"

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class RAGResult:
    """One RAG-augmented generation result."""
    query:             str
    retrieved:         List[str] = field(default_factory=list)
    retrieved_scores:  List[float] = field(default_factory=list)
    prefix:            str = ""
    prefix_bytes:      int = 0
    answer:            str = ""
    full_text:         str = ""
    elapsed_s:         float = 0.0
    compression_ratio: float = 1.0


def build_rag_prefix(query: str, passages: List[str],
                     prefix_format: str = DEFAULT_PREFIX_FORMAT,
                     passage_format: str = DEFAULT_PASSAGE_FORMAT,
                     max_bytes: int = DEFAULT_MAX_PASSAGES_B,
                     compressor: Optional[Callable[[str], str]] = None) -> str:
    """Build a prompt prefix from retrieved passages + a user query.
    If `compressor` is provided, run each passage through it first
    (LongLLMLingua-style).
    Caps the concatenated passages at `max_bytes`."""
    chunks = []
    total = 0
    for i, p in enumerate(passages, 1):
        if compressor is not None:
            p = compressor(p)
        line = passage_format.format(i=i, chunk=p)
        if total + len(line.encode("utf-8")) > max_bytes:
            # Truncate the last passage instead of dropping it
            remaining = max_bytes - total
            if remaining > 32:
                line = line[:remaining] + "... [truncated]"
                chunks.append(line)
            break
        chunks.append(line)
        total += len(line.encode("utf-8"))
    body = "\n".join(chunks)
    return prefix_format.format(passages=body, query=query)


class RAGRunner:
    """In-context RAG runner. Wraps a Brain-shaped backend and a retriever-
    callable that returns (passages, scores) given a query."""

    def __init__(self, backend, retriever: Callable[[str, int], list],
                 compressor: Optional[Callable[[str], str]] = None,
                 top_k: int = DEFAULT_TOP_K,
                 prefix_format: str = DEFAULT_PREFIX_FORMAT,
                 passage_format: str = DEFAULT_PASSAGE_FORMAT,
                 max_prefix_bytes: int = DEFAULT_MAX_PASSAGES_B):
        self.backend = backend
        self.retriever = retriever
        self.compressor = compressor
        self.top_k = top_k
        self.prefix_format = prefix_format
        self.passage_format = passage_format
        self.max_prefix_bytes = max_prefix_bytes

    def run(self, query: str,
            temperature: float = 0.7,
            top_k_sample: int = 40,
            max_new: int = 200,
            constraint=None) -> RAGResult:
        import time
        t0 = time.time()

        # Retrieve
        try:
            hits = self.retriever(query, self.top_k)
        except Exception as e:
            hits = []
        passages = [h[0] if isinstance(h, (tuple, list)) else str(h) for h in hits]
        scores = [float(h[1]) if isinstance(h, (tuple, list)) and len(h) > 1 else 0.0
                  for h in hits]

        # Build prefix
        raw_byte_count = sum(len(p.encode("utf-8")) for p in passages)
        prefix = build_rag_prefix(
            query, passages,
            prefix_format=self.prefix_format,
            passage_format=self.passage_format,
            max_bytes=self.max_prefix_bytes,
            compressor=self.compressor,
        )
        prefix_byte_count = len(prefix.encode("utf-8"))
        compression = (raw_byte_count / max(1, prefix_byte_count)) if self.compressor else 1.0

        # Generate
        out_bytes = bytearray()
        for ev in self.backend.stream(prefix,
                                       temperature=temperature,
                                       top_k_sample=top_k_sample,
                                       max_new=max_new,
                                       constraint=constraint):
            kind = ev.get("kind")
            if kind == "token":
                b = ev.get("byte")
                if isinstance(b, int):
                    out_bytes.append(b & 0xff)
            elif kind == "fast_byte":
                b = ev.get("byte")
                if isinstance(b, int):
                    out_bytes.append(b & 0xff)
            elif kind in ("stop", "error"):
                break

        answer_text = bytes(out_bytes).decode("utf-8", errors="replace")
        full_text = prefix + answer_text

        return RAGResult(
            query=query,
            retrieved=passages,
            retrieved_scores=scores,
            prefix=prefix,
            prefix_bytes=prefix_byte_count,
            answer=answer_text,
            full_text=full_text,
            elapsed_s=time.time() - t0,
            compression_ratio=compression,
        )


def bm25_retriever_from_tool(tool):
    """Adapter: wrap our existing BM25 Tool (veritate_mri.agent.tools.retriever)
    so it satisfies the `retriever(query, k) -> list[(passage, score)]`
    signature expected by RAGRunner. The Tool's execute() returns a
    pre-formatted string; we parse it back."""
    def _retrieve(query: str, k: int):
        out = tool.call({"query": query, "k": k})
        if out.startswith("error"):
            return []
        if out == "no matches":
            return []
        # Format: "[src @off] (score 1.23) <preview>\n\n[src @off] (score ..) ..."
        hits = []
        for chunk in out.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Parse leading score
            score = 0.0
            if "(score " in chunk:
                pre, _, rest = chunk.partition("(score ")
                score_s, _, body = rest.partition(") ")
                try:
                    score = float(score_s)
                except ValueError:
                    pass
                hits.append((body, score))
            else:
                hits.append((chunk, 0.0))
        return hits
    return _retrieve


def identity_compressor(passage: str) -> str:
    """Placeholder compressor, returns passage unchanged. Swap with a real
    LongLLMLingua-style compressor (I53) when one ships."""
    return passage


def make_word_ppl_compressor(brain, keep_frac: float = 0.5, max_ctx_bytes: int = 1024):
    """Word-level perplexity compressor (I53 / S60). Uses a Brain-shaped
    backend as the byte-level scorer: per-byte NLL of the passage is
    aggregated to word boundaries by mean; words with the lowest mean
    NLL (most predictable from preceding bytes, least informative) are
    dropped. Single-space joiner between kept words.
    `keep_frac` ∈ (0, 1] is the fraction of words to retain.
    `max_ctx_bytes` caps scoring cost; longer passages are truncated.
    Smoke (85M, S60): word-level @ keep=0.5 ≈ 2× compression for
    +0.19 nll on tinystories continuations; @ keep=0.25 ≈ 4.3× for +0.27.
    Output is human-readable, unlike per-byte deletion."""
    import re
    import torch
    import torch.nn.functional as F
    _WORD_RE = re.compile(rb"\S+|\s+")
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
        except Exception:
            return passage  # silent fallback: never break the chain
        spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(b)
                 if not m.group(0).isspace()]
        if not spans:
            return passage
        scores = []
        for (s, e) in spans:
            seg = nlls[s:e]
            scores.append(sum(seg) / max(1, len(seg)))
        keep_n = max(1, int(round(len(spans) * keep_frac)))
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
    import re
    sentences = re.split(r'(?<=[.!?])\s+', passage)
    if len(sentences) <= 1:
        return passage[:int(len(passage) * ratio)]
    # Score each sentence by its fraction of capitalized words + digits
    def info_score(s: str) -> float:
        words = re.findall(r"\S+", s)
        if not words:
            return 0.0
        caps = sum(1 for w in words if w[:1].isupper())
        digs = sum(1 for w in words if any(c.isdigit() for c in w))
        return (caps + 2 * digs) / len(words)
    scored = [(info_score(s), s) for s in sentences]
    scored.sort(reverse=True)
    keep = max(1, int(len(scored) * ratio))
    selected = scored[:keep]
    # Restore original order
    selected_set = set(id(s) for _, s in selected)
    out = [s for s in sentences if id(s) in selected_set]
    return " ".join(out) or sentences[0]
