# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - BM25 retriever over a local corpus. Pure Python, no torch dependency.
#   The corpus is split into overlapping chunks at construction; queries return
#   the top-K chunks by BM25 score.
# - Tokenization: simple word-character + lowercase. Stopword removal is
#   conservative (only a handful of high-frequency English words). Good enough
#   for a 1B-class agent's retrieval needs; RAG quality wins come from corpus
#   curation, not from a fancy tokenizer.
# - The index is built once at tool construction; subsequent calls are O(K log N).
# veritate_mri/agent/tools/retriever.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from . import Tool

# ------------------------------------------------------------------------------------
# Constants

_CHUNK_BYTES   = 1024
_CHUNK_OVERLAP = 128
_TOP_K_DEFAULT = 4

_BM25_K1 = 1.5
_BM25_B  = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "it", "on",
    "at", "by", "with", "as", "that", "this", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "but",
    "if", "then", "than", "so", "not", "no", "yes",
}

# ------------------------------------------------------------------------------------
# Functions


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _split_chunks(text: str, chunk_bytes: int, overlap: int) -> List[Tuple[int, str]]:
    """Yield (offset, chunk_text) tuples. Best-effort split on whitespace boundaries."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_bytes)
        # Back up to a whitespace if we're mid-word.
        if end < n:
            j = end
            while j > i and not text[j].isspace():
                j -= 1
            if j > i + chunk_bytes // 2:  # only honor split if meaningful
                end = j
        out.append((i, text[i:end].strip()))
        if end >= n:
            break
        i = max(i + 1, end - overlap)
    return out


class BM25Index:
    """Build once, query many times. Documents are (offset, chunk_text) tuples."""

    def __init__(self, chunks: List[Tuple[int, str]]):
        self.chunks: List[Tuple[int, str]] = chunks
        self.tokens: List[List[str]] = [_tokenize(c) for _, c in chunks]
        self.doc_lens: List[int] = [len(t) for t in self.tokens]
        self.avg_dl: float = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.N: int = len(self.tokens)
        # Inverted index: term -> list of (doc_id, term_freq)
        self.posting: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for did, toks in enumerate(self.tokens):
            c = Counter(toks)
            for term, tf in c.items():
                self.posting[term].append((did, tf))
        # idf cache
        self.idf: Dict[str, float] = {}
        for term, post in self.posting.items():
            df = len(post)
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = _TOP_K_DEFAULT) -> List[Tuple[float, int]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term in set(q_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for did, tf in self.posting[term]:
                dl = self.doc_lens[did]
                norm = 1 - _BM25_B + _BM25_B * (dl / max(1.0, self.avg_dl))
                score = idf * tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * norm)
                scores[did] += score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(s, did) for did, s in ranked]


def make_tool(corpus_path: str, top_k: int = _TOP_K_DEFAULT,
              chunk_bytes: int = _CHUNK_BYTES, overlap: int = _CHUNK_OVERLAP) -> Tool:
    """Index `corpus_path` (file or directory of text files) and return a tool."""
    if not os.path.exists(corpus_path):
        raise ValueError(f"retriever corpus does not exist: {corpus_path}")

    # Collect text
    blobs: List[Tuple[str, str]] = []  # (source, text)
    if os.path.isfile(corpus_path):
        with open(corpus_path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        blobs.append((os.path.basename(corpus_path), text))
    else:
        for dirpath, _, fnames in os.walk(corpus_path):
            for fn in fnames:
                if not fn.lower().endswith((".txt", ".md", ".rst", ".text")):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                text = data.decode("utf-8", errors="replace")
                rel = os.path.relpath(fp, corpus_path)
                blobs.append((rel, text))

    chunks: List[Tuple[int, str]] = []     # (idx, chunk_text)
    chunk_sources: List[Tuple[str, int]] = []  # (source, offset)
    for source, text in blobs:
        for off, ch in _split_chunks(text, chunk_bytes, overlap):
            chunks.append((len(chunks), ch))
            chunk_sources.append((source, off))

    if not chunks:
        raise ValueError(f"retriever corpus has no readable text chunks at {corpus_path}")

    idx = BM25Index([(c[0], c[1]) for c in chunks])

    def _execute(args: Dict[str, Any]) -> str:
        query = args.get("query")
        if query is None:
            return "error: missing required arg 'query'"
        k = args.get("k", top_k)
        try:
            k = max(1, min(int(k), 16))
        except (TypeError, ValueError):
            return "error: 'k' must be an integer 1..16"
        hits = idx.search(str(query), k=k)
        if not hits:
            return "no matches"
        lines = []
        for score, did in hits:
            src, off = chunk_sources[did]
            chunk = idx.chunks[did][1]
            preview = chunk[:480].replace("\n", " ").strip()
            lines.append(f"[{src} @{off}] (score {score:.2f}) {preview}")
        return "\n\n".join(lines)

    return Tool(
        name="retrieve",
        description=f"Search a local text corpus by keywords. Returns top-K chunks by BM25 score.",
        args_schema={
            "query": {"type": "string", "required": True,
                      "doc": "Search query (free text). Tokens are matched case-insensitively."},
            "k":     {"type": "integer", "required": False,
                      "doc": f"Number of results to return (default {top_k}, max 16)."},
        },
        execute=_execute,
    )
