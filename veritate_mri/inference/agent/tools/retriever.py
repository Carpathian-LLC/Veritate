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
from typing import Any

from . import ERROR_PREFIX, Tool

# ------------------------------------------------------------------------------------
# Constants

# File suffixes indexed when the corpus path is a directory. Shared with the
# dashboard's corpus-signature walk (routes/backends_routes.py) so both see the
# same set of files.
CORPUS_EXTENSIONS = (".txt", ".md", ".rst", ".text")

_CHUNK_BYTES   = 1024
_CHUNK_OVERLAP = 128
_TOP_K_DEFAULT = 4
_K_MIN         = 1
_K_MAX         = 16

_BM25_K1 = 1.5
_BM25_B  = 0.75

_ARG_QUERY = "query"
_ARG_K     = "k"

_TOOL_NAME        = "retrieve"
_TOOL_DESCRIPTION = "Search a local text corpus by keywords. Returns top-K chunks by BM25 score."
_QUERY_DOC        = "Search query (free text). Tokens are matched case-insensitively."
_K_DOC_FMT        = "Number of results to return (default {default}, max {maximum})."
_HIT_FMT          = "[{source} @{offset}] (score {score:.2f}) {chunk}"
_HIT_SEPARATOR    = "\n\n"
_NO_MATCHES       = "no matches"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "it", "on",
    "at", "by", "with", "as", "that", "this", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "but",
    "if", "then", "than", "so", "not", "no", "yes",
}

# ------------------------------------------------------------------------------------
# Functions


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _split_chunks(text: str, chunk_bytes: int, overlap: int) -> list[tuple[int, str]]:
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

    def __init__(self, chunks: list[tuple[int, str]]):
        self.chunks: list[tuple[int, str]] = chunks
        self.tokens: list[list[str]] = [_tokenize(c) for _, c in chunks]
        self.doc_lens: list[int] = [len(t) for t in self.tokens]
        self.avg_dl: float = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.N: int = len(self.tokens)
        # Inverted index: term -> list of (doc_id, term_freq)
        self.posting: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for did, toks in enumerate(self.tokens):
            c = Counter(toks)
            for term, tf in c.items():
                self.posting[term].append((did, tf))
        # idf cache
        self.idf: dict[str, float] = {}
        for term, post in self.posting.items():
            df = len(post)
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = _TOP_K_DEFAULT) -> list[tuple[float, int]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
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
    blobs: list[tuple[str, str]] = []  # (source, text)
    if os.path.isfile(corpus_path):
        with open(corpus_path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        blobs.append((os.path.basename(corpus_path), text))
    else:
        for dirpath, _, fnames in os.walk(corpus_path):
            for fn in fnames:
                if not fn.lower().endswith(CORPUS_EXTENSIONS):
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

    chunks: list[tuple[int, str]] = []     # (idx, chunk_text)
    chunk_sources: list[tuple[str, int]] = []  # (source, offset)
    for source, text in blobs:
        for off, ch in _split_chunks(text, chunk_bytes, overlap):
            chunks.append((len(chunks), ch))
            chunk_sources.append((source, off))

    if not chunks:
        raise ValueError(f"retriever corpus has no readable text chunks at {corpus_path}")

    idx = BM25Index([(c[0], c[1]) for c in chunks])

    def _execute(args: dict[str, Any]) -> str:
        query = args.get(_ARG_QUERY)
        if query is None:
            return f"{ERROR_PREFIX}missing required arg {_ARG_QUERY!r}"
        k = args.get(_ARG_K, top_k)
        try:
            k = max(_K_MIN, min(int(k), _K_MAX))
        except (TypeError, ValueError):
            return f"{ERROR_PREFIX}{_ARG_K!r} must be an integer {_K_MIN}..{_K_MAX}"
        hits = idx.search(str(query), k=k)
        if not hits:
            return _NO_MATCHES
        lines = []
        for score, did in hits:
            src, off = chunk_sources[did]
            # Whole chunk, never a preview: this string IS the context the reader
            # gets, so truncating here silently drops the answer whenever it sits
            # past the cut. Newlines flatten because callers split hits on a blank
            # line; consumers cap length themselves.
            chunk = idx.chunks[did][1].replace("\n", " ").strip()
            lines.append(_HIT_FMT.format(source=src, offset=off, score=score, chunk=chunk))
        return _HIT_SEPARATOR.join(lines)

    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema={
            _ARG_QUERY: {"type": "string", "required": True, "doc": _QUERY_DOC},
            _ARG_K:     {"type": "integer", "required": False,
                         "doc": _K_DOC_FMT.format(default=top_k, maximum=_K_MAX)},
        },
        execute=_execute,
    )
