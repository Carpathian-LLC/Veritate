# knowledge base (kb_build)

## What it is

Scalable knowledge-base pipeline for the hybrid RAG chat. Lives at
[experiments/v2/rag/kb_build.py](../../../experiments/v2/rag/kb_build.py). Ingests short
factual statements into a single embedded index `kb_index.npz`, and serves cosine
retrieval over it. Two ingest modes feed one shared index, and writes APPEND so the base
grows across runs.

## How it works

Index file `kb_index.npz` holds two arrays
([kb_build.py:167](../../../experiments/v2/rag/kb_build.py#L167)):

- `vectors`: float32, shape `(n, d)` where `d` is the embedder's dimension, L2-normalized.
- `texts`: object array of the source strings, parallel to `vectors`.

Embeddings come from Ollama `mxbai-embed-large` via POST `/api/embeddings`
([embed at kb_build.py:75](../../../experiments/v2/rag/kb_build.py#L75)). `embed_all`
([kb_build.py:81](../../../experiments/v2/rag/kb_build.py#L81)) loops items and prints a
progress line every 200.

### Ingest modes

- **GENERATE** ([generate_facts at kb_build.py:109](../../../experiments/v2/rag/kb_build.py#L109)):
  asks Ollama `qwen2.5:7b-instruct` (POST `/api/chat`,
  [_gen_batch at kb_build.py:90](../../../experiments/v2/rag/kb_build.py#L90)) for short
  single-answer facts across twelve domains
  ([DOMAINS at kb_build.py:42](../../../experiments/v2/rag/kb_build.py#L42)): geography,
  science, history, technology, arts, sports, nature, space, health, economics, food,
  language. Requested count is split per domain, generated in batches, deduped, and
  trimmed to count.
- **FILE** ([chunk_file at kb_build.py:142](../../../experiments/v2/rag/kb_build.py#L142)):
  reads a plain UTF-8 text file and splits on sentence boundaries
  ([SENT_ENDERS at kb_build.py:58](../../../experiments/v2/rag/kb_build.py#L58)). Each chunk
  is run through `clean_text` ([kb_build.py:135](../../../experiments/v2/rag/kb_build.py#L135)),
  which applies the wikitext fixups in `WIKI_REPLACEMENTS`
  ([kb_build.py:44](../../../experiments/v2/rag/kb_build.py#L44), e.g. `@-@` -> `-`) and
  collapses whitespace. Chunks containing `=` or `<` (wiki headers / markup) are dropped, and
  only chunks within `[MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)` = `[40, 240)` are kept
  ([kb_build.py:38](../../../experiments/v2/rag/kb_build.py#L38)). `--limit` caps the chunk
  count (0 = no cap).

### Append

`save_append` ([kb_build.py:159](../../../experiments/v2/rag/kb_build.py#L159)) loads the
existing index if present, vstacks the new float32 vectors, concatenates the texts, and
re-saves. A fresh build (no file yet) writes the new arrays directly.

### Retrieve

`retrieve(query, k)` ([kb_build.py:171](../../../experiments/v2/rag/kb_build.py#L171))
embeds and L2-normalizes the query, takes the dot product against the normalized matrix
(cosine, since both sides are unit-norm), and returns the top-k `(text, score)` pairs.
Plain numpy is fine up to roughly 100k vectors; FAISS is the scale path beyond that.

## Usage

```
python experiments/v2/rag/kb_build.py --mode generate --count 2500
python experiments/v2/rag/kb_build.py --mode file --file path/to/notes.txt --limit 5000
```

Both append into `kb_index.npz`. Delete the file first for a clean rebuild.

## Dependencies

- Ollama serving `qwen2.5:7b-instruct` (generate mode) and `mxbai-embed-large` (all modes)
  on `http://localhost:11434`.
- numpy.

## Consumers

This is a standalone CLI tool for building an embedded `kb_index.npz`. The front-door chat
([hybrid_chat.md](hybrid_chat.md)) no longer uses it: chat grounding is local BM25 over an
uploaded `KB_DIR`, with no embedder. `experiments/v2/rag/build_grounded.py` still reuses
`chunk_file` for its `--facts_file` mode.

## Pitfalls

- The embed model must match between build and serve, or scores are meaningless.
- GENERATE depends on a running Ollama with the chat model pulled; a missing model fails
  the HTTP call rather than degrading silently.
- Append never dedupes against the existing index; repeated builds of the same content
  duplicate rows.
- numpy cosine scans the full matrix per query. Past ~100k rows, move to FAISS.
