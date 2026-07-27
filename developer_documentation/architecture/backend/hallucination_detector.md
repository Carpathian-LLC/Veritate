# hallucination detector

## What it is

Backend for the hallucination overlay: given a model, a backend, and a prompt or chat message, it generates an answer and grades every span of it for hallucination risk. Two files:

- [veritate_mri/inference/hallucination.py](../../../veritate_mri/inference/hallucination.py) : the detection engine. Pure functions over an already generated answer plus its per-byte confidence stream. No decode loop, no disk access except the training-provenance index build.
- [veritate_mri/routes/hallucination_routes.py](../../../veritate_mri/routes/hallucination_routes.py) : `POST /hallucination/analyze`. Drives generation through the existing platform paths, assembles the answer bytes with their per-byte metrics, and calls the engine.

The per-byte `confidence` it consumes is not computed here: [veritate_mri/training/confidence.py](../../../veritate_mri/training/confidence.py) calibrates it from four components and both backends emit it on every `token` frame (rule 23). This module is a consumer of that field.

## How it works

`POST /hallucination/analyze` (`hallucination_routes.py::register`):

1. **Framing.** A `message` triggers chat framing: when `use_rag` is set and the KB has files, `hybrid_routes.retrieve(message, k, scope)` returns the top-`k` context chunks, and the main prompt is `hybrid_routes.build_prompt(message, facts)`; the no-context prompt (`build_plain_prompt`) is kept for the divergence pass. A raw `prompt` is generated as-is with no context. Backend readiness reuses `hybrid_routes._ensure_c` / `_ensure_pytorch`.
2. **Assembly, two modes.** **Deferred** (`frames` in the request body, used by the auto-run on generation `done`): `_assemble_from_frames` grades the exact answer the client already streamed, reusing `_finalize` so the sanitize/cut/strip is identical to the live path. No re-generation, no divergence, no backend readiness check, no dependence on the model still being loaded: this is the robust, deterministic default. **Generation** (`frames` absent, e.g. the manual Detect button): `_run` + `_assemble` re-run the prompt through `backends_routes._c_engine_stream` / `brain.stream` and `_stop_on_bytes` (no new decode loop) and can additionally measure divergence. Both collect `byte`, `confidence`, `surprise_bits`, `entropy_bits` per emitted byte; `_finalize` sanitizes byte-wise (restores the C wire newline `0x01`, drops control bytes) **keeping the per-byte metrics aligned**, cuts at the first turn marker, and strips edges. The result is `answer` plus a `metrics` list aligned to `answer.encode("utf-8")`, so byte offsets stay exact for the frontend to colour the precise characters.
3. **Span aggregation** (`hallucination.segment_spans`). Paragraphs split on blank lines, sentences on `.!?`/newline, words on whitespace. Byte-exact `start`/`end`. `confidence` rolls up child means (word from its bytes, sentence from its words, paragraph from its sentences); `confidence_min` is the weakest byte in the span (catches a single low-confidence byte). `overall.confidence` is the byte-weighted mean over the whole answer.
4. **Grounded overlap** (`annotate_grounding`, `grounded_sources`), context present only. Each word/sentence/paragraph is `yes`/`no`/`partial`/`null` by whether its content words (lowercased, stopwords dropped, via the retriever's `_tokenize`) appear in the concatenated context. `grounded_fraction` is the fraction of answer content-words found in context. `grounded_sources` points each grounded sentence at the retrieved chunk it best overlaps (exact lexical match, a real source pointer).
5. **Context divergence** (`divergence_score`), generation mode + context only. A second answer is generated with the same settings but no context, near-greedy (`DIVERGENCE_TEMPERATURE`) for stability; divergence is `1 - Jaccard` over the two answers' content-word sets. High divergence means the context materially changed the answer. Deferred mode reports `divergence: null` (it needs a second generation, so it stays on the manual Detect path).
6. **Training-data provenance proxy** (`build_training_index`, `training_matches`). A BM25 index is built once per process over a bounded sample of `trainers/corpus` and cached (`_TRAIN_INDEX`). For the weakest spans (ungrounded first, then lowest `confidence_min`), it returns the nearest training passages. **This is nearest-neighbour similarity, not proven causal attribution** (training-data attribution is unsolved); the field semantics say so.
7. **Verdict** (`build_overall`, `pick_verdict`). `uncertain` is `overall.confidence < UNCERTAIN_CONFIDENCE_THRESHOLD` (the abstain line). `hallucination_risk` blends `1 - confidence`, `1 - grounded_fraction`, and divergence over the terms available (monotonic: worse grounding/divergence/confidence never lowers it). The verdict is checked in priority order (context present unless noted):

   | verdict | condition |
   |---|---|
   | `refused` | the answer matches a refusal marker |
   | `grounded` | `grounded_fraction >= GROUNDED_HIGH_FRACTION` (0.85) |
   | `likely_hallucinated` | `context_divergence >= DIVERGENCE_HALLUCINATION` (0.8) and `grounded_fraction < GROUNDED_HIGH_FRACTION` (0.85) |
   | `partially_grounded` | `PARTIAL_GROUNDED_LOW_FRACTION` (0.15) `< grounded_fraction < GROUNDED_HIGH_FRACTION` (0.85), lower divergence |
   | `low_confidence` | `uncertain` |
   | `ungrounded_ok` | none of the above |

   The escalation catches the partially-grounded-but-wrong case: an answer that lifts a context token yet fabricates the rest (e.g. context "employs 340 staff", answer "400 staff") lands at `grounded_fraction` ~0.5 with `context_divergence` ~1.0 and is labelled `likely_hallucinated`, not a benign verdict. `grounded` requires near-full grounding (>= 0.85), so a half-supported answer is at best `partially_grounded` (the "check this" state), never `grounded`.

### Both backends

The PyTorch brain carries the full four-component calibrated confidence; the C engine carries `confidence`/`margin`/`entropy` (its frames set `lens_consistency`/`residual_stab` to 0, but `confidence` is populated). The route consumes whatever `confidence` a frame emits. If a frame carries no `confidence` at all, `_finalize` derives one from surprise (`hallucination.derive_confidence`) and the response sets `confidence_source` to `entropy_surprise`; otherwise it is `byte_confidence`.

### Calibration knobs

`UNCERTAIN_CONFIDENCE_THRESHOLD`, `GROUNDED_HIGH_FRACTION` (0.85), `PARTIAL_GROUNDED_LOW_FRACTION` (0.15), `DIVERGENCE_HALLUCINATION` (0.8), and `RISK_WEIGHTS` are named constants at the top of `hallucination.py`. Tune them against a labelled set; the code reads no literals in function bodies.

### Training-provenance caps

`TRAINING_SAMPLE_MAX_BYTES` (4 MB total), `TRAINING_SAMPLE_PER_FILE_BYTES` (512 KB), and a printable-byte probe (`_looks_texty`) bound the sample: only text-like `*_train.bin` files are sampled, per file and in total, in filename order until the cap. The build logs what it sampled (`n_files`, `chunks`, `bytes`). The sample is a slice, not the full corpus, so a span with no lexical hit in the slice returns no passages.

## Dependencies

- [inference/agent/tools/retriever.py](../../../veritate_mri/inference/agent/tools/retriever.py) : `BM25Index`, `_split_chunks`, `_tokenize` (content-word tokenizer + chunker, reused for grounding and the training index).
- [routes/hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py) : retrieval, prompt framing, backend loaders, wire-sanitize constants.
- [routes/backends_routes.py](../../../veritate_mri/routes/backends_routes.py) : `_c_engine_stream`, `_chat_stop_seq`, `_stop_on_bytes`.
- [training/confidence.py](../../../veritate_mri/training/confidence.py) : the per-byte confidence field this module consumes.
- Registered in [veritate_mri/app.py](../../../veritate_mri/app.py) via `hallucination_routes.register(app)`.

## Pitfalls

- **Offset alignment.** Metrics are aligned to the sanitized bytes. If a truncated answer contains invalid UTF-8, `decode("replace")` can change the byte count; `segment_spans::_reconcile` pads or clips metrics to the answer's UTF-8 length so offsets never go out of range. For plain ASCII this never fires.
- **Divergence costs a second generation.** Only runs when context was retrieved. Near-greedy, not fully deterministic (the stream takes no seed), so small sampling noise adds a divergence floor.
- **Provenance is similarity, not attribution.** `training_matches` are nearest passages by BM25; they do not prove the model learned the span from them. Keep the field semantics honest in any UI copy.
- **The training index is process-lifetime cached.** It does not rebuild when `trainers/corpus` changes under a running server; a restart re-samples.
