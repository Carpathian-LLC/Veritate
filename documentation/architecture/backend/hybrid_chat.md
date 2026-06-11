# hybrid chat (front-door chat endpoint)

## What it is

The chat endpoint behind the `/` front door. Lives at [veritate_mri/routes/hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py). It answers a user message with a model the user selects: a locally trained model on the PyTorch or C engine, optionally grounded with retrieval, or the cloud teacher when no local model is chosen. Served to users at `/` and `/chat` (page `veritate_mri/web/hybrid.html`).

## How it works

`POST /hybrid/chat` resolves three knobs from the request body: `model`, `backend` (`pytorch` | `c`), and `use_rag`.

1. **Routing.** `model == "teacher"` routes to the user's configured teacher provider/model. Otherwise `is_local_model(name)` ([hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py)) is true when the name maps to a model dir with a checkpoint; that runs locally. Anything else (empty, `cloud`, unknown) goes to the public model.
2. **Retrieval (opt-in).** Only when `use_rag` is set and the knowledge base has files (`has_corpus()`): BM25 over the user's uploaded text returns the top-`k` chunks (`TOP_K = 3`). Retrieval is pure local lexical search, no embedder and no external service. Retrieval off or no hits → a plain `<|user|>…<|assistant|>` prompt; hits → the `PROMPT_TMPL` context prompt. Each chunk is capped at `KB_CHUNK_PREVIEW` bytes to bound the prompt for small models.
3. **Generation.** The selected model+engine is loaded through the platform's canonical helpers, then decoded:
   - PyTorch: `_ensure_pytorch` reuses `_brain.load_pytorch_brain` and caches the brain in `cfg["BRAIN"]`; tokens come from `brain.stream(...)`.
   - C engine: `_ensure_c` spawns/reuses the subprocess via `_spawn_c_subprocess`; tokens come from `_c_engine_stream(...)`.
   - `collect()` joins the stream's `token` / `fast_byte` byte events into text; `_trim()` cuts at the first stop marker (`<|end|>`, `<|user|>`, `\ncontext:`).
4. **Public model.** When no local model is selected, `_cloud_answer()` calls `ai_assist.chat(message)` — the always-available public Carpathian model. Its key is the hardcoded public `ai_api_key` in settings DEFAULTS (a user-set `ai_api_key_user` overrides it), so it works with no setup. The answer is labelled `Carpathian AI (public)`. No key on the install → 503.
5. **Teacher model.** If a teacher provider + model are configured in Settings, `_teacher_answer()` calls `teacher.client.complete(provider, model, ...)` with that provider's key (settings or `VERITATE_TEACHER_API_KEY`). This is any of the known providers (OpenAI, Anthropic, etc.), not just Carpathian. Unconfigured / missing key → 503. The configured teacher is surfaced to the page via `teacher_view()` so it appears in the dropdown.

Generation reuses platform code; this module owns no decode loop and pins no model name. The corpus chunks + BM25 index are cached in `_STATE` and rebuilt on the next retrieve after an upload (`reset_index_cache`). The loaded model lives in shared app config (`cfg["BRAIN"]` / `cfg["C_SUBPROCESS"]`), the same state the platform backends use.

### Knowledge base

Retrieval reads `KB_DIR` (`veritate_mri/data/kb`, gitignored): a folder of plain text files the user uploads. `_kb_chunks()` reads every file and splits it with the platform retriever's whitespace-aware `_split_chunks`; `_kb_index()` builds a `BM25Index` ([inference/agent/tools/retriever.py](../../../veritate_mri/inference/agent/tools/retriever.py)) over those chunks. No Ollama, no embeddings, no `.npz`.

### Chat page UI

The page ([veritate_mri/web/hybrid.html](../../../veritate_mri/web/hybrid.html)) is a self-contained dark page with the platform `V·E·R·I·T·A·T·E` logo. A `settings` toggle opens an inline panel with three sections:

- **Model** select: trained models from `/pytorch-models`, a `Carpathian AI (public)` entry (value `cloud`, always available), the configured teacher (value `teacher`, shown only when `/hybrid/health` reports one) labelled `provider: model`, and a `Train your own →` link to `/app#training`. Selecting any remote model (public or teacher) hides the engine and knowledge sections.
- **Engine** segment: `PyTorch` / `Veritate (C)`. The C button is disabled for models without a `veritate.bin` (cross-checked against `/c-models`), with an inline hint.
- **Knowledge** section: a checkbox to ground answers (disabled until the corpus has files) plus a file uploader that posts to `/hybrid/kb/upload`. Upload is instant (no embedding); the page then refreshes `/hybrid/health` and enables grounding.

Selection persists in `localStorage` under `veritate_chat_settings_v1`. Each send posts `{message, model, backend, use_rag}` to `/hybrid/chat` and renders `sources` only when retrieval returned any.

## Endpoints

- `GET /hybrid/health`: returns `{ok, has_corpus, n_files, teacher: {configured, label}}`. The chat page uses `has_corpus` to gate the retrieval checkbox and `teacher` to add the configured teacher to the dropdown.
- `POST /hybrid/kb/upload`: multipart `file`. Saves it into `KB_DIR` (sanitized name, `.txt` default), clears the index cache, and returns `{ok, filename, n_files, n_chunks}`. Rejects no-file with 400 and oversize (> `UPLOAD_MAX_BYTES`) with 413.
- `POST /hybrid/chat`: body `{message, model, backend?, use_rag?, k?}`. Returns `{ok, answer, model, backend, confident, sources}`. `sources` is the retrieved `[{text, score}]` (empty unless RAG ran). Empty message → 400; load/generation failures → 503 with a user-safe message; cloud auth/availability → 503.

Both are public (auth gate exempts the `/hybrid` prefix and `/`). See [auth.md](auth.md).

## Dependencies

- For local generation: a trained model with a checkpoint (PyTorch) or a `veritate.bin` export plus a built C engine (C). Model lists come from `/pytorch-models` and `/c-models`.
- For retrieval: nothing external. Pure-Python BM25 over the uploaded `KB_DIR` files.
- For the public model: the hardcoded public Carpathian key in settings DEFAULTS (`ai_api_key`, base `https://api.carpathian.ai/ai/v1`), called via [ai_assist.py](../../../veritate_mri/runtime/ai_assist.py). A user-set `ai_api_key_user` overrides it.
- torch and the c engine subprocess (lazy-imported, only for local generation).

## Pitfalls

- BM25 is lexical: it matches on shared keywords, not paraphrase. Retrieval quality comes from the uploaded corpus, not a fancy ranker.
- The C engine option requires a `veritate.bin` export and a built engine binary. The chat page disables C for models without a `.bin` (via `/c-models`); a missing engine binary surfaces as a 503 from `_ensure_c`.
- Switching the selected model swaps the shared `cfg["BRAIN"]` / `cfg["C_SUBPROCESS"]`, the same slots the platform tab uses; the last-loaded model is the current one for both surfaces.
- The public model uses the hardcoded public key (`ai_api_key`); if that is blanked on an install, the public option returns 503 until a key is restored or `ai_api_key_user` is set.
- The corpus + BM25 index are cached in `_STATE`; uploads clear the cache, but editing `KB_DIR` files on disk out-of-band needs a server restart (or another upload) to take effect.
