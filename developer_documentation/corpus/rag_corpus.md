# rag corpus builder

## What it is

Builds the context-grounded (RAG) SFT corpus: examples that place a fact in a context block, ask a question about it, and answer from the context. Training on it teaches a model to use retrieved text instead of recalling weights. Lives at [veritate_mri/tools/build_rag_corpus.py](../../veritate_mri/tools/build_rag_corpus.py).

## How it works

- `gen_facts(n)` asks the configured Teacher Model for short self-contained facts, spread evenly over `FACT_CATS`. `gen_qa(fact)` turns one fact into a question plus a short extractive answer. Both go through `_chat`, which resolves the teacher once via `veritate_core.plugin.get_teacher_client()` and exits with a clear message when none is configured.
- Facts split `TEST_FRAC` held-out / rest train. Held-out facts never appear in the bins, so the test set measures in-context copy on unseen facts rather than memorization.
- `build(n_facts, qa_per_fact, stem)` renders train examples through `TEMPLATE`, writes `<stem>_train.bin` and `<stem>_val.bin` (`VAL_FRAC` tail) into `trainers/corpus/`, and writes the held-out items to `veritate_mri/data/eval/rag/<stem>_test.json` (`paths.rag_eval_path`).
- Entry point: spawned as a subprocess by [rag_routes.py](../architecture/backend/routes.md) with `--n_facts` and `--stem`. Stdout is the job log the RAG panel tails.

## Dependencies

- `veritate_core.plugin.get_teacher_client()` for every generation call.
- `readers.paths` for the corpus dir and the held-out set path. No path strings are built here.
- [rag_sft.py](../architecture/backend/rag_sft.md) consumes the bins this writes.

## Pitfalls

- Teacher-required. With no teacher configured the process exits immediately; the dashboard gates the RAG action behind `#teacherGate` for the same reason.
- Malformed teacher JSON for a single category or fact is skipped, so a run can produce fewer facts than requested. The final line reports the real counts.
- Rerunning with the same stem overwrites both bins and the held-out set. Facts are teacher-sampled, not seeded, so two runs are not byte-identical.
