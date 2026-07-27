# research

Central index for all Veritate research. Start here. Root placement is an explicit user instruction (2026-07-07).

## the ledger (living workflow)

Three root files, one pipeline:

- `ideas.md` : open ideas and active campaigns. Front of the pipeline.
- `successes.md` : validated results with the evidence that proved them.
- `failures.md` : falsified approaches (the kill list), each with a retry condition.

Flow: an idea proven moves from `ideas.md` to `successes.md`; an idea killed moves from `ideas.md` to `failures.md`. Full workflow at the top of `ideas.md`. Nothing lives in two files.

## standing research maps (indexed, not folded in)

Living research maps and plans still under `developer_documentation/`. They stay where component contracts reference them and are indexed here.

- `developer_documentation/training/efficient_architecture_research.md` : the ranked lever map (Muon, patching, recurrence, memory, MoE, MTP, MLA, distillation), each lever with a falsifier and status. Parent program; `successes.md` and `failures.md` are its outcomes.
- `developer_documentation/training/chat_model_80m_plan.md` : the three-phase 80M chat recipe (now a proven success).
- `developer_documentation/training/chat_model_200m_plan.md` : the chat200m scale-up plan (pretrain launched 2026-07-08) with its pre-registered gates.
- `developer_documentation/research/long_context_memory.md` : IDEA 1 in-context streaming-state memory (needle benchmark, transfer-gap, bitwise-exact state carry). The always-on gist tier.
- `developer_documentation/research/external_memory_retrieval.md` : IDEA 2 external addressable memory (trained byte-native key head, sub-quadratic drill-down, FAISS trillion-feasibility, natural-query transfer, productionization plan). The exact-recall tier; folds the 2026-07-11..13 `successes.md`/`failures.md` memory entries into one story.
- `developer_documentation/platform/mem_planner.md` : memory-planning component.

The market and trading research line was retired with the trading extension at 1.0.0. Its outcomes stay in `successes.md` and `failures.md`; the platform no longer carries the code or the plan.

## root working docs

- `worklog.md` : the general work log. A timeline (headlines), a plain-language narrative of the chat models, and the dated technical sections the papers cite. 
