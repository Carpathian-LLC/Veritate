# handoff

2026-08-19 update on the 2026-08-18 wrap. Ledgers hold the full numbers; this is state only.

## active (2026-08-19)

- **User directive (permanent): persistent memory is THE research focus** — "tell it once and it doesn't forget"; stateless serving with re-injected context is the problem, not a solution. IDEA 20 (ideas.md) holds the full program: three tiers (carried state / serialized state + delta or surprise-gated slow memory / nightly sleep consolidation into weights), experiment ladder E1-E5, literature anchors with arxiv IDs, pre-registered skips.
- **E1 resolved 2026-08-19**: arm 1b dose falsified (failures.md) — wrap gap 0.407 -> 0.383 over 2,000 extra steps, content probes zero at both doses. Side result: wren1_3@3000 improved across the board (val 0.5491, identity 1.00, grounded 0.38, loop 0.20 = anchor) and is the new streaming base; flag as a chat-ship candidate to re-ladder after the memory campaign.
- **wren1_4 (delta arm) RUNNING**: seeded from wren1_3@3000 with state_rule=delta (b_proj zero-init, bias -2 -> beta ~0.12), 1,000 steps, low LR. Falsifier: wrap gap below the flat gla trend AND any content probe moves off zero (E2 margins, recall x/6, anticopy engagement); chat holds; nonfinite skips ~0 (monitor alarms if >5). Eval stack is one command per probe (scratchpad: cliff_measure.py, e2_retention_probe.py, chat_state_recall.py, anticopy_probe.py, ladder.py).
- **E3 SHIPPED (both parts) 2026-08-19**: `/generate?fast=stream` — unbounded-context generation over forward_streaming (full windows commit state; partial window recomputed per byte with non-boundary padding; prompt never truncated) — plus `state_id=<id>`: per-conversation state persistence to data/stream_states/, next call sends only new bytes, split-call byte parity pinned. 8 tests in tests/mri/test_stream_fast_streaming.py; documentation.md updated.
- **Experience log SHIPPED 2026-08-19**: every /generate exchange (both backends) recorded to data/experience/YYYYMMDD.jsonl (inference/experience.py, 6 tests) — the replay substrate for sleep consolidation (IDEA 20 T3).
- **Delta arm (wren1_4) staged, one command each**: scratchpad build_delta_seed.py (smoke-tested; bakes b_proj zero-init/bias -2 into a seed forked from wren1_3@3000 — required because load_resume_state rightly refuses missing tensors) then POST scratchpad wren1_4_launch.json to /trainers/run. Launch after arm 1b's evals pick the fork step.
- E6 anti-copy baseline measured 2026-08-19 (wren1_3@1000): in-window copy 0.044/engagement 0.028; absorbed 0.000/0.000 (no content survives the state yet) — harness is scratchpad anticopy_probe.py, re-run per checkpoint.
- Next after E1: launch wren1_4 (above), then re-run E2v2/E6/cliff per checkpoint.
- **Chat UI wired to the streaming path 2026-08-19**: fast-mode select gained "stream (carried state)"; in chat mode it sends ONLY the new turn with a per-conversation `state_id` (localStorage, rotated on chat clear and on stale-state errors). node --check clean.
- **Sleep platform pieces 2026-08-19** (user directive: idle-time sleep, "index its thoughts, train on them"): tools/build_experience_corpus.py turns the experience log into experience_{train,val}.bin (dedupe, min-reply, torn-line tolerant; 4 tests) — rehearsal needs no new machinery, it is the corpus mixer drawing from the model's own base corpus. scratchpad/sleep_trigger.py is the idle-gated launcher (idle check -> build bins -> low-LR consolidation launch); NOT armed — validate the E4 recipe first, and the GPU is mid-campaign.
- **Behavioral recall harness** (user: "ask it what did you just say"): scratchpad/chat_state_recall.py — two-turn conversations through the stream path with a state file, turn 2 sent alone, per-item fresh-state leak control. Baseline on wren1_3@1000 in flight. Size question answered for the user 2026-08-19: mechanisms all provable at 200M; scaling probe ladder before committing a 1B-class flagship (M3 Ultra 256GB fits it; wall-clock ~30 days at wren_base token counts is the constraint; sleep design reduces required pretrain).

- **Chat purged 2026-08-20 (user-ordered extraction to a standalone project)**: Chat tab + `/chat` page + `/hybrid/*` chat endpoints removed (chat_tab.js/css, hybrid.html deleted; hybrid_routes.py trimmed to /v1 serving + shared framing/retrieval). CHAT_HANDOFF.md at repo root holds the full inventory and the /generate + /v1 API contract for the new project. Generation tab, /generate fast=stream + state_id, experience log, /v1/chat/completions, and all chat corpora untouched. claude_preflight.md now carries scope as a standing rule (push back once on any expansion beyond train/eval/run, even from the user).
- **Sleep controller SHIPPED 2026-08-20 (disabled by default)**: training/sleep.py + /sleep, /sleep/wake, /sleep/now + 60 s watcher thread. Usage-scaled dose (new exchanges × steps_per_exchange, clamp 50–500), idle gate off experience-log mtime, recipe from the model's own training_args (sleep levers overridden: constant sleep_lr, no warmup, assistant mask, ckpt every 25), auto-prune (run intermediates deleted, finals thinned to sleep_keep_finals, non-sleep ckpts never touched). Gen-bar chip: countdown / progress + eta + wake button. 12 sleep_* settings; arm by setting sleep_enabled + sleep_model AFTER E4 validates. 8 tests green.
- **E4 night-1 (wren1_5) attempt 3 RUNNING 2026-08-20 16:02**: fork of wren1_3@3000, fact_sft:0.75/mixed_chat:0.25, lr 5e-6 flat, 300 steps. Baseline exam 0/50 fwd, 0/50 rev. Forgetting kill line locked: wren1_3@3000 val bpb mixed_chat 0.824 / veritate_chat 2.071 / hansard 1.059 (scratchpad/val_bpb.py); post-sleep mixed_chat must stay ≤ 0.840. On terminal: e4_qa_probe.py wren1_5 {100,200,300}, ladder.py "wren1_5:300", val_bpb.py wren1_5 300.

## distillation tab (SHIPPED 2026-08-20)

All ten planned items done; full numbers in worklog.md. State that matters going forward:

- **The tab is the distillation surface now.** `#authorPanel` / `#synthPanel` moved out of the
  Training tab; the Training tab's action picker has one "Distill a corpus" card that opens it.
- **It runs during training.** The two `body.training-active { display:none !important }` rules
  that hid the panels are gone. `/teacher/target_status` decides whether to raise the confirm.
- **Acceptance gate is live**: `tools/corpus_audit.py`, returned as `audit` from both build routes.
  Reports; does not block. Thresholds and their calibration are in documentation.md.
- **Trap to remember**: `/teacher/synth/status` `completed` = RECORDS, `state.json` `completed` =
  CALLS. Use `calls_ok` / `calls_failed` / `calls_remaining` for anything with a denominator.

Open, for the user to decide:

1. **RESOLVED 2026-08-20, negatively — see failures.md.** Distilling long-form replies from
   qwen2.5-instruct is falsified across four configurations and both model sizes; nothing
   exceeded 274 B against a 200 B median floor. `records_per_call` is now 3 for `conversation`
   (the one lever that worked, +23 B) and the length-distribution wording is in the briefs for
   `conversation`/`carryover`/`cogito`. Do not tune briefs for length again. **Distillation's
   real value on this project is DEPTH** — the curated sources that gave `mixed_chat` its 267 B
   median cap at 7 assistant turns, and sustained 10+ turn conversation is the thing curation
   cannot produce. Point the next distillation run at turn COUNT, not turn length, and accept
   ~120-150 B turns or fix length in post.
2. `mixed_chat` and the `agent` rebuild stub are `coming_soon` with PLACEHOLDER urls — both need
   COS uploads to be installable elsewhere. `mixed_chat` bins exist locally.
3. `cos_delete_list.txt` at repo root is unexecuted: ~2.3 GB of COS objects and 4 GitHub files
   still to delete by hand.
4. `veritate_mri/data/authoring/corpus_spec.json` genre weights were never re-tuned after the
   2026-07-27 per-genre yield finding (format_constraint 1.3%, grounded_read 0.5%). Planned calls
   are not produced records.

## models

- **wren1_0@1250 remains the ship checkpoint**, now behind the serving-default no-repeat guard.
- wren1_1 (1,500-step v2 SFT) falsified its premise: no checkpoint beat wren1_0@1250 (failures.md 2026-08-17).
- wren1_2 (500-step guard-distilled polish, IDEA 19 mechanism 1) falsified too: best loop 0.13 vs anchor 0.20 and target <0.05, and it traded grounded/identity for it (failures.md 2026-08-18). Unlikelihood loss is next in the IDEA 19 queue.
- wren1_3 (1,000-step state-carry adaptation, IDEA 7 Track A arm 1) VALIDATED the retention mechanism (successes.md 2026-08-19): carried state 1e-15 -> 1.27, wrap cliff 1.44 -> 1.30, chat held. It is the streaming-lane base, not the chat ship model. Three platform defects were found and fixed to get it trained (act-ckpt kwargs, padding nan at entry, padding re-inflation mid-stack; plus the non-finite-grad step guard and the carry seam clip) -- tests in tests/training/.
- Naming: disk `wren1_0` is the 3,000-step run the older successes.md entry calls wren1_1; the fleet was renumbered after the 8,000-step original was deleted.

## environment

- The Python env is `.veritate_venv/` (underscore). The raw `venv/` duplicate is deleted and `veritate.py` now builds `.veritate_venv` directly.
- `trainers/` is deleted (71 GB freed, user-approved); `data/corpus/` is the only corpus location on this install.
- Fork-from-step gap: `fork_model` only forks `latest`; a manual seed needs checkpoint + config.json (see worklog 2026-08-18).

## grounded retrieval (all measured on wren1_0@1250, n=37 held-out teacher facts)

- Two-factor split: retriever precision@1 0.784 natural / 0.297 paraphrased; reader 0.568 given the gold fact. Product ceiling 0.445.
- End-to-end shipped 0.162 because full 1,024 B chunks overflowed the 1,024 B window and evicted the frame. Fixed (`injection_budget` in rag.py + backends_routes, tests in tests/mri/test_rag_prefix.py): 0.270 after the fix.
- Re-ranker graduated (IDEA 2): BM25 top-5 -> mxbai-embed cosine top-1 lifts paraphrased precision@1 to 0.500 (+20 pt). Not yet wired into the route; opt-in wiring is the open task.
- Coherence: grounded replies looped 0.43-0.60 vs 0.125-0.20 plain chat; the engine's `no_repeat_ngram=16` ban kills both (grounded 0.03-0.05, plain chat 0.0) at exactly zero accuracy or length cost. **Shipped 2026-08-18**: chat-framed and RAG prompts now default the ban on server-side (`_rep_defaults` in backends_routes, tests in tests/mri/test_rep_defaults.py); plain completion stays off; explicit params win. Training it into the weights instead is IDEA 19.
- Top open lever: chunk granularity. 1,024 B chunks pack ~8 unrelated facts, re-importing the multi-candidate failure inside one passage; e2e 0.270 vs the 0.445 ceiling.
- Note for any future grounded SFT: build_rag_corpus TEMPLATE puts context in a system turn; the route injects into the user turn. Reconcile before training on it.

## corpora

- rag_ui rebuilt for real (298 train examples + 37-fact held-out set at veritate_mri/data/eval/rag/rag_ui_test.json; the June bins were 0 bytes). grounded_ui_*.bin remain 0-byte husks.
- chrg / hansard / scotus zipped with license attribution (data/corpus/*_bundle.zip: 400M PD, 96M OPL v3.0, 21M PD) and registered in corpus_catalog.json as coming_soon with PLACEHOLDER URLs. Release = upload to COS, swap URL, drop the flag.
- facts/ + facts_pipeline/ untouched per instruction (no SFT with them yet).

## repo

- Ruff clean repo-wide (cleared the 3 deferred violations plus a duplicated decorator). mri suite 472 passed. documentation.md inference section gained the RAG injection bullet.

## cardinal optimization track (TODO, user-approved 2026-08-20)

Goal: wren2 (500M hybrid flagship) serves int8 on cardinal-01. Two work items, sequenced after the memory recipe settles (full detail in ideas.md IDEA 21):
1. **Hybrid+QAT pilot** (training side, cheap, first): 10M-class hybrid run with `qat_enabled` to prove the existing hooks (model_recurrent `set_qat`/`fake_quant_act`, qat.py int8 scheme) hold quality on the recurrent trunk. Gate: val within noise of the non-QAT twin.
2. **v13 int8 compute kernels** (engine side, the real gap): v13 today computes fp32 end-to-end — int8 storage exists but not int8 math. Write int8 matmul kernels for the hybrid forward, linux x86_64 (cardinal) first; weights int8, recurrent state stays fp32. Rule 25 applies: scalar reference + bitwise-identity check per kernel. Can proceed in parallel with wren2 training.

## ideas backlog (2026-08-18 sweep)

- IDEA 6 closed to successes.md: under-training was the fluency ceiling; wren_base val 0.7073 at 145k.
- IDEA 10 gate measured: speculative decode pays on RAG traffic (~2.2x fewer weight streams), marginal on chat; C-engine implementation queued RAG-path-first.
- IDEA 11 unblocked: the format set exists (`ifeval form`, 280 items); needs two matched 200M runs, compute sizing is the user's call.
- IDEA 19 mechanism 1 killed (see models); mechanisms 2 (unlikelihood trainer lever) and 3 (DPO) open.

## waiting on user

1. Deletion approval: data/corpus/_pg_cache/ and trainers/corpus/_pg_cache/ (30 Project Gutenberg source .txt, 22 MB each, already baked into mixed_chat); optionally the 0-byte grounded_ui bins.
2. COS upload + catalog release for the three corpus bundles.
3. Carpathian API key + model name (IDEA 18 stays blocked).
4. use_act_ckpt=False measurement still queued for the next pretrain launch.
