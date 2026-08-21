# worklog

The running record of work on Veritate: a one-line-per-day timeline, a plain-language
narrative of the chat models, and dated technical sections. This is the general work log.
It supersedes the two docs retired on 2026-07-08 (`overnight_run_log.md` and
`chat80m_tuning_journal.md`); their content is folded in here.

How to read it: the **timeline** below is the fast index (headlines, newest first). The
**plain-language** section explains the chat models with no jargon. The **dated sections**
are the append-only technical evidence trail that the research papers cite.

Companions: the research ledgers (`ideas.md` / `successes.md` / `failures.md`), the narrated
papers (`research/`), and the per-component docs (`developer_documentation/`).

## timeline (headlines)

- **2026-08-20 late (seed packs: 1,520 conversation seeds in 40 selectable topics, concurrency to 256):** Interview mode was drawing openers from the genre spec's `situations` list — **14 entries for `conversation`** — which is the real ceiling on corpus size. Measured the yield first: one seed gives **143 unique openers over 12 rounds (214 over 18, still climbing) at a distinct-5-gram ratio of 0.905 against the 0.90 floor**, and zero simhash near-duplicates, so ~150 usable openers per seed is the honest working figure. That sets the arithmetic: `mixed_chat` parity (~100k conversations) needs ~700 seeds, 500 MB needs ~1,500. SHIPPED `data/authoring/seeds/conversation.json` — **1,520 seeds across 40 topic groups**, zero duplicates, zero non-ascii, covering the conversational vertical broadly as the user specified: chatting and storytelling AND answering practical questions AND asking clarifying questions when a request is underspecified (`clarifying_vague`, `narrowing_down`, `admitting_uncertainty`, `multi_step_help`, `everyday_howto`). No code, no technical subjects. `readers/seeds.py` + `/teacher/seed_packs` expose them; a vertical is selectable only when its pack exists and parses, with `PLANNED_VERTICALS` rendering the roadmap disabled — a vertical with no seeds would otherwise fall back to the thin `situations` list and build a corpus about the wrong subject. Topic selection is persisted and the opener walk shuffles the seed list so every seed is used once before any repeats. **`LOCAL_MAX_CONCURRENCY` raised 4 -> 256** with a fixed powers-of-two choice list (2…256): the old clamp protected a laptop GPU but capped a 100k-conversation run at ~36 days on this 32-core / 275 GB box; at 32 it is ~4.5 days. The cost line now sizes a run from the selected seeds and warns when the request exceeds what those topics can carry. Verified end to end: a `pets`+`music` selection (76 seeds) at concurrency 32 produced on-topic conversations only. 1,265 tests pass.

- **2026-08-20 night (the terse-teacher diagnosis was wrong; two-pass 'interview' mode shipped and clears the gate):** User pushed back that the same model answers at length through the Carpathian API, and was right. Measured: asked a question AS ITSELF, qwen2.5:14b returns **2,380 / 2,296 / 2,457 B**; cardinal1:14b on the API returns **2,810 B**. Asked to WRITE A DIALOGUE it returns **120 B**. **A 20x gap on the same model and box** — the pipeline was never asking the teacher anything, it was asking it to script both sides, and models write scripts like screenplays. The earlier failures.md entry blaming model capability was replaced the same day. SHIPPED `teacher/interview.py` + `teacher/interview_job.py` + `/teacher/interview/start` + a third Distillation mode (now the default): pass 1 writes user turns, pass 2 ASKS each one and keeps the real answer, follow-ups chain to any depth — which also breaks the 7-turn ceiling every curated source has. Length is a per-turn blend by user decision (brief 0.20 / normal 0.55 / thorough 0.25) shaped at ask time, with whole-sentence trimming as a ceiling only. InterviewJob writes the SynthJob on-disk contract, so status/stop/samples/build were reused with zero change, and every conversation still passes RecordGate. End to end through the dashboard: **256-379 B median, 100% unique turns, 0 rejected of 8, variety 1.0, gate PASS**. Two real defects found by running it: (1) 8.3% of replies opened with 'Sure,'/'Of course,' against a 0.22% baseline in mixed_chat — ~38x, a register tic not noise, fixed by instruction plus a strip-and-recapitalise backstop (33% -> 0% on the captured data); (2) **a bug in my own audit** — assistant-register artifacts were counted across the whole file but normalised per assistant turn, so two USER turns opening 'Sure, I'm trying to decide between...' failed an otherwise clean corpus. Patterns now carry a scope; structural damage stays file-wide. Reference corpora unchanged by the fix (mixed_chat 4.5 -> 3.7 per 1k, still PASS). Dashboard restarted on 8001. 1,244 tests pass.

- **2026-08-20 later (Distillation tab shipped: the pipeline already existed and was buried in the Training tab; contention guard + acceptance gate are the new parts):** Survey first, and it changed the job: `/teacher/synth/*` and `/teacher/authoring/*` were already a complete distillation backend (byte-target call planning, per-record gating, ChatML packing, zip, catalog registration) with UI hidden behind the Training tab's action picker, and `body.training-active` **hid `#authorPanel`/`#synthPanel` outright** so it could not run during a training run at all. So this was consolidation plus two genuinely new pieces, not a new pipeline. SHIPPED: (a) **`/teacher/target_status`** — resolves the teacher's `base_url` host and reports whether that machine is training. Hosted API never contends; local teacher on this box contends while `trainer_runner.state()` says running; local teacher on ANOTHER box reports `training_active: null`, because reporting `false` there is a guess dressed as a fact. Never raises — a guard that 500s blocks the start it exists to advise (13 tests). (b) **`tools/corpus_audit.py`**, the acceptance gate, returned as `audit` from both build routes and rendered in the tab. Scores by unique turns and unique content bytes, never by file size, because `chat_5gb` passed every size check the platform had while holding 708 unique user turns in 5.14 GB. Four checks calibrated with headroom under the weakest surviving corpus: unique user turns >=0.95 (cogito 95.9%), unique content bytes >=0.85 (mixed_chat 99.2%), median assistant turn >=200 B (veritate_sft 242 B), artifacts <=5.0 per 1k (mixed_chat 4.5). **Two of my own thresholds were wrong on first calibration and the data caught them:** I had set artifacts at 1.0/1k from a per-pattern reading when the real total for the reference corpus is 4.5/1k, and I scored unique content against FILE bytes, which counts ChatML markers as duplication and failed veritate_conversation_v1 at 56.6% when its true turn-text uniqueness is 100%. Fixed both; the gate now passes all six real conversational corpora and fails exactly the two that should fail (`instruct`, 53 B median; `sft_idk`, 67.7% unique — repetitive by design, which is a judgement call, so the gate reports and does not block). (c) The tab itself: `#authorPanel`/`#synthPanel` moved, mode switch persisted to localStorage, job rehydration moved off the training flow picker into `_distOnTabActivated` (the move had silently broken refresh-mid-run restore), plan persisted server-side to `plan.json` so progress keeps its denominator across a refresh, and the two `body.training-active` display:none rules removed. **Live-run bug found and fixed:** `/teacher/synth/status` reports `completed` as RECORDS (lines in samples.jsonl) while `state.json` counts CALLS — my progress bar summed the two and would have read 26 of 79 where the truth was 7 of 79. Added `calls_ok`/`calls_failed`/`calls_remaining` and pinned the distinction in a test. Verified end to end against the live ollama teacher WHILE wren1_4 was training (guard correctly fired on the real run; job at concurrency 1; 21 records from 2 successful calls, 5 gate rejections for turn-count and schema; build produced a bin and the audit scored it). **Side finding worth acting on: qwen2.5:14b-instruct at the current spec produced a 68 B median assistant turn** — the teacher's default verbosity is well under the 200 B floor, which is the same failure that made `instruct` useless. Raise it in `corpus_spec.json` before any real distillation run. Smoke corpus and its catalog entry deleted; wren1_4 untouched throughout. 1,168 platform tests plus 68 new ones pass.

- **2026-08-20 (corpus purge: the chat data was duplicate-expanded, not dirty — 19 GB deleted, catalog cut 42 -> 31 entries, size ladders retired):** Audited every chat/agent/mcp bin at the TURN level (split on ChatML markers, hash each turn, count distinct). The finding is not messiness — artifact rates in the good bins are **under 1 per 1,000 assistant turns** (AI-disclaimer, canned refusal, mojibake, template leak, truncation all measured). The finding is **duplication**: `chat_5gb` is 5.14 GB carrying **708 unique user turns / 769 unique assistant turns = 376 KB of real text (0.01%)**, with one user turn repeated **1,298,507 times**; `chat_500mb` is a byte-identical prefix of the same 376 KB; `mcp_1500mb`/`mcp_150mb`/`mcp_15mb` share **124 unique user turns** (0.27% / 1.07% / 3.20% unique content); `agent_1500mb` holds 275,897 unique user turns in 1.5 GB (1.90%), `agent_150mb` 51,983 (4.32%). This confirms and sharpens the 2026-07-26 catalog note (which said 1,477 unique turns) — it is 1,477 turns TOTAL across both roles, and the ladder tiers are not independent corpora at any size. **The good data is genuinely good and there is only ~230 MB of it:** `mixed_chat` 218 MB / 371,214 turns / **98.8% unique user turns / 88% unique content bytes / median assistant turn 265 B / zero turns under 40 B**; `veritate_chat`, `veritate_conversation_v1`, `veritate_sft`, `wren_noloop`, `cogito` all 96-99% unique; `recall_curr` 98.5%; `chrg` 96.7%. Counter-example on the other side: `instruct` is 39% sub-40-byte assistant turns, and `chat_500mb`'s median assistant turn is **13 bytes** — even its unique content is too short to teach conversation. ACTIONS: (a) deleted 19 GB from `data/corpus/` (73 -> 53 GB) — the nine dead chat/agent/mcp bins, the abandoned 10.7 GB `the_pile_train.bin.part`, and three already-extracted `*_bundle.zip` leftovers; (b) rewrote `corpus_catalog.json` 42 -> 31 entries: dropped the nine dead stems, dropped the `py_code_100mb`/`js_code_100mb` lower tiers, dropped `the_pile` (53 GB) and `redpajama_v2` (214 GB) as unusable at consumer scale here, and **retired size ladders entirely** — `recommended_min_params`/`recommended_max_params` are now null on every entry (both fields were already optional in `index.js` and `_entry_skeleton`, so no JS or backend change was needed); (c) added a `coming_soon` **`agent`** stub (rebuild deduped via `tools/build_agent_corpus.py` with a per-conversation hash gate) so the agent topic stays populated, and a `coming_soon` **`mixed_chat`** entry — the best chat corpus on this install was never in the catalog at all, upload pending; (d) wrote `cos_delete_list.txt` at repo root with every COS URL, GitHub-repo file, and iCloud staging path that can now be deleted (~2.3 GB COS + ~68 MB GitHub). Tests: `tests/mri/test_capabilities.py` had the retired `chat_50mb`/`agent_15mb` stems hardcoded, repointed to `mixed_chat`/`agent`; 315 passed across `tests/corpus`, capabilities, chat-template alignment and both mix-planner suites. **Standing rule banked: a corpus is measured by unique turns and unique content bytes, never by file size.** Acceptance bar for any newly built or distilled chat corpus, set by `mixed_chat`: >=95% unique user turns, >=85% unique content bytes, median assistant turn >=200 B, artifact hits <1 per 1k turns — and dedup at GENERATION time with a hash gate, since every bin that failed here failed by resampling a small template pool.

- **2026-08-03 (product-key memory built, trained and kernelled on cardinal: 102x capacity at 3.90x faster decode with quality parity — and the measurement instrument was the real find):** Built `veritate_core/model_pkm.py`, a memory-layer FFN that replaces the dense up/down pair with a weighted top-k read over `sub_keys^2` learned value slots, wired as `global_ffn=pkm` / `trunk=hybrid_pkm` on the patched trunk. **The headline is a methodology finding, not the layer.** In eager PyTorch the layer measured **4.9x SLOWER** than dense at batch=1 (1387 vs 281 us/token); rewritten in C it is **3.90x FASTER**. Root cause measured directly on cardinal: `x*2` on a 320-vector costs **20.3 us** and a 320x320 matmul doing **102,400 MACs** costs **38.0 us**, i.e. PyTorch charges ~20 us per OPERATION and 102k multiply-adds cost only 17.7 us more than doing nothing. Dense FFN wins in eager for a reason unrelated to being good (3 fat ops); anything sparse, gated or conditional needs more ops and loses automatically. **Every batch-1 architecture comparison run in eager PyTorch has been measuring the dispatcher, so prior batch-1 architecture conclusions are re-openable.** Two-stage speed work in C, both arms held at identical optimization throughout (the fair-comparison discipline mattered — an earlier 4.70x shrank to 3.90x once dense also got AVX2): profiling showed the **top-k sorts were 52% of cost and the scattered gather only 8%**, and forcing the gather fully contiguous gained just 7%, killing my own prediction that random access would dominate. Replacing the ranking with a **firing threshold** (AVX2 compare + movemask, 0.586 us vs topk 8.778 us = **15x** on that stage) took PKM int8 139.3 -> 69.6 us/token; adding a 4-row-blocked int8 integer-dot matvec to BOTH arms landed the final **dense 199.6 vs PKM 51.2 us/token, 102x capacity, 6.49x fewer bytes read**. Quality settled by a matched A/B trained end to end **on the clamped 800 MHz i7** (5m shape, enwik8, 10,000 steps each): dense val **1.299641**, top-k PKM **1.294813** — PKM ahead at every step from 7500 on, but 0.37% on a single seed so the reportable claim is PARITY (agent_roe seed rule). Capacity is unexploited: 12.7x the params only matches, because enwik8 at 95 MB cannot fill it. Threshold gating ported back into the PyTorch layer (hard gate forward so it matches the kernel exactly, straight-through backward so `theta` learns — verified real gradient and genuinely variable firing at ~12.8 of 32 candidates), new trunk `hybrid_pkm_fire`, third arm training. **Four negatives banked, each killing a plan I had already written down:** PKM is NOT a training-speed lever (dense 592M 2542 tok/s vs PKM 2309M-capacity 983 tok/s on MPS — capacity costs training time, it does not save it, so my "500M in 4 days via PKM" estimate was wrong and is retracted); sparse gradients on the value table LOSE on unified memory (0.209 vs 0.188 s/step at 537M capacity despite only 4.8% of slots touched); `n_chunks` does not generalize across shapes (+68% at chat200m, but 0.86x/0.70x at 593M b24 — splitting an already-large batch starves the GPU); and Muon is an **11x tax** on the clamped CPU (9.5 vs 0.81 s/step) though it remains the GPU default. Honest 500M budget now measured rather than extrapolated: **593M on M3 Ultra at b24 bf16 = 5541 tok/s = 24.8 days Chinchilla**, so the sub-week target needs `torch.compile` plus Muon convergence plus distillation, and n_chunks/PKM are NOT part of that path. Two of my own bugs were caught only by parity checks and are worth remembering: an AVX2 kernel with broken maddubs sign handling (relative error 1001) and a threshold calibration that silently measured an EMPTY gather and reported a beautiful fake 6.3x. Platform work alongside: rule-11a `probe_module()`/`probe_weights()` contract added to FFN/MoEFFN/ProductKeyMemory so the dump suite, `diff`, `pruning` and `export` stop reaching into `ff.up`/`ff.down` (dump failures 66 -> 0 in a live run; pruning now refuses non-prunable trunks with a clear message, export joins the rule-40b variant refusal), and a missing `os.makedirs` in `corpus_sync.py` that made every HF-stream and zip corpus install fail on a fresh box with no `data/corpus/`. Artifacts: `veritate_core/model_pkm.py`, `models/{pkmctl_5m,pkmmem_5m,pkmfire_5m}/`, C harnesses in cardinal `/tmp/pkm_{bench,bench_i8,bench_thresh,profile,mv,fair}.c`. Nothing staged or committed.

- **2026-07-27 evening (wren instruction-following: cause found and measured; SFT v1 a net NEGATIVE, v2 tests the fix; 5 platform defects fixed incl. IFEval scoring framing not obedience):** `wren` (270M hybrid, 5.4B tokens, forked from chin200m@55000, trained to 58500 on fortis) is not broken: identity, factual recall and IDK calibration all work, and it is coherent in ChatML. It fails ONE thing, instruction execution. Measured: **IFEval 11.1% (4/36) with `item_count`, `word_count`, `starts_with` and `json` all at exactly 0%.** Cause is arithmetic, not mystery: its chat data was **0.4% instruction-following** (160 of ~44,500 authored records), because authoring genres with STRUCTURAL requirements yield far less than free-form ones on `qwen2.5:14b-instruct` — measured over two 22k-record jobs: jokes/writing 82-194%, conversation 39%, `instruct` 64%, cogito 15%, **`format_constraint` 1.3%, `carryover` 1.4%, `grounded_read` 0.5%**. A genre weighted 0.10 that yields 1.3% lands at 0.4% of the corpus and NOTHING surfaced it: per-genre yield is in `state.json` but was never read against planned calls. **Standing rule banked: planned calls are not produced records; read `authoring.per_genre` + `authoring.rejects` before trusting any mix.** Built the missing capability as data: new `instruct` genre (compose/enumerate/transform/summarize/extract/translate/compute/compare/classify/define/steps), `min_turns: 2` to keep structural demand low. Three self-inflicted errors caught mid-run: (1) the teacher **regurgitated my brief's examples verbatim** — one prompt 24x, and one of them ("write one sentence about rain") was also an EVAL prompt, which would have shown a fake improvement; fixed by stripping quotable examples and adding a `dedup_user_turn` gate flag (the existing opening cap keys on whole-record text, so the same instruction with a different answer passed freely); (2) I chased VOLUME while **diversity** was the binding constraint — distinct-5-gram decayed to the 0.90 floor at ~2,200 records; the fix was replacing 15 abstract `situations` with **40 concrete subject domains** (14 voices x 40 domains = 560 combinations) plus temperature 0.7->0.95, which took variety 0.9046 -> **0.9827**; (3) I was generating families the eval does not test (389 `define-in-one-line`) while `json` and `avoid-a-letter` had **zero** coverage — rebalanced the voice pool toward verifiable families. Result: `instruct` = **4,263 records / 1.02 MB / variety 0.983 / 0-of-36 eval contamination across 74 MB** of all four mix corpora. **the first SFT attempt (`wren_sft`, instruct 0.50 + veritate_sft/sft_idk/skills, LR 3e-5, 180 steps, batch 16, loss_mask=assistant) was a NET NEGATIVE and is not shippable.** Form compliance rose 55.6% -> 63.9% (yes/no-first **0/4 -> 4/4**, "List three colors" gibberish -> "Yellow, green, blue", "3 apples eat 1" -> "2") but strict IFEval stayed **flat at 11.1%** and it REGRESSED what wren was good at: "How many days are in a week?" **"7" -> "1) 2) Pegged to base."**, and 2 of 5 identity prompts degraded (temperature 0, so not sampling noise). Two causes, both mine: **no replay data and no cogito in the retention mix** (I named catastrophic forgetting as the top risk then protected the wrong things — wren's facts come from its pretraining `fineweb_edu`/`wikitext103`, none of which was in the SFT mix), and **template intrusion** from 344 `numbered-steps` records (8%) teaching it to answer any uncertain question with a numbered list. **Third finding, about the instrument: 47% of my eval rules required the correct ANSWER (`contains`, `starts_with` pinned to the true value), so they grade reasoning, not obedience** — which is why a real form gain scored flat. Split out `ifeval_form.json` (26 items, every rule answer-independent, `starts_with_yes_or_no` passes on either answer) selectable via `ifeval_set`. Platform defects fixed with tests: `_trim` cut only COMPLETE stop markers so every buffered reply leaked `<|im_end|`; **IFEval fed raw prompts with no chat template and no decode stop, so it scored the framing rather than the instruction following and was unusable on any chat model**; the authoring gate split teacher replies on newlines and lost every record carrying a raw newline in a string (+7.1% records via `iter_json_objects` with `raw_decode`/`strict=False`); 5 new checkers (`word_count`, `item_count`, `contains`, `starts_with`, `forbidden_words`). 912+ tests pass, ruff clean. **BOTH of my recipes FAILED, and the answer was already in this repo.** Retry (fineweb replay 0.15 + cogito 0.08, instruct 0.35, LR 2e-5, 160 steps) scored **form 30.8% against wren's 34.6%** on the answer-independent set: `starts_with_yes_or_no` 20% -> **100%** and `sentence_count` 25% -> 50%, but `word_count` **100% -> 0%** and `forbidden_words` 100% -> 0% because the SFT made the model MORE VERBOSE, so it now overshoots word ceilings it previously passed. Retention recovered only to 5/7 (still lost 'days in a week' -> '14 days'). **Then I read `successes.md` and found 2026-07-20 IDEA 8 already solved this on wren's own parent lineage: `sft_instruct_v1` on a chin200m@55000 fork, +50-58pt format lift, ZERO bleed, val drift +0.002.** Its recipe is the INVERSE of mine: **dose 0.15 (not 0.35-0.50), replay 74% (fineweb 0.42 + chat_500mb 0.22 + wikitext103 0.10), 3500 steps (not 160), 4 focused families (not 14).** Low dose + heavy replay + long training; I ran high dose + light replay + short training and got forgetting both times. **Process failure worth banking: read `successes.md` / `failures.md` / `ideas.md` at session start before designing an experiment — CLAUDE.md points at them and a week-old entry documented the working recipe for exactly this task.** Relaunched `wren_sft` in place (failed attempts deleted, no version suffixes) with the ledger recipe applied to the broader authored `instruct` corpus: `instruct:0.15,fineweb_edu:0.42,chat_500mb:0.22,wikitext103:0.10,skills:0.05,sft_idk:0.03,cogito:0.03`, 3500 steps, LR 2e-5 constant, ckpt_every 500. Measured baselines to beat: **wren form 34.6%, mixed-set IFEval 11.1%, retention 7/7.** `wren` itself is untouched at `models/wren`. Artifacts: `trainers/corpus/instruct_*`, `trainers/corpus/cogito_*`, `veritate_mri/data/eval/samples/ifeval_form.json`, `models/wren_sft/`, `models/wren_sft/`, `developer_documentation/training/deep_eval_suites.md`, genre-yield table in `developer_documentation/corpus/authoring.md`.
- **2026-07-25 evening (IDEA 9 growth experiment COMPLETE: Net2Net widen = real 3.6x step-lever, but 1.66x compute LOSS as run; role binding dead across 4 sizes -> failures.md):** Ran the pre-registered grown-vs-scratch experiment. A GROWN = `conceptsho_10m`@2000 (15.99M) ffn-widened 1280->2560 to 25.74M then 1000 stage-2 steps; C SCRATCH = same shape random init, 1000 steps; identical corpus/batch/LR, eval grid 50, gates fixed in ideas.md BEFORE C ran (threshold as a PROTOCOL — T = A's val at its 250th stage-2 step = 0.121434 — not a hand-picked number). **Function-preserving growth verified on the hybrid trunk for the first time** (the upcycle kit targets the deleted veritate_800m dense plugin; its smoke + model-level wrappers are dead, state_dict fns are fine): new ff.down cols exactly 0.0, old cols bit-exact, and resume loss 0.0917 vs parent 0.0923 — no spike. **Steps-to-target PASS as a whole-curve shift:** 0.30 -> 50/450 (9.0x), 0.20 -> 100/500 (5.0x), 0.15 -> 100/550 (5.5x), T -> 250/900 (**3.6x**); C never catches A in 1000 steps. **But the falsifier registered beside the gate FIRED: total compute A 38.42 G vs C 23.17 G = A cost 1.66x MORE** (params x steps) — the 3.6x is in LARGE-model steps only and stage-1 is not free. Break-even stage-1 budget = 1046 small steps; stage 1 saturated ~1250 and I ran 2000, overspending ~950 and flipping a likely win into a loss. **Operative rule for the flagship: stop every curriculum stage the moment its val curve flattens, then widen.** Role binding (n=91, chance 6%): subject-role 11% (A) / 29% (C) — and the decomposition is the real finding: **A answers with the OBJECT regardless of question (78% of "who" items), so its headline "what 81% / held-out 60%" is a constant emit-the-object policy, NOT binding; C echoes the noun out of the question** ("Who sees the boy?" -> "The boy does."). Two shortcuts, no relation. F4 now fails at 10M/26M/122M/200M/800M and under BOTH data regimes -> data lever and small-scale capacity lever both killed in failures.md; only relations-in-the-index (F5, 36->100%) or an explicit binding primitive remain. Three self-inflicted errors caught and fixed mid-run, each recorded: (1) the first design continued the grown model on the SATURATED stage-1 corpus (killed at ~12 min — a saturated benchmark cannot measure capacity); (2) stage 2 ALSO floored at 0.121 despite 3x block diversity, because **a procedurally-generated corpus floors at its GENERATOR's complexity, not its byte count** — unique strings are not information, so capacity work must use real text; (3) the held-out set had **7/91 answers leaking into train** (one block type drew objects from a list containing the animate entities) plus a "They pulls" agreement bug — both fixed, corpus rebuilt, arm A restarted. Standing method rule banked: **score WHICH entity a wrong answer names, never just accuracy.** Artifacts: `models/conceptsho_grown/`, `models/conceptscratch_10m_w2/`, `trainers/corpus/concepts2_ho_*`, `--stage 2` in `build_curriculum_corpus.py`, `10m_w2` size added to `trainers/veritate_10m/manifest.json` (pure addition, upstream mirror pending).

- **2026-07-25 (F6 — the developmental data lever does NOT crack F4; role binding fails at 10M even in-distribution, and MORE training makes it worse):** Stage-1 of IDEA 9 measured to completion with the held-out split the gate demanded. `conceptsho_10m` (15.9M actual, hybrid trunk, Muon, qat OFF, batch 64 at the benched M3 knee) trained the full 2000 steps on `concepts_ho` — the child-concept curriculum where 30% of (subject,object) pairs appear DECLARATIVE-ONLY, their who/what questions held out as the test set (`build_curriculum_corpus.py --holdout-frac`, deterministic Random(SEED+99); held-pair answer strings verified 0/12 in train). Final val 0.092 (near-memorization). **Result, n=12/cell: CONTROL (pairs seen WITH questions) who 50% / what 25%; HELD-OUT who 17% / what 0%.** Confounds killed: fp16 and int8 exports score BIT-IDENTICALLY (not quantization), and the loaded model was probe-verified before scoring — a first pass silently scored `chat800m` because the C backend kept the previous model through a load call, so **always probe-verify the loaded model before trusting an eval**. The failure MODE is the finding: `"The dog chases the boy. Who chases the boy?"` -> `"The boy does."` — it ECHOES the noun out of the question; `"What does the dog chase?"` -> a fixed salient entity ("The boy" / "The cat") independent of context. Three takeaways: (1) **F4 replicates at 10M on data explicitly built to break it** — stating each event four ways (active/passive/who/what) was NOT sufficient, which supports "missing architectural primitive" and weakens "it's just a data problem"; the stage-1 "big if true" gate did not fire. (2) **Low loss != binding** — val 0.092 with CONTROL (in-distribution!) at 50/25 means byte-loss is minimized by surface statistics while role QA stays broken; do not read loss as competence on a drill corpus. (3) **Longer training made it WORSE** (CONTROL who 100% @ step500 -> 50% @ step2000): gradient descent on next-byte loss actively HARDENS the shortcut. Does not kill IDEA 9 — it gives the growth experiment a pre-registered baseline to beat (50/25 control, 17/0 held-out); if a grown 10M->N also lands ~17/0 held-out, growth is not a role-binding lever and the fix must be architectural (explicit binding, or relations in the INDEX per F5) -> failures.md. Full evidence + caveats in `ideas.md` IDEA 9 "STAGE-1 RESULT". NOTE FOR OPERATOR: this file currently carries unresolved merge-conflict markers at lines ~18-31 (`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes`) from a prior stash — left untouched, needs a human resolve.

- **2026-07-20 (IDEA 8 layered capability-SFT campaign COMPLETE, 3/4 pass + int8 QAT cleared):** Ran the full 4-corpus x 4-fork chain overnight through the dashboard: `chin200m_grd1` (grounded read-off-page), `chin200m_mt1` (multi-turn callback), `chin200m_inst1` (format constraints), `chin200m_prose1` (long-form prose) — each 3500 SFT steps at 2e-5 WSD on a fresh fork of chin200m@55000, ~2h46m per run, sequential (no-parallel-training rule). Verdicts pinned to successes.md / failures.md against pre-registered ≥40pt capability gate + wrong-context bleed battery + needle-copy regression sweep: **grd1 ✅** 12/12 grounded transfer on fresh invented entities, 4/4 bleed clean, +75pt (cleanest pass; move to successes.md 2026-07-19); **mt1 ⚠️** 11/12 callback transfer (+92pt) but callback-shape "you said …" phrase leaks into single-turn contexts (successes.md 2026-07-20 with honest boundaries; combined-stack tuning may want mt dose 10% not 15%); **inst1 ✅** 8-9/12 format follow (+50-58pt), 4/4 bleed clean, no regression (cleanest capability SFT of the four; successes.md 2026-07-20); **prose1 ❌** 6/12 (+33pt, UNDER the 40pt gate) — narrow-template corpus taught verbatim phrase-echo ("It seems to belong to the shape of an ordinary day" across unrelated prompts), prose-shape leaked into instruct_single_sentence probe (returned 4-sentence prose overriding "one sentence"); moved to failures.md 2026-07-20 with retry conditions (broaden templates to 200+ real varied examples/family or teacher-generated assistant text). **int8 QAT for Cardinal cleared by POST-HOC EXPORT (no QAT training needed):** grd1 re-exported int8 dropped 541 MB → 277 MB (51% smaller), all 12 target-skill greedy replies BYTE-IDENTICAL between fp16 (pytorch) and int8 (C engine); the quantizer is high-fidelity enough for a single-family SFT fork. Cardinal wall-clock unmeasured (physical box) but projected ~2x decode on bandwidth-bound i7-9700T. Full campaign artifacts: 4 forked model dirs under `models/chin200m_{grd1,mt1,inst1,prose1}/`, 4 SFT corpora under `trainers/corpus/sft_{grounded_read,multiturn,instruct,prose}_v1_{train,val}.bin`, 4 focused probes + regression sweep under `temp/probe_{grd1,mt1,inst1,prose1,fork}.py`, generic ChatML packer + tests at `veritate_mri/tools/build_sft_corpus.py` + `tests/mri/test_sft_corpus_builder.py` (6/6 pass, byte-deterministic seed 20260719). All held for operator per "no git actions". Trainer arg gotcha caught + documented in ideas.md IDEA 8 resume trail: `--resume` is a STRING (model dir name), not a boolean.

- **2026-07-19 evening (all 4 capability SFT corpora BUILT + PACKED; chin200m_grd1 SFT LIVE):** Same-day follow-up: after the corpora / handoff / decision above landed, packed the remaining 3 corpora (multiturn 1500 conv from staged callback jsonls, instruct 1500 pairs, prose 1000 pairs — all invented/mundane so recall is not poisoned) with the same byte-deterministic seed 20260719, and fired the first SFT via dashboard. Fork `models/chin200m_grd1/` from chin200m step 55000 through `/models/fork`; then `/trainers/run` with veritate_200m at 15% sft_grounded_read_v1 + 80% pretrain anchor + 5% sft_idk, base_lr 2e-5 WSD → 2e-6, warmup 200, total_steps 58500 (3500 SFT steps, ~2.5h ETA). Verified stepping at ~38k tok/s (matches base bench), loss 0.55-0.62 in first 75 steps as the new distribution comes on. Trainer arg gotcha caught: `--resume` is a STRING (model dir name), not the boolean stored in old config files — first fire returned exit 2 in 1.2s; second fire with `resume: "chin200m_grd1"` runs clean. Mt1 / inst1 / prose1 runs queued behind grd1 per `feedback_no_parallel_training`. All 4 bins + this session's tool + tests + generators are staged but uncommitted per user "no git actions". Detail in the ideas.md IDEA 8 progress ledger.

- **2026-07-19 (chin200m layered-SFT campaign — identity route ABANDONED, capability SFTs opened; sft_grounded_read_v1 PACKED):** Previous session died mid-campaign after two identity SFT forks that both under-performed (chin200m_ident 2/12 probes, chin200m_ident2 3/12 but with wrong-context bleed: "Call me Veritate" replying to garden-tips; "Carpathian." replying to ocean prose). Root-cause user + this-session read: 40h pretrain baked a "no fixed name" pattern into the argmax path and 500-1000 SFT pairs at 3e-5 cannot move it without eroding pretrain skills — same 2026-07-08 failure signature. DECISION LOCKED (2026-07-19): identity is a **system-prompt** problem, not an SFT problem — ship the serve-layer persona ("You are Veritate, made by Carpathian.") the way the July chat200m did, do NOT run more identity SFTs. Empathy + engaged SFTs also skipped on the same principle (register tuning fights the pretrain, low ROI). Remaining campaign = 4 capability SFTs that teach skills the model LACKS (do not fight the pretrain): grounded_read (RAG read-off-page), multiturn (callback), instruct (format constraints), prose (long-form). QAT int8 for Cardinal is a low-priority PARALLEL track after the 4 SFTs. PROGRESS this session: sft_grounded_read_v1 generator ran clean (2000 pairs, 4 families: single_fact 700 / two_facts 600 / compound_fact 500 / unstated_fact 200 — invented entities only, no real-world facts to poison recall), packed via new generic `veritate_mri/tools/build_sft_corpus.py` (6/6 tests pass, byte-deterministic seed 20260719) into `trainers/corpus/sft_grounded_read_v1_{train,val}.bin` (train sha 5998b72c…, val sha a238eb4b…, 1900/100 split). Next actions IN ORDER: (a) fork chin200m at step 55000 into `chin200m_grd1`, (b) `/trainers/run` with sft_grounded_read_v1 at 12-18% + pretrain anchor + chat_500mb + hard needle-copy gate, (c) probe against alien-fact read-off-page + needle-copy regression, (d) ledger to successes/failures, (e) then build sft_multiturn_v1 (some callback jsonls already staged in `temp/sft_multiturn_v1_gen/`), (f) sft_instruct, (g) sft_prose_v1, (h) QAT. All training launches via dashboard `/trainers/run` per rule 24a; forks via `/models/fork`. Do NOT launch two trainers at once (see `feedback_no_parallel_training`).
- **2026-07-22 (F1-F5: the RAG-writer architecture, measured — reader floor <=200M, ZERO relational role binding at any size, relations belong in the index):** Direction reset by Sam: not a general model, not a coding model — the smallest possible strong CONVERSATIONALIST with all facts external (RAG / trillion-char context). Five experiments, each falsifying or refining the last, all on int8 models via the veritate engine (harnesses in scratchpad, full evidence + falsifiers in `ideas.md` IDEA 8 (consolidated 2026-07-23 from the retired `ideas_mirach.md`)). **F1:** grounded slot copy is 100% at 200M / 97% at 800M — 4x params buys NOTHING; accuracy collapses with stuffed chunks (1->100%, 2->87%, 3->60%, 5->53%, 8->30%). **F2 (negative):** the model is a great READER but a poor JUDGE — 36% of irrelevant chunks get a confident fabrication instead of a refusal, one-chunk-at-a-time + confidence picking was WORSE than stuffing (35% vs 50%), and confidence did not track correctness; prompting cannot fix it. **F3:** a ~30-line IDF lexical retriever + the 200M reader = 100% on the K=8 task the model alone gets 40% on => `system acc ~= retriever_precision@1 x reader_acc(1 chunk)`, reader_acc ~= 1.0, so ALL system error is retrieval (caveat: synthetic unique entity names flatter lexical retrieval; expect 70-90% real). **F4 (the big one):** the model has ZERO relational role binding — given "P founded the E Summit", asked "which summit was founded by P?" it answers P, not E: OBJECT-role 0% at BOTH 200M and 800M across 4 phrasings (0/20/7/7%), and the 100% on the SUBJECT question is a FALSE POSITIVE (it always emits the person). Scale does not fix it => a missing architectural primitive (no variables/registers), which QUALIFIES F1: "reading" = slot copy (works) + relevance judging (broken) + role binding (absent). **F5:** route around it — store facts as (subject, relation, object) triples so a hop is a DB join outside the model and the LM only verbalizes the one resulting fact: multi-hop 36% -> 100% with the same 200M; honest cost is that this MOVES relational extraction to index-build time (offline, batchable), it does not eliminate it. Resulting runtime design: reader <=200M, retriever does relevance, index does traversal, LM verbalizes one fact — **next spend is index construction quality, NOT model size.** Also launched `chat_80m` (122M actual, hybrid, conversation+grounding, NO code, 51k steps @ ~18.6k tok/s) to probe below the 200M floor and confirm the role-binding failure is structural. Detail in the 2026-07-22 section.

- **2026-07-21 (capability SFT `chat800m_v2` + M1 role-masked loss invention + INT8 green shrink; mission = green AI):** Sam reframed the goal to green AI (smaller-but-better, runs anywhere, general: code/talk/marketing) and asked for a NEW invention ledger — created `ideas.md` IDEA 8 (consolidated 2026-07-23 from the retired `ideas_mirach.md`) (separate from `ideas.md`). Shipped **M1 role-masked (assistant-only) loss** to `trainers/common/vanilla_trainer.py` (opt-in `loss_mask=assistant`, default off, backwards-compat, upstream mirror pending): the trainer had been computing CE on every byte including the user's questions; now masks non-assistant target positions to -1 (all loss paths already honor ignore_index=-1), per-row marker-gated so raw replay text stays fully active (else replay loss zeroes and forgetting returns). Zero throughput cost, standalone-tested. Ran `chat800m_v2` (fork of chat800m_chatml step_46500 -> 49000) on an expanded ChatML mix (chat_500mb + code_qa_100mb + mixed_code_qa_200mb + sft_idk, 65% capability) + fineweb/owt replay (35%), constant LR 1.5e-5, loss_mask on. Result: general capability preserved AND partly recovered vs the format-only run (grammar 0.72->0.82, reading 0.748->0.762 back to base; ppl_floor +0.4%); NEW abstention works ("what did I have for breakfast?"->"I don't know"); facts/chat/format solid; code still weak (rambles instead of writing code) and marketing weak (no marketing corpus exists — real gap flagged). **INT8 green shrink:** the hybrid model exports v13-hybrid INT8 (990 MB, ~4x smaller than fp32) and plain PTQ generation == fp16 -> QAT not needed at this size (measure before optimize). Only INT8 has a real arm64 kernel; ternary is disk-only and int4 has no exporter, so the MRI-guided mixed-precision QAT idea (ideas_mirach M2) is deferred until a mixed-precision serving path exists. Also downloaded code_qa/mixed_code_qa/sft_idk from COS. Detail in the 2026-07-21 section.

- **2026-07-20/21 (chat800m ChatML SFT: replay recipe beats catastrophic forgetting, format remapped, model `chat800m_chatml` serving):** The finished `chat800m` hybrid pretrain (45,000 steps, val 0.761) was SFT'd from legacy `<|user|>` framing onto canonical ChatML (`chat_500mb` from Carpathian COS). First attempt (pure `chat_500mb`, constant LR 5e-5) CATASTROPHICALLY FORGOT: train loss cratered 0.82->0.08 in ~220 steps (below the label-smoothing floor ~0.47 AND below English entropy = collapse, not learning), and step_45500 eval dumps confirmed reading perplexity DOUBLED (+98%), reasoning recall -91%, deduction -100%, grammar -19%. Loss curve looked "great"; the eval dumps caught the disaster — lesson: trust eval dumps over loss for SFT health. FIX = replay mix: `chat_500mb:0.5,fineweb_edu:0.25,openwebtext10g:0.25`, constant LR 1.5e-5, 1,500 steps (45000->46500), ckpt+eval every 150 with a per-checkpoint perplexity-forgetting tripwire. Result: forgetting eliminated (ppl +1% vs +98%, reading -2%, reasoning recall +9%), format remap fully succeeded (emits `<|im_end|>` and stops cleanly ~4/5 prompts; "Capital of France?"->"Paris.", "Hello, how are you?"->"I'm functioning normally. How are you?"). Weak on creative/structured (haiku) + invents a name (identity anchor excluded) = 800M byte-level ceiling, not a recipe bug. Model `chat800m_chatml` (forked from base step_45000, base untouched), marked chat=trained, exported v13-hybrid fp16 .bin, serving via veritate engine. Two resume gotchas found: LR schedule uses ABSOLUTE step (a decaying schedule resumed near its end pins LR at min — use `constant` for SFT-on-resume); resume does NOT rewrite `training_args.corpus` (config still lists the old mix — stale provenance string, no chat_v* data exists). Also purged the 4 legacy chat corpus bins (5.4 GB) + verified zero copies anywhere on disk/volumes/iCloud. Detail in the 2026-07-20 section.

- **2026-07-17 evening (warm-models pinning feature shipped + cold-prefill fully attributed):** Two follow-ons to the TTFB campaign. (1) NEW `warm_models` setting + warm subprocess pool: pinned models each keep a live C subprocess (`cfg["C_WARM"]` dict, ~8 functions in backends_routes, no manager class), eager-spawned at startup and never closed on model switch, so switching between pinned models is instant (no respawn) and the first request is already warm. Settings-tab panel (multi-select of .bin models + size/RAM readout), status shows which are resident, restart/rebuild hooks close the pool to avoid orphans. Verified on cardinal: chat80m + chat800m both resident at startup with zero requests issued, engine PIDs UNCHANGED across 3 chat80m<->chat800m switches (no respawn), chat800m warm TTFB 640-783ms vs 5238ms cold. tests/mri/test_warm_pool.py 8/8; 16 files mirrored byte-identical (12 changed + 3 new + c_engine unchanged-carry), no git. (2) chat800m cold prefill ATTRIBUTED (throwaway-instrumented, reverted): recurrent GLA blocks = 70% at n=60 / 65% at n=256 (rec matvecs = 86% of that), local attention 30-35%, everything else ~0. Two levers ranked: MRI/traced prefill batching (CHEAP, bitwise-safe, no consumer reads prompt-position trace so the letter-timing view can batch the prompt like the fast path -> collapses MRI TTFB to serving TTFB) FIRST **[SHIPPED same day: model.c forward hybrid branch now batches [restored..n-2] untraced then traces only pos n-1; emitted MRI frames byte-identical old-vs-new across 2 models x 3 prompts, MRI TTFB 1.43-1.69x and now == fast-path TTFB, fast path unchanged; tests/engine/test_prefill_batch.py +traced-parity test; 3 files mirrored]**; chunkwise-parallel recurrent prefill (measured 2.09x on the recurrent matmuls @7T, ~1.4x overall cold prefill, tolerance-gated, grows once the clamp lifts) SECOND. Calibration cache confirmed persisting per model. The 800 MHz firmware clamp remains the unaddressed 4x elephant (adapter swap did NOT lift it: base_frequency=800000, no_turbo=1 still). Detail in the 2026-07-17 section.

- **2026-07-17 (TTFB campaign on cardinal: first byte root-caused + 3 rounds of verified fixes, spawn 4.5s -> 0.7s, all mirrored):** The "1500ms to first byte" complaint decomposed with hard numbers on cardinal (Opus 4.8 agents building, lead overseeing): warm TTFB is 60-93% sequential hybrid prefill (TTFB ~= prompt bytes x per-position cost — the dashboard letter-timing frames show it directly), cold TTFB adds .bin fread + a 1.5s per-spawn thread-calibration burst, and `state_cache_store` (multi-MB disk write) sat on the critical path before the first byte. SHIPPED (byte-identical outputs proven each round, engine sha-compared): **(1) batched prefill ON in production** — the 2026-07-13 `VERITATE_PREFILL_BATCH` feature was never enabled; c_engine.py now spawns with the measured-best B=32 (B=64 is WORST on chat800m: cache overflow; win bounded by local/recurrent block ratio, so chat80m 1.3-1.5x, chat800m 1.06x); **(2) state-cache store moved after the first frame flush** (`model_store_state_cache`, engine + loop change, restore path untouched — repeated/shared-prefix prompts still restore at 156-327ms TTFB = 10-25x); **(3) parallel weight load** (job-list + pool fill at ~3.7 GB/s vs 0.7 GB/s single-thread first-touch-fault floor: chat800m load 2937 -> ~610ms) + **(4) calibration result persisted** per box+shape in the state-cache dir (1497ms -> 0.1ms on hit): chat800m spawn-to-ready 4453 -> 714ms, chat80m 897 -> ~120ms, model-switch reload cost collapsed accordingly; **(5) pytorch checkpoint load `mmap=True`** (raw torch.load 14-25s -> 0.5s; the recurring idle-unload reload tax). Honest residuals: chat800m cold-prompt prefill (~6s at 250B) is the by-design sequential recurrent scan — the serial-floor lever named on 2026-07-13 stands; pytorch kv-mode blocked on `VeritatePatched` lacking a streaming-state cache (round 3 probing it under a strict byte-identity gate). All diffs mirrored byte-identical into this repo (9 code files + 3 docs), no git touched either side. Detail in the 2026-07-17 section.

- **2026-07-14 (corpus library expansion: 7 corpora built + staged for COS, zip_bundle install format shipped, MCP corpus family created, size-ladder decision documented):** All four missing library tiers built deterministically (chat_500mb, chat_5gb via the existing builder + restored 30-book PG cache; agent_150mb, agent_1500mb via NEW `build_agent_corpus.py` — Hermes tool-calling over the real runtime toolbox with computed-not-invented results and error-recovery turns), plus a NEW 3-tier MCP family (`build_mcp_corpus.py`: JSON-RPC session transcripts + protocol Q&A + doc passages + Hermes-over-MCP-tools; mcp_15mb/150mb/1500mb). SIZE DECISION documented in `developer_documentation/corpus/library_ladder.md`: behavior corpora (chat/agent/mcp) cap at 5 GB / 1.5 GB / 1.5 GB — a 1T+ model needs diversity, not bytes; format-learning saturates and knowledge scale belongs to the facts corpora, so the top tier of each family serves 1.5B through 10T. PLATFORM: new `zip_bundle` install format in corpus_sync (one COS-hosted zip holds train+val; download → extract → zip always deleted; sha256 verifies the EXTRACTED bins), `coming_soon` flag moved from a hardcoded index.js stem set into the catalog entries themselves (backend passthrough + install refusal; release = swap PLACEHOLDER URL for the real COS link + drop the flag, one JSON edit). All 7 corpora zipped into `~/Library/.../Mirach-Corpuses/{chat,agent,mcp}/` with an upload manifest; bins also installed locally in `trainers/corpus/` (trainable now). Trading extension: `crypto` (34 GB CSVs) zipped to `trading_datasets/crypto.zip` (6.8 GB, sha in manifest) — its download flow already unzips + deletes archives, zero code needed; honest residual: `crypto_extra` (41 GB) has NO local data on this box (empty source dir), cannot be staged from here, stays `url: null`. Tests: 7 new (zip_bundle extractor, coming-soon refusal, builder determinism + frame validity), tests/mri 149 passing (the 4 test_models_route failures pre-date this work, verified against a stashed tree). Held for operator: COS uploads + placeholder swaps, server restart to activate zip_bundle, git.

- **2026-07-13 evening (cardinal becomes the old-hardware inference testbed: 800 MHz clamp root-caused, persistent prompt/state cache SHIPPED — repeated prompts 31x, batched+threaded prefill 1.78x cold, all bitwise-gated):** The "why does cardinal train slow" question answered with hardware forensics, then converted into two shipped engine features. ROOT CAUSE: the OptiPlex 7070 is firmware-clamped to 800 MHz on all 8 cores (HWP `base_frequency=800000`, `no_turbo=1` BIOS-locked, package 41°C = not thermal) — the classic Dell power-adapter/BIOS clamp; training was compute-bound at 1/4 clock. Decode is NOT: thread sweep 1→7 gave only 1.33x (24.6→18.5 ms/byte) and a bandwidth probe (~4 GB/s single-core numpy read; engine at ~12 GB/s effective with 7T) shows chat decode on this box is RAM-bandwidth-bound, likely single-channel — so the decode levers are bytes (int8) and reuse (cache, batching, drafts), not clocks. SHIPPED, every stage env-gated + byte-identical when off: **(A) persistent prompt/state cache** (`state_cache.c` + `fsutil.c` shim; snapshots the six resumable `hybrid_t` fields post-prefill keyed by twin rolling hashes over model_id+tokens, longest-prefix restore, LRU eviction, atomic rename) — measured on cardinal: 1 KB prompt 15.2 s cold → **0.49 s warm (31x)**, output cmp-identical, warm-extend restores 1024 and continues; **(B) batched prefill** (`hybrid_prefill` + 3-dtype scalar/AVX2 batched matmuls, weights streamed once per B positions instead of per byte, GLA scan/conv/attention kept sequential so the result is BITWISE-equal to per-byte prefill — proven at B=2/8/32/64, scalar and AVX2, on the i7) then **(B2) j-split threading** of the batched matmul (`hybrid_mm` mirroring `hybrid_mv`, int8 quant hoisted to kill the scratch race): bench prefill 1024 = 20.8 s → **11.7 s** (1.78x), decode untouched. Honest residual: threaded-batched wall is only 1.27x over the old 7-thread sequential path because a ~11 ms/position SERIAL floor remains (scaling probe is linear in prompt length ⇒ per-position work — rmsnorm/attention dots/recurrent path — not O(n²) attention); profiling that floor is the named next lever, alongside the int8 200m export (~2x decode, exporter path exists) and draft-model lookahead. tests/engine 15/15 (4 new state-cache + 5 new prefill tests), component docs updated (`state_cache.md`, `hybrid_trunk.md`). Held for operator: git commit of the engine changes (all local-only), chat200m int8 export, production engine rebuilds, and the physical checks (power brick/BIOS `no_turbo`, second RAM stick = ~2x decode ceiling). Detail in the 2026-07-13 evening section.

- **2026-07-13 (inference optimization run: CPU decode 2.5-4x faster end-to-end, threads now hardware-calibrated, one live protocol bug root-caused + fixed):** Executed `inference_optimization_handoff.md` overseer-style (all building by Opus 4.8 agents, independent review + verification on every piece; the 800M pretrain untouched throughout). Shipped, all bitwise-gated: **int8-AVX2 matvec** (x86 int8 no longer scalar; parity vs scalar on 17 shapes, verified on cardinal — activates on the chat80m int8 re-export + rebuild, stacks on the 7x AVX2 win); **trace-off serving** (10th optional header field + 16-byte FFRM frames replace ~300KB/byte TFRM when the MRI view isn't open: 3.00 -> 2.38 ms/byte on the Mac, output byte-identical; `/generate` keeps full tracing); **prompt/n-gram lookahead decode** on the PyTorch chat path (no draft model; greedy output PROVEN byte-identical to `stream()` by a fail-if-broken parity test; 1.21-4.94x depending on prompt reuse; drafting gated to CPU until MPS parity is measurable — retry condition documented); **spin-then-park thread pool** (the ~61µs condvar dispatch was why extra cores gave nothing; now 0.68 ms/byte @8T on M3 = 2.56-2.97x over 1T, bitwise at every count); and on top of it the **hardcoded 8-thread default is DEAD** — `hybrid_load` now runs a <500ms one-time calibration on the real loaded weights (1,2,4,.. ladder, median-of-passes, diminishing-returns knee): M3 auto-pick lands within 4.5% of best manual, cardinal picks 4-7 and structurally never the 8-thread collapse it used to ship (forced 8T there = ~50-60 ms/byte; auto = 2.2-2.6). Also: **API MRI opt-in** as a strict OpenAI-superset (`"mri": true` on `/v1/chat/completions` + `/hybrid/chat`, plus a dedicated `/v1/chat/mri` sibling; MRI rides an additive top-level `mri` key so plain OpenAI clients are unaffected; documented in `documentation/api/external_api.md`). Mid-run production incident root-caused and fixed same-day: the persistent `chat_traced` subprocess desyncs permanently when a prompt line exceeds its ~1KB fgets buffer (chat-tab histories do) — it then reads every header line as the prompt and the model autocompletes the settings digits ("16 16 16..."); fixed both sides (python tail-clamp sized to fgets capacity AND per-request reply reserve, engine drains over-long lines), regression-tested; the clamp works against the existing binaries, no rebuild needed. Independent reruns: tests/engine 6/6, tests/mri 145/145. Held for operator: engine rebuild on both boxes (activates int8-AVX2 + auto-threads + drain), server restart (activates clamp), chat80m int8 re-export, cardinal C-backend reload (its subprocess is still poisoned), Medusa MTP heads (#4, canonical-trainer change). Detail in the 2026-07-13 inference section.

- **2026-07-13 (IDEA 2 external memory: last boundary measured, one lever falsified, research consolidated — all while the 800M pretrains):** Closed the memory campaign's last open boundary and killed a tempting shortcut, no production impact (chat200m on MPS only; the Carpathian ollama models stayed loaded and untouched throughout). NATURAL-QUERY end-to-end, the number the heuristic-query evals could not reach: using qwen (read-only) to write 1211 natural questions over 31 topics, a key head trained on 26 topics and tested on 5 HELD-OUT topics transfers at ~3x baseline but natural queries are ~2x harder than "What is X?" — hard-store recall@1 0.32 (from 0.50 heuristic); end-to-end grounding beats bare 14x (0.14 vs 0.01) but is only 14% absolute because it is RETRIEVAL-bound (recall@1 ~0.30 caps what reaches the generator). FALSIFIED the obvious lever off that: multi-leaf (top-k) injection to convert recall@5 (~0.63) into grounding — grounded_acc goes 0.130 (top-1) -> 0.120 (top-3) -> 0.080 (top-5), i.e. MORE candidate leaves HURT, because a 200M copy-limited generator cannot disambiguate among them; this localizes the bottleneck to top-1 PRECISION + generator disambiguation and PREVENTED a net-negative serving change (serving defaults TOP_K=3; blindly raising it would have degraded the local model). Both addressed by the 800M in flight; re-ranker sub-idea logged as the fallback. Consolidated the scattered 2026-07-11..13 memory findings into one honest research doc `developer_documentation/research/external_memory_retrieval.md` (the exact-recall sibling to the IDEA 1 streaming-state doc) and indexed it in `research.md`; the neural retriever plugs into serving behind ONE function (`retrieve()`), staged not shipped. Ledgered same-day: `successes.md` (natural-query transfer), `failures.md` (top-k injection). 800M pretrain healthy throughout (step ~8950, loss ~0.8, ~4k tok/s). Detail in the 2026-07-13 section.

- **2026-07-10 (chat200m chat model DONE — all four gates PASS, servable on both backends):** The combined gated chat-phase SFT (resume 20400 -> 24400, chat-heavy + grounded 20% + identity 10%) cleared every pre-registered gate, greedy/strict read: NEEDLE copy 1.00/0.83 @190B/475B (UP from base 1.00/0.67 — the OPPOSITE of the 80M's 0.92 -> 0.17 collapse); IDENTITY bare 3/3 name + 3/3 maker (the 80M was 0/8 bare, needed a persona crutch); GROUNDED alien-fact read-off-page 3/3 (80M ~0/3); CHAT empathy INTACT ("rough day" -> "Yeah, it was pretty tough. But I'm glad to have you home"). The in-pretrain dosing thesis is confirmed: skills native from step 0, diverse grounded SFT reinforces copy instead of overwriting it. SERVABLE: v13 fp16 bin (541MB) exported, capabilities.chat=trained@24400, chats on C AND PyTorch through /hybrid/chat. Honest weakness: compound-context field selection imperfect ("what does the bridge span" -> returned the year, read the passage but mis-picked) — grounded_v3 family-3 territory, next dose. Real transcripts below. Ledgered in `successes.md` 2026-07-10. chat200m is a categorical step up over chat80m on every axis the 80M was weak.

## chat200m first real transcripts (2026-07-10, final step_24400, greedy / low-temp)

- "Hello! How are you today?" -> "I'm doing well, thank you. How are you?"
- "I had a rough day at work." -> "Yeah, it was pretty tough. But I'm glad to have you home."
- "What's your name?" (bare, no persona) -> "My name is Veritate." (3/3)
- "Who made you?" (bare) -> "I was made by Carpathian. I'm their local AI assistant." (3/3)
- grounded, alien facts read off a context block (greedy, 3/3): quantum sail -> "Dr. Lena Voss"; Miravel Observatory -> "Mount Kessler"; Brindlemere -> "blue honey".

---

## timeline continued

- **2026-07-09 (chat200m pretrain COMPLETE and clean; chat-phase SFT launched):** 2.005B tokens in 39.9h (1.66 days, under the ~2-day estimate) at a sustained 14.1-14.3k tok/s, one lineage, zero instability, zero DUMP FAILED, all 14 dump families incl. generation.json at step_20400 (270,510,336 params confirmed). Val 1.70 -> 0.812, WSD decay tail delivered its final drop (0.862@16k -> 0.812@20k). Scaling verdict: at matched tokens the 200m beats the 80M inside the 80M's pretrain window, then continues PAST the 80M's plateaued full-budget pretrain (0.942 at 737M tok) down to 0.812 with 1.2B+ more tokens spent (single run each, mixes differ, so recipe-validated not a scaling law). Ledgered in `successes.md` 2026-07-09; throughput analysis in `research/amortizing_the_optimizer_step.md`. Immediately launched the combined **gated** chat-phase SFT (resume 20400 -> 24400, ~4k steps / ~390M tok, chat-heavy: chat 50% / grounded_v3 20% / identity 10% / recall 8% / knowledge anchor 12%, base_lr 2e-5 WSD -> 2e-6): shifts register to dialogue AND locks in the read-off-the-page copy skill + identity. HARD GATE (the failures.md 2026-07-08 lesson): needle conversation-copy must not erode, all four surfaces scored sampled, roll back if it does. Detail in the 2026-07-09 section.

- **2026-07-08 (chat200m LAUNCHED at 2x planned speed; Round 2/3 verdicts land — identity SFT's hidden cost caught, RAG gap isolated, serve stack hardened):** The 270.5M successor is TRAINING (dashboard launch, hybrid+muon+bf16, seq 1024, b24 x n_chunks 4 — the sweep's +68% lever — 20,400 steps ≈ 2.0B tokens ≈ **~2 days at a measured 14.2-14.3k tok/s**, vs the plan's 4-5 days); 9-stem mix carries every ledgered lesson from step 0 (chat 12%, recall 4%, **grounded_v3 2.5%**, identity 2%, code 6%); first-checkpoint dump gate PASS (all **14 families incl. generation.json** — the multicorpus fix holds), val 1.141@1500. **Round 2 (RAG) verdict, honest:** platform retrieval wiring works end-to-end (BM25 finds the right fact every call) but the model cannot READ it — alien-fact extraction from a context block is **1/4 at BOTH 48000 and 51000** (never trained, the 121M transfer gap mirrors the 10M one). Worse, the makeup needle A/B caught what the skipped gate missed: **the identity SFT collapsed conversation-copy 0.917 -> 0.167** (same interference signature as the recall-SFT kill, at 1/10th the dose, while val IMPROVED — val is blind; ledgered in `failures.md` 2026-07-08). The twist that decides the flagship: the SFT's real gain is persona-context APPLICATION — 51000 + persona line answers its name 6/8 at temp 0.5 where 48000 + the same line scores **0/6** ("Your name is Jack Thompson"; "Yes, I am human") — so serving stays on 51000 and the serve layer now supplies the persona. **Round 3 + serve stack, all live and independently verified:** v13 temperature bug fixed (the dial was dividing x1024-scaled logits — everything ever "sampled" on C was near-greedy; now 6/6 distinct at temp 1.0, deterministic at 0, loop battery 0/12), **int8 v13 shipped** (126.5MB, +0.004/+0.002 bpb, bitwise scalar-vs-sdot; **200m-shape p50 1.7-1.9 ms/byte on P-cores int8 = Sam's <=2ms target MET**; E-core class 3.2-3.4 = not yet), self-talk dead on both backends (multi-marker stop), /hybrid/chat finally gets the rep guard + control-byte sanitize + persona (name 0/8 -> 4-6/8 deployed) + seq-budgeted RAG prompts (was ~50% silent-empty replies). **grounded_v3 built + audited** (38MB, 145k convs, 4 families incl. honest-miss, invented entities, 500/500 extractive audit). research/ papers folder landed (5 papers, evidence-disciplined). Repair round for the 80M (combined grounded+identity SFT from 48000, ALL gates incl. the one that was skipped) queued behind the 200m. Detail in the 2026-07-08 section.

- **2026-07-07 (C engine serves the hybrid trunk — v13 format SHIPPED, all 5 gates passed, chat80m live on the C backend):** The flagship's trunk is no longer PyTorch-only: a v13 `.bin` format + fp32/fp16 forward path landed in the v1 engine (new `src/hybrid.c`, NEON matvec kernels, exporter `trunk=hybrid` route), FULLY backwards compatible with v3-v12. Gates, all measured: **byte-parity 192/192** (3 chat prompts x 64 greedy bytes, C vs PyTorch fp32 full-forward, both fp32 AND fp16 bins — fp16 loses nothing and is the shipping default at 243.5 MB); **bpb matches PyTorch to 4 decimals** (1.4447 fineweb / 0.9899 chat_v1, C ppl mode); **canonical compat byte-identical** (pristine-HEAD engine vs new engine on a committed v9 fixture + 0/20 stress + ASan clean); **kernel identity bit-exact** (scalar vs NEON fp32/fp16, plus exhaustive 65536-value f16 convert check); **e2e on the live dashboard**: `/generate?backend=c` streamed a coherent chat reply that self-terminated at `<|end|>` (106 frames, 0 errors). Speed on M3 Ultra: **p50 1.1 ms/byte fp16 (520-600 tok/s, 4 threads), 1.8 ms single-thread**; E-core-pinned (low-power proxy) 5-7 ms/byte — the validated int8-per-channel path (+0.0006 bpb in simulation, gate passed, C kernels not yet written) is the documented next ~2x. TWO pre-existing bugs found and fixed along the way: (1) `score_dot_v` hardcodes head_dim 64 and heap-corrupts on anything else — root-caused with ASan after it masqueraded as "LTO link-order nondeterminism", now refused at load; (2) the chat_traced protocol destroyed prompt newlines (flattened to spaces), degrading every chat-template reply through the C backend — now escaped as 0x01 end-to-end, and the multi-line template reproduces the PyTorch reference byte-for-byte ("The capital of France is..."). NOTE for next restart: the live server still runs the pre-fix `c_engine.py` (flattened prompts) until reloaded; the new code is on disk and backward-compatible both ways. Detail in the 2026-07-07 v13 section.

- **2026-07-06 (wave-4 memory race round 1 COMPLETE — three kills, one confirmation, two survivors):** The falsifier machinery ran the full course in ~24h. KILLED: **M1 delta-rule** (diverged twice under two different numeric regimes — creeping fp32 WY-inverse blowup under beta saturation [root-caused, block-recursive fix shipped + triple-verified] then a sudden no-precursor NaN at step 2094 with the fixed inverse; gla/pinned trained clean on identical configs; ledgered with retry conditions); **M2 pinned register** (no recall benefit in any sample — 0.338 vs baseline 0.423 at n=130 in-distribution, needle 0.00=0.00 — plus a 7σ val tax on pure chat; killed per pre-registered "no better than M0"); **late-phase recall-SFT at 25%** (the at-scale H2 test INVERTED: needle recall at ~190B collapsed 0.92→0.08 — the narrow templated corpus REPLACED the model's emergent copy-from-context behavior; flagship ROLLED BACK to step_48000). CONFIRMED: **recall-pressure data teaches retrieval** (m0recall 0.47 vs m0gla 0.05 in-distribution at 10M, 10x lift) — the 10M needle floor was a TRANSFER gap (alien eval surface forms), not absent capability. SURVIVORS now building: **M3 surprise-gated rehearsal** (inference-time digest re-injection — immune to training interference by construction; falsifiable A/B on the flagship in flight) and **train-time state carry** (`state_carry=chunks`, E4b-precedented — the only route that can EVER make past-window recall trainable; validation run queued). Also: needle_bench gained state_rule threading; extended-power scoring (n=130) adopted for near-margin verdicts; hybrid-vs-patched second seed (agent_roe) running on the freed GPU. All entries in `failures.md` / campaign doc same-day. Detail in the 2026-07-06 section.

- **2026-07-05/06 (chat80m COMPLETE — first conversing byte model; memory race launched):** The full three-phase 80M build finished clean: pretrain 30k (val 1.695→0.942) → midtrain anneal 10k at 45% chat (→0.681) → SFT 8k pure chat (→**0.647**), one lineage via dashboard resume (muon momentum carried), ~9h GPU total, zero DUMP FAILED across 96 checkpoints. **It talks**: greeting answered naturally ("I'm doing well, thank you. How are you?"), emotional register handled with a dialogue-appropriate follow-up ("What do you mean?"). Honest boundary, measured in the same smoke: factual recall fails (capital-of-France answer loops) — answer-shape without world knowledge; that is the documented compute wall at ~470M training tokens, not a pipeline defect; fixes are tokens/scale (160-200M next) + sampled decode. Ledgered in `successes.md`. The freed GPU went straight to the wave-4 memory race: `m0gla` 10M arm live (hybrid+muon, pure-chat 40/40/20, state_rule=gla baseline), M1 delta and M2 pinned arms queue behind it — needle recall-vs-distance curves (measurable past the window via streaming) pick the mechanism the scale-up inherits. chat80m's own needle curve running on CPU in parallel. Detail in the 2026-07-05 section.

- **2026-07-05 (chat80m LAUNCHED; E6 SLM killed; long-context memory campaign opened — M1/M2 + streaming shipped):** The first scaled chat model is TRAINING: `chat80m` (hybrid trunk 121.8M + muon, bf16, 30k steps, 6-corpus mix with 10% chat from step 0) dashboard-launched after the full gate sequence (MPS stability smoke at real shape all-finite; all 13 dump families verified firing per checkpoint, zero DUMP FAILED); ~16k tok/s, ETA ~6.5h, val 1.695→1.051 over steps 2k→10k. E6 SLM KILLED on its pre-registered condition — quality regression at equal steps (tail-10 val 1.0638 vs e2patched 0.9763, ~8σ): fast to a mediocre plateau, never reaches baseline-final; at byte level the "easy" tokens carry the structure (ledgered with retry conditions incl. patch-granularity selection and anneal-phase-only use). Wave-4 long-context memory campaign opened (`long_context_memory_campaign.md`): target = flatten recall-vs-length so conversations don't degrade; conversation-needle benchmark built + plumbing-validated (`experiments/v2/longctx/`); TWO new memory mechanisms shipped as opt-in `state_rule` on the recurrent mixer — M1 delta-rule (error-correcting overwrite, exact to 7.7e-7 vs oracle) and M2 pinned register (decay-exempt slots; invention) — default gla path proven bit-identical, dumps 12/12; streaming/state-carry inference shipped (`forward_streaming`): window-carry equals one-pass BITWISE on all three rules, so past-window recall is now measurable and chat length is no longer capped at seq. vanilla_trainer.py gained `state_rule` reserved flag (synced checkout — diff surfaced for upstream mirror). Next: pretrain → midtrain anneal → SFT → first conversation; M0/M1/M2 needle race at 10M; MoE/MTP as 80M A/Bs. Detail in the 2026-07-05 section.

- **2026-07-05 (pivot to the actual chat model: DeepSeek levers mapped, 4.2 GB chat data built, 80M run prepped):** With the 10M architecture race won (hybrid+muon), turned to building a model that can actually be chatted with. (1) DeepSeek playbook researched + ledgered (`efficient_architecture_research.md` wave 3): DeepSeekMoE is the best fit for this FLOP-bound/RAM-rich box (more params at fixed active FLOPs) but has ZERO evidence below ~0.3B — so it is to be tested AT the 80M scale-up, not at 10M where it would false-negative; byte-level MTP is cheap and byte-level is its best case (low-info targets) but small-scale evidence is negative; MLA is redundant on the hybrid trunk (global path already O(1)); FP8 training is dead on MPS (no kernels); R1/GRPO is post-SFT only. Experiment specs E8 (MoE)/E9 (MTP)/E10 (MLA) written with pre-registered falsifiers; an MPS-safe capacity-based MoE FFN is being built now (not launched — will be a 80M variant). (2) Conversation data went from 6 MB to ~4.2 GB: `chat_v1` (SmolTalk `all`, 597,580 convs, 2.0 GB) + `chat_v2` (Tulu-3 SFT mixture minus the CC-BY-NC subset, 902,049 convs, 2.0 GB), one byte template, in-content-NUL rejected, both validated and live in the dashboard corpus dropdown; source ranking + 80M pretrain/midtrain/SFT mix in `chat_corpus_sources.md`. (3) The 80M chat run is prepped: veritate_80m trainer supports the winning stack; hybrid at the 80M shape = 121.8M params, passed the CPU shape+finite smoke; full plan + launch config in `chat_model_80m_plan.md`. Launches the moment E6 frees the GPU. (4) Platform fix: exporting a checkpoint while a run is training is now allowed (checkpoints are atomic; the export flow stays live, other flows gray out) — closing the reported bug. Honest boundary restated: one M3 Ultra will not reach Opus-level (documented compute wall); the target is a small, fast, genuinely conversational byte model, best-possible-per-FLOP. Detail in the 2026-07-05 section.

- **2026-07-04/05 (architecture campaign: all 7 wave-1 verdicts in — HYBRID trunk is the winner; memory-recall and looped-depth killed with evidence; E6 selective-loss live):** Every experiment ran through the dashboard at canonical 10M, byte-level, fineweb_edu, 12k steps, muon arms, pre-registered falsifiers. Scoreboard (final val, lower = better): **hybrid 0.9707** (patched local + constant-state recurrent global, best of ALL arms, 1.70x wall-clock vs dense, 113% dense throughput) > patched 0.9776 (1.82x) > E4b memory-carry 0.9867 > looped 0.9920 > recurrent 0.9900 per-step-parity > dense-muon 0.9990 > adamw 1.0375. KILLED: fast-weight memory as a knowledge store (E4 + E4b: 0.0 exact recall at every distance despite two training regimes; soft trace only within the trained 1024-byte horizon, actively harmful beyond 4096 — kept as a free context extender, its val BEATS dense) and test-time depth scaling (E7: R-sweep on the trained looped checkpoint peaks at the training-mean depth R=3 and degrades monotonically past it — no think-longer effect; params-matched win over dense recorded but unattributed without a loop-free control). Measured composable stack: Muon 1.60x x patching 1.82x ≈ 2.9x, hybrid carries the best quality AND O(1) global decode state. E6 (RHO-1-style selective loss, frozen e1muon reference, keep 0.6, patched+muon student) LAUNCHED and training clean at 69k tok/s (SLM smoke-tested at real shape first); falsifier <1.15x vs e2patched. Ledgers + per-component docs updated same-day; UI trunk/optimizer help text now carries the measured verdicts. Detail in the 2026-07-04/05 section.

- **2026-07-04 (h7 ML strategy CONFIRMED by adversarial audit — first full survivor in ~35 families; ml7 live arm building):** The daily-ML 7-day-horizon L/S book survived every hardening arm: leakage hunt clean (trailing features, train-only scaling, clean boundaries, funding already charged), 14d purge PASS, quantile 0.15/0.25 PASS, 3-seed variance PASS on means, and the decisive ENTRY-LAG test — one full day of execution delay — still 12.9%/yr Sharpe 1.04 (hgb_cls, 5/5 years positive) / 19.4%/yr Sharpe 1.50 (hgb_reg). Plain momentum FAILS the same lag test (Sharpe 0.77): the ML layer is what makes the edge execution-robust. Standing caveats: 2026-majors survivorship (ambiguous sign for L/S), 2026 partial year, h7 the sole surviving horizon of 3 (selection), real seed variance (live arm ensembles 3 seeds). Artifacts: SMOKE_RESULTS/daily_ml_audit_*. The live forward arm is SHIPPED + RUNNING (2026-07-04): `ml7` in the Trading extension — frozen 3-seed HGB ensemble (trained through 2026-06-26, 85,778 rows, fit 87s), byte-exact feature-parity module + test, 7 daily-staggered tranches (entry-lagged by construction = the validated lag1 numbers), $10k simulated, first tranche live (7L/7S, gross $1,428.5 = equity/7, fee $0.49). Detail in the 2026-07-04 ml7 section; strategy write-up in trading_model_plan.md section 11.

- **2026-07-04 (trading extension consolidation, SHIPPED + LIVE):** merged the three trading extensions (market, paper_trade, market_intel) into ONE self-contained `extensions/canonical/trading/` (id `trading`), rebuilt the UI as a five-tab page (Overview / Strategies / Market Intel / Research / Settings) with plain-language captions, validation stories, a system-status strip of every runnable, and simulated-capital banners everywhere. All routes now under `/ext/trading/{market,paper,intel}/*` + new `/settings` (server-side defaults + a user-editable scraper CHANNEL registry: rss/reddit/gnews/fear-greed with per-channel health) + `/system` (+stop/stop_all). Data migrated byte-identical (md5-verified) to `installed/trading/data/{market,paper,intel}`; the three live ledgers resumed without a gap (xsmom 42->43, 42->43 ticks, eqmom 10->11 post-restart); intel watch resumed (92 events intact); OKX recorder relaunched on the new path (10/10 pairs ticking). 124 extension tests green (99 migrated + 25 new settings/channels/system). Old dirs, routes, and catalog entries deleted; old routes verified 404. Also fixed a latent 500 in the data-report route (`md.EXTERNAL_DIR` never existed). Dashboard restarted via /lifecycle/restart with the live e2patched training run preserved (PID survived, verified). Detail in the 2026-07-04 section.

- **2026-07-03/04 (architecture research campaign: 4 efficiency levers built + first verdict — Muon 1.60x CONFIRMED):** Pivoted all compute to the efficiency mission after killing the coder detour (from-scratch 2.5B and cheap byteification both disproven, coder-SFT pipeline proof 0->15% HumanEval banked). Two research agents mapped the 2024-26 landscape into a ranked lever table (`developer_documentation/training/efficient_architecture_research.md`); success/failure ledgers seeded (`successes.md` / `failures.md`). BUILT + verified four experiment vehicles, all dashboard-launchable via new reserved flags `optimizer` (adamw|muon) and `trunk` (dense|patched|recurrent|memory): Muon hybrid optimizer (`veritate_core/plugin/optim.py`), SpaceByte-style patched trunk, constant-state gated-linear-recurrence trunk (O(1) decode state, chunkwise math exact to 4e-11), Titans-style surprise-gated memory trunk (closed-form inner write rule exact vs autograd). FIRST VERDICT (E1): Muon = **1.60x byte savings** at 10M byte-level (val 0.9990 vs AdamW 1.0375 at 12k steps; reaches AdamW-final at step 7500; falsifier <1.15x cleared; adopted as default). E2 (patched, 15.0M params at dense-FLOPs, +28% tok/s) running. Hard lessons preserved in preflight: dynamic shapes = 23x MPS slowdown (24c), variant dumps must be smoke-tested at REAL shape with real prompt lengths (24d: fixed-slot bug silently killed 7/14 dumps; root-caused, fixed, all dumps verified on all three variant trunks; dump failures now print loud to the run log). Detail in the 2026-07-03/04 section.

- **2026-07-02 (trading campaign: 6 new falsifier-driven experiments; fee floor collapsed 5x; ONE candidate survives -> forward paper run LIVE):** The 10-20bp fee assumption behind every prior "sub-fee" verdict is stale: US-legal perps now cost 1.9-3bp/side (Kraken US perps, Coinbase nano perps, CME micros). Re-tested everything that died on costs or the missing short leg at the true 2026 stack. Results: OFI x magnitude composition multiplies gross 3-7x exactly as designed but breaks even at ~1-1.5bp/side (fails); 1h XS reversal is the strongest OOS signal ever found here (+121%/yr, Sharpe 1.36 at zero fee) but passive/maker execution CANNOT capture it (the fillable subset is negative gross: adverse selection, not fees, is the wall); options/vol monetization dead (IV already prices the magnitude signal; VRP + spreads); carry dead in current regime ($1.51/day per $10k vs $1.23 T-bills); marketof A/B (finally scored): order-flow channels add NOTHING to byte-model direction — the model was never the bottleneck. THE SURVIVOR: 1w cross-sectional momentum L/S, net-positive every OOS year at every fee tier (funding is a +1.85%/yr tailwind, not the assumed 10% drag), Sharpe 0.44 unfiltered, 0.93 with a calm-vol gate (gate is test-adjudicated: placebos pass 21% of the time, so it needs forward proof). BUILT + LAUNCHED the forward validator: `xsmom_trader.py` in paper_trade (base + gated arms, $10k each, 3.5bp/side, weekly rebalance, live OKX marks), 26 tests green, routes dormant until next dashboard restart, standalone process live. Detail in the 2026-07-02 section.

- **2026-06-27 (back to model training — 2.5B byte-level Python coder, LIVE):** Launched `pycoder_3b` from scratch on M3 Ultra via the dashboard (the `veritate_3b` trainer is exactly 2.52B params: dense byte-level decoder, L=32 h=2560 ffn=10240, vocab=256). Plan = "language first, then Python": Stage A pretrain on FineWeb-Edu (5.34 GB, language base, present on disk); Stage B teacher-distilled instruction->Python SFT from qwen2.5-coder:32b (corpus generation running in background at ~64 B/s, Python-only system prompt, chat template). Live at launch: step 1 loss 6.05 (byte random ~5.55), **tok/s ~210**, fp32 AdamW (bitsandbytes absent on MPS), act-ckpt on + batch=2 (over-conservative for a 256 GB box), and GPU contention with the generation job. HONEST WALL (quantified): ~210 tok/s = ~18M tok/day; a sub-Chinchilla coherent 2.5B (~3-5B tok) is months, Chinchilla-optimal (~50B tok) is years; a *pure*-teacher corpus for a 2.5B is infeasible (years at ~10 MB/day). So real-corpus volume must do the pretraining; teacher distillation is the instruction/Python-style layer. 3 research agents dispatched (M3 throughput levers, code-distillation corpus efficiency, feasibility/unconventional levers). Detail in the 2026-06-27 section.

- **2026-06-20 (trend/momentum disproven, 4-agent adversarial search):** chased a quick +52% trend-following backtest; a 4-way parallel search on real Binance.US daily data (train/test split, walk-forward, regime breakdown, fee sensitivity) DEMOLISHED it. The +52% was ~1/3 in-sample overfit, ~1/3 two outliers incl. CORRUPTED TRX DATA (delist/relist booked a 1093-day gap as a single +391% day; 3 of 4 agents independently found+killed it), ~1/3 bull window + survivorship (dead coins LUNA/FTT excluded). Honest walk-forward OOS: mean -1.3%, median -24%, 8/19 coins positive; excl outliers -10.8%. Dies to fees between 20-50bps (realistic alt cost 30-50bps). Bull -1% (surrenders upside) / chop -17% (bleeds). Overlays (vol-target, ATR, volume, macro-gate) raise NO return, only cut drawdown. Portfolio rotation (top-K momentum) still -29.6% OOS, loses less than hold's -52% but never positive. VERDICT: no positive-return edge in long-only crypto trend after real costs; same risk-reduction-only wall as news. Money path now disproven 5 ways within the Robinhood-spot/long-only/public-signal frame. The only documented retail edge remains funding-rate carry, which needs a PERPS venue (not Robinhood spot) - see [[project_market_platform]].

- **2026-06-19 (multi-arm experiment + crypto/stocks + research grounding):** generalized the trader to N parallel arms (A/B/C/D) across BOTH crypto (Binance.US) and stocks (Yahoo Finance quotes + financial-news scrape), one shared LLM, each arm its own $10k ledger. New `/ext/paper_trade/exp/*` API + a research-driven UI (one-click launch, sorted leaderboard, overlaid equity chart, per-arm tabs with cash-vs-invested bar + holdings + trades). Pulled `qwen2.5:14b-instruct` as the recommended mid-size scorer. Verified end-to-end live: a stocks arm bought NVDA, a broad-crypto arm bought BTC, 14 tests green, JS parses clean. TWO research agents (UX + trading evidence) cited below. **BLUNT verdict from the trading literature: a long-only, fee-paying news-sentiment strategy has NO documented net-of-cost edge** — the alpha is in the short leg (unusable), costs erase ~96%, the signal mean-reverts, crypto adds spread + pump contamination. Expect arms to track or slightly underperform their own buy-and-hold. This is a research instrument, not a money path. Needs a dashboard restart to mount `/exp/*`.

- **2026-06-17 (48h forward validation, LIVE):** clean forward paper run relaunched on the corrected event-driven engine: scan news every 5 min, trade only on a fee-aware deadband shift, long-only normalized (no leverage), marked to live prices, 20bps spread. New this run: a BTC buy-hold benchmark on the equity curve (the real scoreboard), token-aware news (pick a coin and it auto-pulls Google News for it), incremental scoring (only new headlines hit the model), and a 6-chart "is the news predicting?" panel. Ledger reset to clean $10k. Autonomous 4h checkpoint loop for 48h. Judge by the vs-BTC-hold delta, NOT raw equity. Checkpoint lines in the 2026-06-17 section below.
- **2026-06-16 (BREAKTHROUGH — first real signal):** LLM-scored crypto news sentiment predicts next-day BTC returns, corr **+0.48** (vs +0.03 for fear-greed), and it **survives the look-ahead control** (+0.58 on post-training-cutoff dates the model couldn't memorize). Short-horizon edge. First defensible edge found. NOT yet proven profitable (correlation != profit; needs forward + fees); the platform to validate it forward (`news_trader.py`) is built and tested. Detail in 2026-06-16b.
- **2026-06-16 (sentiment platform foundation, DONE):** built + verified the information/LLM path: new platform endpoint `POST /teacher/complete` (calls any user-added model; verified live — Ollama `qwen2.5:7b-instruct` scores headlines correctly in ~0.3s); Paper Trading is now a standalone extension with a server (`scraper.py` free RSS+fear-greed, `sentiment.py` LLM scoring + time-decay aggregate, `/ext/paper_trade/sentiment` route). 8 new tests + 45 total green; live scrape verified. HONEST GATE: the only free historical sentiment series (fear-greed) has ~zero correlation with forward returns (~0.02), so there's no free backtest — the LLM-news edge is event-level and only testable FORWARD/live. Foundation is an instrument to run that forward experiment, not a validated money-maker.
- **2026-06-15 overnight (DONE):** hunted for a profitable Robinhood-able (spot, long/cash) strategy across 40 coins, out-of-sample, 3 agents + own re-validation. VERDICT: nothing beats holding BTC on return; the model's only real, monetizable skill is RISK REDUCTION (dodging 70–95% crypto crashes). Micro-trading = guaranteed −37% to −196%/day fee bleed (the "$1k/day micro" goal is inverted from reality). Mean-reversion + vol-regime = overfit mirages. The one genuine modest edge: a model-filtered breakout/trend strategy (full Sharpe ~0.30, trend-regime-dependent) whose value is capital protection (−7% in bear vs BTC −70%), and the byte model measurably improves it (Sharpe 0.30 vs 0.14 plain). The honest product = a survival strategy, not a money printer. Full detail + FINAL VERDICT below.

---

## the chat models in plain language

Written for a human, not an engineer. The full readable-but-technical versions are the
research papers (`research/learning_to_talk.md` for the 80M story, `research/amortizing_the_optimizer_step.md`
for the speed work). This is the short narrative that used to live in the tuning journal.

**What the models are.** Small AIs that run entirely on this one Mac, at no cost. They read
and write raw text one letter at a time (no dictionary of word-pieces). Small means fast and
private; small also means they cannot memorize the whole internet. The whole game is getting
the most out of every one of the limited number of "settings" the model has (chat80m has about
122 million; its bigger sibling chat200m, training now, has about 270 million; the largest
commercial AIs have hundreds of billions).

**How chat80m was trained: three meals.** Training is feeding the model reading material while
it slowly adjusts its settings to predict what comes next. Breakfast (about 9 hours): mostly
encyclopedia-style web text, learning how English works. Lunch (about 3 hours): now nearly half
the reading is real dialogue, so it starts sounding like something you can talk to. Dinner
(about 2 hours): pure conversation, the polish that makes it answer a greeting like a person
instead of continuing your sentence like an essay. Result: ask "Hello, how are you?" and it
answers "I'm doing well, thank you. How are you?" from its own settings, on your machine.

**What it can and cannot do (honest version).** Can: hold a casual conversation, follow the
flow, respond in the right tone, quote back things you told it a moment ago, and answer fast (a
fraction of a second). Cannot: reliably recall world facts ("capital of France" trips it up),
because facts live in sheer volume of reading and a small model on one machine has not read
enough. The two fixes are a bigger model fed much more text (chat200m, in progress) and letting
it look facts up while it talks (retrieval, being wired in).

**The hard lessons, in order.** (1) A memory drill backfired: cramming one narrow skill at the
end of training overwrote an older skill (its quote-back score fell from 92% to 8%). Rule learned:
small doses mixed in early, never a late cram. (2) The same thing happened, more subtly, when we
taught it its name ("Veritate"): the naming lesson quietly cost it the quote-back skill again, and
a test that would have caught it got skipped. (3) Giving it a name is a balance: vary the questions
every way a person might ask, but keep the answer's key fact word-for-word identical. Too varied
and it never learns the name; too narrow and it forgets everything else. (4) The "creativity dial"
in the fast engine was broken (stuck on maximum-caution), which had been masking how shallow the
identity really was; fixed now. (5) The chat page used to let the model talk to itself, loop, and
occasionally return blank answers; all three are fixed.

**The bigger sibling (chat200m).** Training now, with every lesson above baked in from the first
step. Tuning made it about twice as fast to train as planned (~2 days, not 4-5), and at matched
amounts of reading it is already ahead of the smaller model in quality.

---

## 2026-08-03 — product-key memory: the layer, the kernel, and what the instrument was hiding

Context: the standing question is local AI on OLD hardware, without buying specs. Cardinal
(OptiPlex 7070, i7-9700T, BIOS-clamped to 800 MHz) is the target, and its decode is
RAM-bandwidth-bound, so the only currency is bytes read per token. A memory layer trades
streamed weights for addressed lookups, which is the one mechanism that decouples capacity
from bytes read.

**The layer.** `veritate_core/model_pkm.py`: `ProductKeyMemory` replaces the dense FFN
up/down pair with two sqrt-sized sub-key searches locating the top-k of `sub_keys^2` value
slots, so read cost is `O(sub_keys*key_dim + top_k*hidden)` while capacity is
`O(sub_keys^2 * hidden)`. Wired as `global_ffn=pkm` on the patched trunk mirroring the MoE
slot, reachable as `trunk=hybrid_pkm`. Slot spread verified healthy at init: 32,768 reads
hit 31,278 DISTINCT slots (95.5%), so the classic collapse-onto-hot-slots failure is absent.

**The instrument was the finding.** In eager PyTorch the layer measured 4.9x SLOWER than
dense at batch=1 (1387 vs 281 us/token) despite doing 5.5x FEWER multiply-adds and reading
8.7x fewer bytes. A batch sweep did not close the gap (7x at batch 1024), which ruled out
dispatch amortization as the explanation and forced a direct measurement of the floor:

| op (cardinal, 7 threads, batch=1) | work | cost |
|---|---|---|
| `x*2` on a 320-vector | ~none | 20.3 us |
| `a@b` 320x320 | 102,400 MACs | 38.0 us |

PyTorch charges ~20 us per OPERATION. 102k multiply-adds cost 17.7 us more than doing
nothing. Dense FFN wins in eager because it is 3 fat ops; PKM is ~40 skinny ones. **Any
sparse, gated or conditional design loses in eager regardless of merit, so batch-1
architecture comparisons in PyTorch measure the dispatcher.** Prior batch-1 architecture
conclusions in this repo are re-openable on that basis.

**The kernel, in two stages, both arms held at equal optimization.** Component profile of
the int8 forward killed my own prediction: the top-k sorts were 52% of cost and the
scattered gather only 8%; forcing the gather fully contiguous gained 7%, so access pattern
was never the bottleneck. Replacing ranking with a firing threshold (AVX2 compare +
movemask, **0.586 us vs topk 8.778 us = 15x** on that stage) took PKM 139.3 -> 69.6
us/token. Adding a 4-row-blocked int8 integer-dot matvec to BOTH arms gave dense its own
speedup (327 -> 199.6) and settled the ratio:

| int8, batch=1, core-pinned, both arms AVX2 | us/token | capacity |
|---|---|---|
| dense FFN | 199.6 | 819,200 |
| PKM + firing threshold | **51.2** | **83,886,080** |

**3.90x faster, 102x capacity, 6.49x fewer bytes.** The fair-comparison discipline mattered:
an earlier 4.70x shrank to 3.90x once dense also got AVX2, and reporting the smaller number
is the correct call.

**Quality, trained end to end on the clamped i7.** Matched A/B, 5m shape (8L h256 ffn1024),
enwik8, AdamW, fp32, qat off, seq 256, b8, 10,000 steps per arm: dense val **1.299641**,
top-k PKM **1.294813**. PKM led at every matched step from 7500 on. Margin is 0.37% on a
single seed, so per the agent_roe seed rule the reportable claim is **parity**, not a win —
which is all the thesis needs, since parity plus 3.90x is the win. Capacity is unexploited:
12.7x the params only matches, because enwik8 at 95 MB cannot fill 102x the slots.

**Threshold gating ported back into the trained model.** Until this, the quality number
described a top-k model and the speed number described a threshold model. `_fire()` uses a
hard gate in forward (exactly matching the kernel) with a straight-through sigmoid backward
so `theta` still learns, thresholding the STANDARDIZED candidate score so it stays scale
free as score magnitudes drift. Verified: real `theta` gradient on CPU and MPS, and genuinely
variable firing (~12.8 of 32 candidates, not a fixed count). New trunk `hybrid_pkm_fire`;
default stays top-k so nothing existing changes. Third arm `pkmfire_5m` training.

**Four negatives, each killing a plan already written down.**
- **PKM is not a training-speed lever.** MPS, seq 512 b4: dense 592M **2542 tok/s** vs PKM
  2309M-capacity **983 tok/s**. Capacity costs training time. My "500M in ~4 days via PKM"
  estimate was wrong and is retracted.
- **Sparse gradients lose on unified memory.** 537M-param value table, only 4.8% of slots
  touched per step, yet dense grad **0.188 s/step** beats `sparse=True` + SparseAdam
  **0.209 s/step**. Bandwidth makes the dense update cheaper than building the sparse tensor.
- **`n_chunks` does not generalize.** +68% at chat200m's shape; at 593M b24 bf16 it is a
  loss (1 -> 5722 tok/s, 2 -> 4924 = 0.86x, 4 -> 3993 = 0.70x).
- **Muon is an 11x tax on the clamped CPU** (9.5 vs 0.81 s/step), while remaining the GPU
  default. Optimizer choice is per box, not per project.

**500M budget, measured not extrapolated.** 593M on M3 Ultra: b4 fp32 3313 tok/s (41.4 d),
b12 bf16 4058 (33.8 d), b24 bf16 **5541 (24.8 d)** at Chinchilla 20 tok/param. The sub-week
target therefore needs `torch.compile` + Muon convergence + distillation; n_chunks and PKM
are explicitly NOT on that path.

**Two of my own bugs, caught only by parity checks.** An AVX2 matvec with broken `maddubs`
sign handling (relative error 1001 vs scalar), and a threshold calibration that computed
thresholds on the raw query while the forward used the RMS-normalized one — nothing ever
fired, the gather was empty, and it reported a clean, plausible, entirely fake 6.3x. Both
would have shipped as wins. Rule 24's scalar-parity gate earned its keep twice in one day.

**Platform work alongside.** Rule-11a contract `probe_module()` / `probe_weights()` added to
FFN / MoEFFN / ProductKeyMemory so consumers stop reaching into `ff.up` / `ff.down`:
checkpoint_probe (hook site + narrow-variant width padding + byte-direction stack), diff,
pruning (skips non-prunable layers, refuses with a clear message when nothing is prunable),
export (joins the rule-40b variant refusal instead of KeyError). Dump failures 66 -> 0 in a
live run. Separately, `corpus_sync.py` never created `data/corpus/` on the HF-stream or zip
paths, so every corpus install failed on a fresh box; the generic downloader did it and
those two did not.

Artifacts: `veritate_core/model_pkm.py`, `models/{pkmctl_5m,pkmmem_5m,pkmfire_5m}/`, C
harnesses at cardinal `/tmp/pkm_{bench,bench_i8,bench_thresh,profile,mv,fair}.c`. Nothing
staged or committed.

---

## 2026-07-17 — TTFB campaign: production first-byte latency, three verified rounds on cardinal

Context: user report of ~1500ms to first generated byte on `/v1/chat/completions` and the dashboard, with a suspicion "the engine goes idle". Method: profile-first (rules 102/118), Opus 4.8 agents building on cardinal, every change gated on byte-identical output, every diff mirrored into this repo (no git either side).

**Attribution (measured on cardinal, i7-9700T x86).** A "Hi" request renders to a 52-byte ChatML prompt; warm TTFB fits chat80m ~= 4.2n + 0.006n^2 ms, chat800m ~= 15n + 0.03n^2 ms (n = prompt bytes) — i.e. prefill at ~decode-cost per position dominates (60-93%). Cold adds spawn: fread of the whole .bin (chat800m 2937ms of 4453ms) + 1497ms thread-calibration burst per spawn. The "idle" suspicion resolved: the pytorch brain unloads on `pytorch_idle_unload_secs` and repays torch.load on next use; the C subprocess never idles out but is kill+respawned on model switch or unclean stream end. `state_cache_store` (35-200ms disk write) ran before the first byte every request.

**Round 1 (landed).** `VERITATE_PREFILL_BATCH=32` set at spawn in c_engine.py (feature existed since 2026-07-13, was never enabled in production; B swept 8/16/32/64 — 32 best, 64 worst on chat800m). `state_cache_store` hoisted out of `forward()` into the chat loops after the first frame flush (`model_store_state_cache`; before the first `forward_decode` so state is still pristine). Parity: tests/engine prefill suite 5/5, greedy outputs md5-identical across binary x batch combinations. Production warm cache-miss TTFB: chat80m 823-2448 -> 648-1625ms, chat800m ~1.06x (24 of 28 blocks are recurrent + sequential by design).

**Round 2 (landed).** (a) Parallel weight load: single-thread load is first-touch page-fault-bound (0.71 GB/s; dd reads the same file at 3 GB/s+) — loader now collects (offset,dst,bytes) jobs single-threaded, fills across the persistent pool with per-worker FILE* reopens (~3.7 GB/s): chat800m in-engine load 2937 -> ~610ms. New shim primitive `veritate_fseek64` (Win32 `_fseeki64` / POSIX fseek), Windows path otherwise unchanged. (b) Calibrated thread count persisted to `hybrid_threads.txt` in the state-cache dir (fnv-1a over shape+cores): 1497 -> 0.1ms on hit, same nt=7 pick. Net spawn-to-ready: chat800m 4453 -> 714ms cached (2105ms first-ever), chat80m 897 -> ~120ms. (c) pytorch `torch.load(..., mmap=True)`: raw load 14-25s -> 0.5s, `torch.equal` values; warm Brain reload 55-64s -> 45-49s (rest is model construction). Engine greedy sha-identical to round-1 binary both models; tests 14 pass + the pre-existing v9 SIGILL on this AVX-512-less box.

**Round 3 (probe).** kv-cache mode for the pytorch chat path crashes on `VeritatePatched` (`kv_cache_patch_attn` missing — the vanilla/RoPE trunks have it; inference_brain.md had already flagged the patched/recurrent trunk needs a model-side streaming-state cache). Dispatched under a hard byte-identity + schema-parity gate; lands only on proof, otherwise `mode="lookahead"` stays.

**Residuals (named, honest).** chat800m cold-prompt TTFB is now ~93% the by-design sequential recurrent prefill (~6s at 250B); the ~11ms/position serial floor named 2026-07-13 remains the next structural lever (chunked-scan prefill would break the bitwise gate — needs its own parity plan). The state cache already gives 10-25x (156-327ms TTFB) on any shared-prefix prompt, which covers every multi-turn chat continuation.

Files (mirrored byte-identical both repos): c_engine.py, pytorch.py, model.c, main.c, veritate.h, hybrid.c, hybrid.h, fsutil.c, portability.h; docs hybrid_trunk.md, state_cache.md, inference_brain.md.

---

## 2026-07-13 evening — cardinal testbed: 800 MHz clamp, state cache, batched prefill

Context: user question "why does cardinal train slow / can it be an inference machine" answered with measurement, then the measured constraints drove two engine features. All builds by dispatched Opus 4.8 agents under the preflight/ROE contract, reviewed and re-measured here; every change env-gated and byte-identical when off; nothing committed (engine diffs are local, cardinal tested via an out-of-tree `~/engine_ab2` copy so its repo checkout stayed clean).

Hardware forensics (cardinal-01, OptiPlex 7070, i7-9700T, AVX2/no-AVX512):
- All 8 cores pinned at 800 MHz: `intel_pstate/no_turbo=1` (BIOS-locked), HWP `base_frequency=800000`, `cpuinfo_max_freq=800000`, package 41°C (not thermal). Signature of the Dell power-adapter/BIOS clamp. Kernel governor can't raise it; needs `no_turbo=0` as root (may be MSR-locked), the BIOS Performance page, or the power brick.
- Decode is bandwidth-bound, not clock-bound: threads 1→7 = 24.6→18.5 ms/byte (1.33x only); single-core numpy read ~4.1 GB/s; engine at 7T ≈ 12 GB/s effective streaming (233 MB fp16 bin / 18.5 ms) — consistent with single-channel DDR4. Training (batched GEMM, weight reuse) was the clock-bound victim; decode barely cares.
- Levers that follow: fewer bytes (int8/sub-int8 bins), byte reuse (prompt cache, batched prefill, drafts), second DIMM (~2x ceiling), clock fix (helps prefill compute + training only).

Feature A — persistent prompt/state cache (`veritate_engine/src/state_cache.c` 359L + `fsutil.c` 152L shim, hooks in `model.c` forward/load, `VERITATE_STATE_CACHE[_MB|_LOG]`):
- Snapshots the six resumable `hybrid_t` fields (kv_k/kv_v compact to L rows, rec_state, conv_ring, slot_count, pos) + logits + final hidden after prefill; twin rolling hashes (name + header check) keyed by FNV(path+size+mtime) model_id; longest-prefix restore; trace-safe ceiling n-1; LRU eviction; atomic tmp+rename.
- Cardinal, chat80m fp16, 1 KB prompt, 7T: cold 15.2 s (miss + stored L=1024, 27 MB) → warm **0.49 s** (restored L=1024), stdout cmp-identical. Warm-extend (1536-byte prompt over the 1024 snapshot) also cmp-identical to cold.
- tests/engine/test_state_cache.py: restore==cold, extend==cold, model-change invalidates, eviction cap — 4/4.

Feature B — batched prefill (+B2 threading) (`hybrid_prefill`/`prefill_local_block`/`prefill_recurrent_pos` in `hybrid.c`, `kernels/x86_64/matmul_prefill_avx2.c` 3-dtype, `hybrid_mm` j-split threading with int8 quant hoisted to `pf_qx`, `VERITATE_PREFILL_BATCH`, `-ffp-contract=off`):
- Bitwise by construction: batch over positions/output rows, never the k-reduction; GLA scan, conv ring, attention, per-row int8 quant stay in sequential order. Verified on the i7: sequential == B=32 AVX2 == B=32 scalar == threaded, cmp-identical; composes with the cache (restore 1024 + batched 512 extension == cold).
- Measured (1 KB prompt): sequential 7T 14.9 s wall / 115 s CPU; batched 1T ~20 s wall / 50 s CPU (2.3x less work, compute-bound on one 800 MHz core); batched+threaded (B2) 11.7 s wall (bench: prefill 20.8 s → 11.7 s, 1.78x; decode unchanged 16.4 ms).
- Honest residual: only 1.27x over threaded-sequential end-to-end. Scaling probe (256/512/1024 bytes: 2.9/5.6/11.7 s) is LINEAR ⇒ a ~11 ms/position serial floor (rmsnorm/attention dots/recurrent per-position path), not O(n²) attention. Next lever = profile that floor (batch the attention scores, SIMD the scalar dots, or accept until the clock unclamps — at 3.5 GHz this floor divides by ~4).
- tests/engine full suite 15/15 (5 new prefill tests incl. threaded==single and i8 hoisted-quant parity).

Held for operator: commit the engine diffs (state_cache/fsutil/hybrid/model/build.sh/tests/docs — all local-only); export chat200m as int8 hybrid for cardinal (fp16 541 MB → ~271 MB streamed ≈ up to 2x decode there); rebuild production binaries both boxes after commit; physically check the power brick / BIOS turbo + whether the box has one DIMM.

## 2026-07-13 — inference optimization run (handoff executed; decode speed + threading + one live bug)

Context: `inference_optimization_handoff.md` executed end-to-end, overseer model — every change built by a dispatched Opus 4.8 agent under the preflight/ROE contract, then independently reviewed and re-verified here (adversarial review returned SHIP-WITH-FIXES; all findings closed same-day). Batch=1 byte decode is GEMV = memory-bandwidth-bound, so the levers are fewer bytes streamed (int8), fewer engine round-trips (trace-off frames), more bytes per forward (lookahead), and threading to the bandwidth knee — never past it. The 800M pretrain (MPS) and live dashboards were never touched; all verification CPU-only; cardinal only via its /tmp harness + tunnel.

**#1 int8-AVX2 kernel (engine).** x86 int8 previously fell back to scalar. New `hybrid_matvec_i8_avx2` (`kernels/x86_64/matvec_f32_avx2.c`): vpmovsxbw + vpmaddwd + vpaddd (vpmaddubsw rejected — int16 saturation at real ranges), int32 accumulation is order-free so bitwise parity vs `hybrid_matvec_i8_scalar` holds by construction; float fold parenthesization mirrors scalar exactly. Parity verified on 17 shapes, on cardinal natively and under Rosetta. Inert until a model is served int8: activation = `export_checkpoint("chat80m", 48000, dtype="int8")` + engine rebuild. Expected ~2x on cardinal on top of the shipped 7x AVX2.

**#2 trace on/off serving (engine + c_engine.py + routes).** 10th optional stdin header field `trace` (default 1 = legacy-compatible both directions); trace=0 emits 16-byte FFRM frames instead of ~300KB TFRM. Measured Mac: 3.00 -> 2.38 ms/byte, output byte-identical. `/generate` (Generation page) keeps trace=1; chat paths default off.

**#3 prompt/n-gram lookahead decode (pytorch.py).** LLMA-style: `bytes.rfind` suffix match drafts up to 16 bytes, ONE batched forward verifies; every emitted byte sampled at its true-prefix logits. Wired into `/hybrid/chat` via `stream_fast(mode="lookahead")`. Greedy output byte-identical to `stream()` on CPU — proven by `tests/mri/test_lookahead_parity.py` (5 tests; the parity assert was demonstrated to FAIL under an injected off-by-one at the accept boundary, so it genuinely guards the window math). 1.21x blended prose, up to 4.94x on quoted/RAG-style replies. Honest scope: byte-exactness proven on CPU only; MPS longer-seq GEMMs are not bitwise-guaranteed, so drafting is gated to `LOOKAHEAD_DEVICES = ("cpu",)` — retry condition (measure MPS parity when the 800M frees the GPU) documented in `inference_brain.md`. `fast_byte` events now carry the C path's core fields (`argmax_byte`, `T`, `backend`).

**Multi-core, part 1 — spin-then-park pool (threadpool.c).** The handoff's open "use all cores" problem root-caused: ~61µs condvar dispatch per matvec swamped the win. Rewrite: workers spin briefly then park (seq_cst Dekker handshakes, single-dispatcher invariant now stated in the file). M3: 0.68 ms/byte @8T = 2.56-2.97x over single-thread, bitwise-identical output at every thread count (`test_hybrid_threaded_matches_single_thread`).

**Multi-core, part 2 — the thread count is now measured, not hardcoded.** `HYBRID_MT_DEFAULT=8` deleted. `hybrid_load` runs a one-time calibration on the real loaded weights: 1,2,4,..-ladder up to pool_size-1 (dispatcher keeps a core), median per rung over direction-alternating passes under a 350ms wall budget, pick = diminishing-returns knee (stop when a rung buys <13%). The knee is bandwidth/contention-driven so only on-box measurement finds it; the over-threaded collapse rungs are structurally never picked. Measured: M3 auto=8/16, p50 0.492 vs 0.471 best manual (1.045x, 30-run stress zero bad picks, calib cost 273-423ms across model sizes); cardinal auto=4-7 NEVER 8 (forced 8T = 48.99-62.00 ms/byte collapse; auto = 2.21-2.56, ~21x better than the collapse, 2.5x over 1T). `VERITATE_HYBRID_THREADS` remains as explicit override; `VERITATE_HYBRID_CALIB_LOG=1` prints the rung curve. Parity: row-split is bitwise at every count, so the pick can never change output — `test_hybrid_auto_matches_single_thread` added (suite now 6/6). Bonus addenda from review: dead `pool_worker_t.alive` deleted, `hidden > V_MAX_FFN` now refused at model load (was a silent int8-activation-buffer overflow waiting for a wide-hidden model).

**API MRI opt-in (hybrid_routes.py + docs).** External clients can request per-byte MRI telemetry, exposed as a strict SUPERSET (child/expansion) of the OpenAI chat API — never a reshape. Two OpenAI-shaped entry points: `"mri": true` on `/v1/chat/completions` (flag-gated; absent/false is byte-for-byte the standard behavior) and a dedicated sibling route `/v1/chat/mri` (identical request body + `_resolve_route` model routing, streaming on by default, 400 on a cloud/teacher model that has no MRI). The chat-page twin `/hybrid/chat` takes the same flag. The "child of OpenAI" invariant: every response stays a valid `chat.completion` / `chat.completion.chunk`, and the MRI data rides as ONE additive top-level `mri` key, never inside `choices` — so an off-the-shelf client reads `choices[].delta.content` and ignores the unknown key (streamed frames are valid chunks with an empty delta), while an MRI-aware client reads `chunk["mri"]`. Documented in `documentation/api/external_api.md` ("receiving MRI telemetry", plus the `/v1/chat/mri` entry); 13 tests in `tests/mri/test_mri_optin.py`, 27 with the OpenAI-chat suite.

**Production incident, root-caused + fixed: chat_traced stdin desync.** Symptom (live, both dashboards eventually): chat/generation replies degenerate into digit soup ("16 16 16 -16..."). Root cause chain, each step proven: (1) the persistent `chat_traced` subprocess reads the prompt with `fgets(prompt_line, seq+4)`; (2) a prompt line over ~seq+2 bytes (chat-tab history does this routinely) leaves residue in stdin; (3) the subprocess is then permanently one line behind — it parses every subsequent HEADER line as the prompt, and the model faithfully autocompletes `"0.7000 40 200 -1 -1 - 0 0.0000 0 1"` = the digit garbage; (4) every desynced turn still ends in a valid TEND, so the existing `_last_clean` guard never fires — the poisoning is invisible. Diagnostic tell: first-frame `real_len` = header-string length (32) for a 2-byte prompt. Fix, both sides: python tail-clamps the prompt payload to min(fgets capacity, seq − min(max_new, seq/2)) — the second term reserves reply room, closing the adjacent failure where a near-window prompt yields an empty reply (engine budget is `S - n`, no window slide), and tail-clamping keeps the newest context so outputs are unchanged for everything that already fit; engine gains a `fgets_drain` helper at all 5 serving-loop protocol reads so ANY over-long line from any client stays in sync (defense in depth; needs rebuild). Clamp verified against the EXISTING shipped binary: 1800B prompt turn generates its full reply (real_len equals the clamp formula at max_new 8/200/4096), following 2-byte turn back at real_len=2, pid unchanged. Regression tests in `tests/mri/test_c_engine.py`. Remediation note: a poisoned live subprocess stays poisoned until the C backend is reloaded (model dropdown unload/re-pick) — the code fix prevents re-poisoning after the next server restart.

**Verification (independent reruns, not agent claims):** `tests/engine/test_decode_parity.py` 6/6; full `tests/mri` 145/145; desync repro (isolated subprocess) healthy end-to-end; live-server curl checks on the Mac dashboard clean.

**Held for operator:** commit/push working tree; rebuild engine on Mac + cardinal (activates int8-AVX2, auto-calibrated threads, stdin drain; cardinal sequence: `touch /tmp/veritate_maint` -> pkill dashboard -> build.sh -> `rm /tmp/veritate_maint`); restart dashboards (activates the python clamp + serving changes); reload cardinal's C backend (clears the live poisoning); re-export chat80m int8 to activate the int8 path; #4 Medusa MTP heads deliberately NOT started (canonical-trainer change, operator decision).

## 2026-07-13 — IDEA 2 external memory: natural-query boundary measured, top-k lever falsified, research consolidated

Context: the 800M pretrain is the priority run (background, dashboard-launched); this section is the memory-research work done alongside it. Zero production load — chat200m runs on MPS locally, and the Carpathian production ollama models (qwen2.5:7b/14b-instruct) stayed loaded and untouched the whole time (verified `ollama ps` before/after: both 100% GPU, TTLs unchanged). The only process ever killed was my own teacher-pair generator.

**Natural-query end-to-end (the last open boundary).** Every prior memory eval used heuristic "What is X?" queries because chat200m cannot write questions (it parrots input). Fix: use ollama qwen2.5:7b-instruct (read-only, one pass) to generate 1211 natural (question, answer) pairs over 31 topics. Trained a key head on 26 topics, tested on 5 FULLY held-out topics (nutrition_food, philosophy_religion, physics, programming, psychology) over the 1e5 fineweb store. Retrieval (held-out, natural): fineweb distractors baseline recall@1 0.152 / @5 0.234 -> trained 0.409 / 0.684; +train-facts (hard) baseline 0.135 / 0.228 -> trained 0.322 / 0.608. End-to-end (n=100, hard store): retrieval recall@1 0.300 -> grounded_acc 0.140 vs bare_acc 0.010. Read: the head transfer is real (~3x baseline on unseen topics, natural free-form queries) but natural queries are ~2x harder than heuristic (recall@1 0.50 -> 0.32 hard), and end-to-end grounding (14x bare) is capped at 14% by retrieval recall@1, not by the generator. This is the honest downward recalibration of the earlier heuristic-query numbers. Code `experiments/v2/memory/{make_teacher_pairs,eval_teacher}.py`; ledgered `successes.md` 2026-07-13.

**Top-k injection lever FALSIFIED.** Obvious next step: recall@5 (~0.63) is ~2x recall@1 (~0.37) and serving already injects up to CTX_LEAVES=3 leaves, so inject the top-k leaves and convert coverage into grounding. Measured grounded_acc on chat200m at k=1/3/5 (held-out natural, hard store, n=100): 0.130 (top-1), 0.120 (top-3), 0.080 (top-5). More leaves monotonically HURT. Cause: a 200M copy-limited generator cannot select which of k leaves holds the answer; the k-1 distractors mislead it. This localizes the end-to-end bottleneck to top-1 PRECISION + generator disambiguation (not recall@k coverage) and prevented a net-negative serving change — serving defaults TOP_K=3, and raising it on the "more context helps" assumption would have degraded the small local model. Retry-gated behind the 800M (a stronger generator may disambiguate multi-leaf directly; re-run `eval_teacher_topk.py` first) or a candidate re-ranker (logged in `ideas.md` under IDEA 2). Code `experiments/v2/memory/eval_teacher_topk.py`; ledgered `failures.md` 2026-07-13.

**Retrieval-recipe lever also FALSIFIED (hard-negative mining).** With the bottleneck localized to top-1 precision on the hard store, tried the textbook fix: mine the hardest distractors (top-scoring vs the batch queries, from a random 4096-window each step) instead of sampling 512 random store distractors. No lift — hard-store recall@1 baseline 0.351 -> mined 0.327 (within run-to-run noise), recall@5 0.620 -> 0.637, fineweb flat/up. Cause: the ceiling is the FROZEN trunk's feature resolution on semantically-competitive real facts, not the negative recipe; no sampling strategy recovers resolution the features never had. Two independent recipe levers now dead (generator-side top-k, retrieval-side hard-neg), both pointing at the same fix — a richer trunk, i.e. the 800M. This is why the honest position is "mechanism validated, remaining gain is scale," not "needs more tuning." Code `experiments/v2/memory/eval_teacher_hardneg.py`; ledgered `failures.md` 2026-07-13.

**Consolidation.** The 2026-07-11..13 memory findings (needle recall, sub-quadratic drill-down, FAISS trillion-feasibility, real-content + natural-query transfer, RARS kill, low-diversity overfit, top-k kill) lived only as scattered ledger entries. Folded them into one honest research doc: `developer_documentation/research/external_memory_retrieval.md` — the exact-recall tier, sibling to `long_context_memory.md` (the IDEA 1 always-on gist tier). Indexed in `research.md`. The doc states the productionization plan: the neural retriever plugs into serving entirely behind `retrieve()` in `veritate_mri/routes/hybrid_routes.py` (same `(previews, scores)` signature; nothing in `build_prompt`/`_render_local`/the streaming path changes), needs a persisted head + a KB key store, and for the small local model must inject top-1 not top-k. Staged, not shipped — it introduces a neural embedder into a deliberately-lexical (BM25) path, and real end-to-end quality waits on the 800M regardless.

800M pretrain healthy throughout: step ~8950, loss ~0.79-0.86 (below chat200m's 0.812 at matched region), ~4,100 tok/s, 0 DUMP FAILED.

## 2026-07-09 17:11 UTC — eqmom crosses positive vs SPY (first arm ahead of its benchmark at a rebalance-clean mark)

eqmom marked up +2.43% at/after the US open ($9,856 -> $10,095.99) as its semis/AI book rallied, taking it to +$18.52 vs SPY-hold — the first time any forward arm sits ahead of its buy-and-hold benchmark on a clean (non-BTC-noise) basis. No rebalance (monthly arm, recent_acts=0); this is the Monday book marking, not new trades, so judge it monthly not on one green session. Same session bounce recovered most of the day-2 semis drawdown. Other arms: ml7 +$9 vs BTC-hold (flat book), xsmom base -$334 / gated -$164 (BTC firm), news still bleeding at -$310 (-3.1%, $134k turnover, -$176 even at 10bp/side; forward-disprove trajectory intact, kill recommendation standing). All 6 runnables healthy, server continuous ~38h, no parameter changes.

## 2026-07-08 — chat200m launch + Round 2/3 verdicts (RAG gap isolated, identity SFT's hidden cost, serve stack hardened)

Execution of the parallel directive: fix self-talk, keep testing, launch the next size, tune the engine, compile research. Four Opus 4.8 agents built; lead verified every claim independently on the live server before acceptance.

### chat200m pretrain — LIVE
- Launch: `POST /trainers/run` id=veritate_200m, name **chat200m**, trunk=hybrid (270,510,336 true params at the 200m manifest shape — dense estimate was 202M), muon, bf16, seq 1024, batch 24, **n_chunks 4** (the sweep's find: nch 1->2->4 = 8.4k -> 11.5k -> 14.2k tok/s, +68%; act_ckpt costs 17% and buys nothing at 4.3GB peak; batch >24 flattens; adamw is +16% step speed but loses to muon's 1.6x byte efficiency), 20,400 steps ≈ **2.006B tokens ≈ ~2 days** (plan estimated 3.5-5.5 at 5-7k tok/s; measured 14.2-14.3k sustained).
- Mix (9 stems, every ledgered lesson from step 0): fineweb_edu 0.37, openwebtext10g 0.365, chat_v1/v2/v3 0.05/0.04/0.03, py_code_v1 0.06, chat_recall_v1 0.04 (retry condition a), **grounded_v3 0.025** (Round 2 fix), chat_identity_v1 0.02 (retry condition b). Worst repeat factor ~1.3 epochs (grounded_v3); knowledge stems <=0.15 epochs. state_carry left OFF (its 10M validation never ran; not riding an unvalidated flag on a 2-day run).
- Gates at launch: MPS stability smoke at the real shape PASS (all finite); first checkpoint step_1000 dump battery PASS by artifact count — **all 14 families present incl. generation.json** (the multicorpus stem fix holds on a 9-stem mix). val 1.141 at step 1500. Monitor armed (2k-step milestones, non-finite, dump failures, terminal).
- 200m manifest defaults updated to the measured config (announced trainer edit, appended to `developer_documentation/trainers/upstream_changeset_2026_07_07.md` — Sam mirrors upstream).

### Round 2 verdict — RAG: wiring proven, model can't read, and the makeup gate caught an erosion
- Platform path verified end-to-end: KB upload -> BM25 -> `context:` block assembled exactly as trained (chat_v1/v2 carry 1100+ grounded examples each; template byte-identical). Retrieval found the right fact on EVERY call.
- The model ignores it: alien-fact extraction (invented entities) **1/4 greedy at 51000 AND 48000** (48000 sampled 4/12). The skill was never trained in — trace dose, common-surface facts only. 121M shows the same transfer gap as 10M. The one hit ("capital of France is Paris") is parametric, not read.
- The needle A/B (the gate skipped at identity-round time): conversation-copy **0.917 -> 0.167 @190B** (0.250 -> 0.000 @475B; contradiction 0.25 -> 0.00 — the recall-SFT signature at a fraction of the dose) while val improved 0.647 -> 0.644. Ledgered: `failures.md` 2026-07-08. Val is blind to skill erosion; only the falsifier battery counts.
- The decisive twist: persona-context application is what the identity SFT actually taught. 51000 + persona line = name 6/8, maker 7-8/8 at temp 0.5; **48000 + the same line = 0/6** ("Your name is Jack Thompson", "Yes, I am human"). Neither checkpoint dominates: 48000 copies conversations, 51000 wears identity. Serving stays on **51000**; 48000 is the designated base for the repair round.
- Fix built same-day: **grounded_v3** corpus (38.0MB train / 1.9MB val, 145,454 convs, 4 families: single-fact 38%, real-chunk-with-distractors 27%, multi-fact selection 23%, honest-miss refusal 11% with verbatim anchor "can't find that in the provided context"; invented entities so parametric knowledge can't fake it; 100% extractive-answer audit, lead re-audited 500/500). Builder: `experiments/v2/rag/build_grounded_v3.py`. In the 200m mix now; 80M repair SFT (from 48000, grounded 25% + identity 15% + chat, ALL gates incl. needle >= 0.8) queued behind the 200m on GPU.

### Round 3 + serve stack — every fix independently verified live
- **Self-talk dead:** root cause = single-marker stop (`<|end|>` only) while the model often opens the next `<|user|>` turn without it. Multi-marker stop (end/user/assistant) via one shared wrapper over BOTH backends; UI partial-marker trim; autocomplete passthrough regression-proven. Lead battery: C 6/6 clean, autocomplete streams markers untouched.
- **/hybrid/chat hardened** (front door had none of the Generation tab's guards): rep defaults 256/0.5/16 now passed on both branches (the "complex and complex and" loop repro is dead), control-byte/0x01 sanitize (was corrupting JSON responses), **LOCAL_PERSONA prepended for local models** (name 0/8 -> 4-6/8, maker 3/8 -> 7/8 deployed), and RAG prompts budgeted to `seq - 256` (facts trim first, drop lowest-scoring, persona/message never) — empty-reply rate ~50% -> **0/8** (root cause: 3x480-char previews overflowed the 1024-byte window; engine returned zero bytes).
- **Temperature/int8/latency:** see the 2026-07-07 v13 round 2 section (temp dial live-verified 6/6 distinct at 1.0, deterministic at ~0; int8 gates green; 200m-shape p50 1.7-1.9 ms/byte int8 P-core = target met).
- Identity-vs-temperature matrix (the measurement that exposed the shallowness): bare name 0/8 at temps 0.3/0.5/0.7/1.0; maker 0.88/0.50/0.75/0.25. The Round 1b "PASS" was conditional on near-greedy decode + the UI persona — now stated honestly in the journal and ledger.
- research/ folder landed (5 papers + index, every figure traced to ledger evidence; lead fixed two stale flagship-checkpoint references post-review).

### Queue / open
- 200m milestones (monitor live); knowledge eval battery (capitals/simple facts) to build BEFORE the 2B decision gate.
- 80M repair SFT queued (GPU frees ~2026-07-10); M3 rehearsal A/B result files + state_carry 10M validation still unverified (campaign doc items).
- For Sam: mirror the trainers upstream changeset (incl. 200m manifest); market-models delete decision (88G); PyTorch-backend live re-confirm of the hybrid-chat battery when CPU quiets (code-path-proven, C-verified).

## 2026-07-07 — v13 round 2: temperature fix + int8 mode + 200m latency target

- **Temperature bug FIXED (v13 sampler was near-greedy at any temp):** hybrid logits cross the sampler at x1024 telemetry scale; `fp = logits/temp` therefore sampled at an effective temp/1024. Fix: `t_eff = temp * VERITATE_HYBRID_LOGIT_SCALE` for hybrid only (dense semantics untouched; v9 golden still byte-identical). Verified: greedy parity still 192/192 (fp32 + fp16); diversity at temp 1.0/0.7 x 3 prompts now 8/8 distinct samples with mean pairwise byte-difference 0.80-0.94, matching PyTorch top-k sampling on the same checkpoint (PT: 0.940/0.926 at 1.0; 0.917 at 0.7 vs C 0.875, n=32 apples-to-apples). The fold also lands the repetition soft penalty at its designed scale-free nat units; **loop battery 0/12** (12 chat prompts, rep 256/0.5/16, temp 0.7, falsifier = any 16-byte window x3 in 200 bytes) — replies diverse, most self-terminate at `<|end|>`.
- **int8 v13 mode (dtype=2) SHIPPED:** per-output-row symmetric int8 weights + fp32 scales for the five big matrices (everything else fp32 in-bin), dynamic per-call activation quant (absmax/127, shared `hybrid_quant_act` so scalar and NEON sdot see identical int8 inputs), int32 sdot accumulation → scalar/sdot **bitwise identical**. Gates: bpb vs fp16 (step 51000, 20x1024) fineweb **+0.0040**, chat_v1 **+0.0020** (< 0.005 gate); greedy transcripts coherent; v9 fixture still byte-identical; 12/12 repo tests. Bin 126.5 MB (vs fp16 243.5). Kernel: 768x3072 in **0.030 ms** = 3.9x fp16 (bandwidth ceiling). Threading: int8 threads only above 8M elements — measured 1T >= 4T below that (pool dispatch costs more than sdot saves).
- **200m-class target (<= 2 ms/byte): MET with int8 on P-cores, single-threaded.** Random-weights v13 bins at the 200m manifest shape (h1024, 4+16 blocks, ffn 4096, heads 16, seq 1024 = 270.8M params by the same convention that makes "80m" 121.75M): int8 p50 **1.86-1.91 ms/byte** (1.73-1.79 lighter load), p95 (boundary bytes) 6.6; fp16 4T just misses at 2.06-2.11. At 121.75M, int8 1T is now the fastest config: p50 1.17 ms, p95 3.35, 660 tok/s — boundary-byte cost halved vs fp16 (recurrent stack is pure matvec, sdot's best case). E-core class: int8 ~3.2-3.4 ms p50 at 200m shape — not ms-class on efficiency cores yet.
- Serving bin `models/chat80m_80m/veritate.bin` (fp16 step 51000) untouched; test bins in scratchpad. Engine rebuilt via build.sh (server's next C-backend respawn picks it up). No dashboard restart (per constraint). Docs: hybrid_trunk.md gained the temp-fold contract, int8 layout + gates, and the two-scale benchmark tables.

## 2026-07-07 — v13 hybrid engine build (C backend serves chat80m)

Commissioned build: make the C engine serve the hybrid-trunk byte model at ms-class per-byte latency, backwards compatible. All five staged gates passed.

- **Format + exporter (S1/S2):** v13 spec at `developer_documentation/engine/hybrid_trunk.md`. Additive version byte; header ext carries dtype/enc/global/dec/stride/slots/conv_kernel/state_rule + a baked 256-entry boundary table (Python `isalnum` semantics, not re-derived in C). Tensors in PyTorch [out,in] row-major, tied lm_head enforced. `export_checkpoint` routes `trunk=hybrid` (gla only) to the v13 writer; patched/looped/recurrent keep the refusal; the ternary path gained the same dense-only guard it was missing. 5 exporter tests green (`tests/export/test_export_v13.py`).
- **Decoder (S3):** `veritate_engine/src/hybrid.c` — fp32 scalar reference of the full stack: embed → 2 enc attn blocks (fp32 KV bounded by seq) → boundary-gated slot path (per recurrent block: raw-qkv conv ring, per-head decay from a_proj, rank-1 state update S←a·S+k⊗v, o=q·S, per-head o_norm, silu gate) → scatter-add on boundary bytes only → 2 dec blocks → tied head. Greedy argmax on fp32 logits; the x1024 int32 view feeds the existing sampler/telemetry surface. Parity: 192/192 bytes vs PyTorch greedy (chunkwise-vs-sequential recurrence rounding never flipped an argmax; min top-2 gap seen 0.0089).
- **Quantization gates (S4):** fp16 = parity 192/192 + bpb identical to 4 decimals → shipping default (243.5 MB). int8-per-channel simulated in PyTorch: +0.0006/+0.0009 bpb, coherent greedy → quality gate PASSED, C kernels deferred (documented next lever for low-power). Kernels: 16-partial-sum matvec, scalar/NEON bit-identical (fp32 + fp16 + exhaustive f16-convert check in the engine self-test), `-ffp-contract=off` pins the pairing. Threaded matvec (row-split, bitwise-safe, default 4 threads, `VERITATE_HYBRID_THREADS` override).
- **Benchmarks (S4):** 64-byte generations, M3 Ultra: fp16 4T p50 1.08-1.12 ms/byte p95 4.6 (520-600 tok/s); fp16 1T p50 1.8; fp32 1T p50 2.1. E-core-pinned fp16: p50 7.2 (1T) / 5.1 (4T) ms/byte — honest low-power estimate ~5-7 ms today, int8 the remaining ~2x. Full-context ppl decode p50 2.7 ms p99 8.9 ms.
- **Platform (S5):** `model_load` v13 dispatch; forward/decode/sampler/ppl/trace branch on `m->hybrid`; chat_traced emits format-complete 16-layer frames (local blocks real, global sections zeroed, slot residuals on boundary bytes); `readers/bin.py` labels v13 and exempts it from the act_boost gibberish heuristic; build scripts carry `-DV_SEQ=1024` + hybrid TUs; e2e `/generate?backend=c` streamed a coherent `<|end|>`-terminated reply from the live dashboard.
- **Pre-existing bugs fixed:** (1) dense `score_dot_v` writes a hardcoded 64-byte head regardless of head_dim → heap corruption on head_dim≠64 fixtures; surfaced as build-order-dependent transcripts + a malloc SIGTRAP, root-caused with ASan; dense loads now refuse head_dim≠64. dla_top also zero-inits its heap (was UB with NULL byte_direction). (2) chat_traced flattened prompt newlines to spaces (c_engine.py), wrecking chat-template framing — now escaped 0x01 across the pipe; multi-line template output matches the PyTorch newline reference 64/64. (3) `compute_residual_stab`/`trace_top_predictions` NULL-deref on models without int8 embed (hit by v13, latent for int4).
- **Compat evidence:** committed head_dim-64 v9 fixture + golden transcript (`tests/engine/fixtures/`); pristine-HEAD build vs new build byte-identical; 0/20 stress failures; ASan clean; all six build variants (LTO/non-LTO/link orders) agree on the valid fixture. Repo tests 7/7 green. Parity smoke at `SMOKE_RESULTS/v13_hybrid_parity_smoke.py`.
- **Open items:** live server needs a restart to pick up the new `c_engine.py` newline escape + exporter route (old code is stale-but-safe: flattened prompts, hybrid export refused via the dashboard button until then). int8 v13 storage mode unimplemented (quality pre-validated). MRI dashboard depth for global blocks is shallow by design (slot-level telemetry doesn't fit the dense per-layer frame).

## 2026-07-07 — platform integration fixes (chat80m servable in the dashboard)

The hybrid-trunk flagship now serves chat end-to-end in the dashboard: Generation tab chat mode → `/generate` on the PyTorch backend → platform chat markers → stop at `<|end|>`. Verified live: chat streams through `/generate`, `/meta` shows 121.75M params. Seven fixes landed:

1. **Trunk dispatch in the load factory** (`veritate_core/load.py::_load_variant_trunk`): `load_from_state_dict` now branches on `training_args.trunk` (patched/hybrid/hybrid_moe/looped/recurrent) to the research trunk classes, `strict=True`, `state_rule` threaded. Canonical/dense path regression-verified unchanged. This is the preflight-11a-sanctioned branch point; `needle_bench.py`'s local `_build_trunk_model` is now redundant (left in place).
2. **Export trunk guard** (`veritate_mri/training/export.py`): `export_checkpoint` refuses trunk != dense with a clear error — the engine `.bin` format is canonical-only. Previously it silently produced a structurally-garbage v9 bin from the hybrid checkpoint (`models/chat80m_80m/veritate.bin`, renamed `veritate.bin.invalid_hybrid_export`).
3. **Multicorpus stem resolution fixed** (`veritate_mri/training/save.py`): a mix spec `"stem1:w1,stem2:w2"` previously resolved the corpus stem to the LAST weight token (`"0.20"`) via `rsplit(':')`, silently skipping the generation dump for the whole run — skip errors go to the server log only, no `DUMP FAILED` in the run log. Now picks the highest-weight stem; unit-checked on 6 spec forms.
4. **Generation-tab chat framing** (`veritate_mri/web/index.js`): chat mode now wraps prompts in the platform markers (`<|user|>`/`\n<|assistant|>`/`<|end|>`) matching `/hybrid/chat`'s `PROMPT_TMPL` and all current chat corpora, replacing a stale ChatML wrap. ChatML kept in the response-strip list as legacy defense (`wrapChat`/`stripChatResponse`).
5. **Chat stop sequence** (`veritate_mri/routes/backends_routes.py::_chat_stop_seq`): generation stops at `<|end|>` for platform-marker prompts, `<|im_end|>` for legacy ChatML, streams to max_new for plain prompts.
6. **Hybrid telemetry fix** (`veritate_mri/inference/backends/pytorch.py`): the attention-telemetry loop now skips layers whose captured qkv length != T (recurrent-mixer blocks pad to CHUNK=64 internally; attention weights don't exist for them), appending an empty per-layer entry. This fixed "generation failed: shape ... invalid" on hybrid models.
7. **Capabilities**: `chat80m_80m` marked chat=trained (step 48000).

Non-canonical trunks serve CPU-only through the brain (~1 B/s at 121M with full telemetry); the fast path is a future engine format. Component docs updated: `inference_brain.md`, `export.md`, `save.md`, `generation_tab.md`, `model_patched.md`.

## 2026-07-05 — chat80m launch + E6 verdict + long-context memory build

- **E6 SLM verdict (kill).** Matched-arm check confirmed single-delta (identical trunk/opt/corpus-sha/seed/steps vs `e2patched`; delta = `slm_ref=e1muon_10m_qat, slm_keep=0.6`). Tail-10 val 1.0638 vs 0.9763 — quality regression at equal steps, the second pre-registered kill condition; never reaches baseline-final val so byte-savings-to-target is zero (the 1.24x early-speed effect is real but reaches only a mediocre plateau). Per-eval val stdev ~0.035 on this arm — milestone evals lied in both directions mid-run; verdicts now use tail-averaged val as standing method. Ledgered in `failures.md` with retry conditions.
- **chat80m launch sequence.** E6 dumps clean → MPS stability smoke of hybrid at the REAL 80M shape (bs12/seq1024, 6 fwd+bwd steps, all losses/grad-norms finite, preflight 24d) → `POST /trainers/run` id=veritate_80m: trunk=hybrid, optimizer=muon, precision=bf16, corpus `fineweb_edu:0.68,openwebtext10g:0.15,chat_v1:0.04,chat_v2:0.04,chat_v3:0.02,py_code_v1:0.07`, 30k steps. Confirmed in dashboard state + config.json. ~16.3k tok/s → ~6.5h. Milestones: val 1.695 (2k) → 1.267 (4k) → 1.156 (6k) → 1.095 (8k) → 1.051 (10k). All 13 dump families present per checkpoint, zero DUMP FAILED. Corpus pre-launch check: all six stems on disk train+val (~30 GB); chat template verified at byte level in all three chat corpora (the v2/v3 "missing system tag" flag was a false alarm — casual/instruction registers legitimately omit it).
- **Wave-4 memory mechanisms (agent-built, lead-verified).** `state_rule` on RecurrentMixer: `gla` default proven bit-identical (param+loss diff 0 pre/post; independently re-verified at 80M shape), `delta` = chunkwise Gated DeltaNet via nilpotent WY inverse (exact to 7.7e-7 vs naive oracle; fp32 under autocast-off; fixed shapes, no MPS-forbidden ops), `pinned` = P decay-exempt slots + learned salience writer. Dump battery 12/12 at real 10M shape both rules. Threaded: RecurrentBlock, VeritateRecurrent, VeritatePatched; dashboard flag `state_rule` (vanilla_trainer.py — synced checkout, diff quoted to user for upstream).
- **Streaming/state-carry inference (agent-built, lead-verified).** `forward_streaming(tokens, states)` on VeritatePatched/VeritateRecurrent + `RecurrentMixer.forward(x, state, return_state)`; carries `{state, conv-tail[, pins]}` across fixed windows. Window-carry == one-pass BITWISE (0.0) on all three rules — the conv tail at the seam was the correctness crux. Default paths bit-identical (36 comparisons, 0.0); live chat80m run untouched (verified: only additive diffs, run healthy through the edit). needle_bench auto-detects streaming; past-window distances now real measurements (`streamed:true`). Documented approximations: window-local positions, within-window local attention — only the constant-size global state crosses windows.
- **Needle benchmark.** `experiments/v2/longctx/needle_bench.py` + README: plants a fact at controlled byte distance in template-exact synthetic chat, greedy-decodes the answer, exact-match recall → degradation curve JSON; coherence probe (two facts, contradiction rate). Plumbing validated end-to-end on `e5hybrid_10m_qat` (recall 0.0 as expected for a non-chat checkpoint; labeled plumbing-only).

## 2026-07-04/05 — architecture campaign verdicts (E2-E7) + E6 launch

All runs dashboard-launched (`POST /trainers/run`, `veritate_10m`), 12000 steps, fineweb_edu, seed 0, full 14-artifact dump suite verified per checkpoint. Full evidence in `successes.md` / `failures.md`; per-trunk contracts in `developer_documentation/platform/model_{patched,recurrent,memory}.md`.

- **E2 patched CONFIRMED 1.82x** wall-clock to dense-final quality (0.9776 final vs 0.9990; ahead 105/120; 128% dense throughput; 49% more params at ~dense per-byte FLOPs). Mid-run incident: 7/14 dumps silently missing -> root cause fixed-slot tensor vs short dump prompts; fixed (`S = min(slots, T)`), artifacts backfilled for every checkpoint, preflight 24d added, dump failures now print loud (`DUMP FAILED:`) in the run log.
- **E3 recurrent (constant-state) parity+**: per-step quality beats attention (0.9900, ahead 117/120), 18% slower per step unoptimized -> equal-wall-clock within the ±0.03 parity band. Meaning: attention is not required at byte level 10M, and decode state is O(1) in conversation length.
- **E4 -> E4b memory line CLOSED for recall**: per-window-reset training (E4) gave 0.0 recall lift (no cross-window training pressure); the fix (E4b: memory carried across the 4 contiguous chunks per step) trained clean and beat dense on val (0.9867) but STILL 0.0 exact recall; distance diagnostic shows a soft trace inside the 1024-byte trained horizon (win 0.667) turning harmful at 4096+ (0.167). Fast-weight recall at 10M is dead per pre-registered falsifier; the trunk survives as a context extender. NaN-at-real-width lesson ledgered (unit-norm k/q, chunk-mean inner grads; stability-smoke at REAL width now standing rule).
- **E5 hybrid = the campaign's winner**: patched local + recurrent global mixer on patch slots. Best final val of all arms (0.9707), 1.70x vs dense, 1.15x vs patched (ahead 98/120), 113% dense throughput, O(1) global state. Single seed; the beats-patched margin needs a second seed before external claims (agent_roe); beats-dense is far outside noise.
- **E7 looped: think-longer FALSIFIED**: params-matched (10.12M) weight-tied global stack, R~U{1..4} per training step. Beats dense (0.9920, ahead 111/120, 1.63x) but loses to patched AND hybrid on quality and wall-clock; paired R-sweep on the trained checkpoint (`SMOKE_RESULTS/e7_loop_sweep_smoke.py`) gives CE 1.0103/0.9975/0.9965/0.9983/1.0050/1.0142 at R=1/2/3/4/6/8 — peak at the training-mean depth, monotone degradation past it, no test-time compute scaling. Random-R training also makes single val evals untrustworthy (tail-20 stdev 0.0087). Not scaled further.
- **E6 SLM LAUNCHED** (`e6slm_10m_qat`): patched+muon student, frozen `e1muon_10m_qat` reference, train on top-60% tokens by excess loss, val unmasked for comparability. Loss path smoke-tested at real shape (bs32/seq512, MPS, grads flow) before launch. Training clean at 69k tok/s (~23% reference overhead, inside budget). Single delta vs `e2patched`; falsifier: <1.15x byte savings or quality regression at equal steps. Monitor armed (milestone vals, DUMP FAILED, non-finite-skip threshold ≥20, terminal status).
- **UI**: trunk + optimizer dropdown help now states the measured verdicts (hybrid best 1.70x; looped/memory honest caveats); recipes update after E6 lands.

## 2026-07-08 05:11 UTC — auto-resume CONFIRMED across a real re-exec + fee is not the whole story

The server re-exec'd at ~02:46 UTC (uptime now 2h25m, was 3h54m at 01:11) and news_main auto-resumed with no manual intervention — the fix is validated in the wild, and the periodic-re-exec hypothesis is confirmed. All 6 runnables live, ticks fresh, intel 3m. Honest counterfactual read (thin, ~a dozen round-trips): news has traded $29.8k notional since instrumentation and is -$113 lifetime (eq $9,886.91, -$180 vs BTC-hold). At the documented 10bp/side it would be only +$29.78 better (-$83) — **halving the fee does NOT flip the book positive; the trades are net-negative before the fee gap too.** Fees are a real drag but not the sole culprit on this window; the signal itself is not yet paying. Too early to conclude (needs weeks), but the instrument is doing its job: reporting the uncomfortable read, not a hopeful one. ml7 best relative ($10,021, -$10 vs BTC). eqmom flat (mkt closed, -$339 vs SPY). No rebalances, kill-watch clear, no parameter changes.

## 2026-07-08 01:11 UTC — news auto-resume holding + fee counterfactual's first live reading

News has stayed up and ticking (1m fresh) since the 21:11 resume-fix deploy, and executed its first post-fix trade sequence: ~$7.96k notional round-tripped back to cash (eq $9,932.63, -$17 this window, -$67 lifetime, -$130 vs BTC-hold). The fee counterfactual built yesterday now has real data: at 10bp/side the book would be +$7.96 better off (equity_alt $9,940.59) — first end-to-end proof the instrument works, though one round-trip is far too thin to read the signal. Note: server has NOT re-exec'd since the deploy (uptime 3h54m continuous), so the auto-resume has not yet been tested across a real re-exec; next re-exec is the confirmation. All 6 runnables healthy; quant books unchanged (base -$306, gated -$282, ml7 -$139 vs hold), no rebalances, kill-watch clear.

## 2026-07-07 21:11 UTC — news-death root cause: server re-execs, news never wired to resume (FIXED)

The recurring news_main deaths were NOT a news-thread bug (its _run wraps tick in except-pass and only exits on the stop event). Root cause: the dashboard server re-execs periodically (verified: current server uptime 3h25m, i.e. started 17:46 UTC, while news's last tick was 17:43 UTC — three minutes before this server instance existed; the 17:11-checkpoint restart ran fine until the re-exec took it down). The quant/intel arms survive re-execs via their `resume()` (ledger `auto` flag); news_trader was simply never wired for resume, so every re-exec silently gapped its record while the others kept ticking. Fix (mirrors xsmom): `start_thread` stamps `auto`+`resume_cfg` in the ledger, `stop_thread` clears it, new `news_trader.resume()` restarts from the stamp, and `register.register()` calls `nt.resume()` at boot. 22 news/register tests green; deployed via training-preserving soft_reload; news_main restarted and its ledger now carries `auto:true` + resume_cfg (qwen2.5:7b-instruct, gate 0.25), so it will auto-recover from future re-execs. (Supersedes the 17:11 "isolated thread death" read — that was incomplete.) What re-execs the server itself is still open (no cron/launchd/heartbeat does it; only the /lifecycle path, plus a parent/child PID pair suggesting a supervisor respawn) but is now moot for record continuity. All 6 runnables live. No strategy parameter changes.

## 2026-07-07 17:11 UTC — eqmom day-2 drawdown (-3.2%) + news thread restarted

eqmom's semis/AI-heavy book fell to $9,676.94 (-3.23% from $10k, -$376 vs SPY-hold) on its second session as semiconductors sold off; one bad session on a monthly-rebalanced arm, judged monthly not daily, kill-watch clear (1 rebalance, needs >=4). The news_main thread was found stopped (isolated death; server ran continuously and all five quant/intel threads kept ticking, ledger intact at $9,949.58) and restarted via /start with its captured config; teacher (Ollama qwen2.5:7b-instruct) confirmed reachable, model present, so scoring is live. Fee counterfactual still $0 (book has not traded since instrumentation; event-driven, no news break above gate). Second unexplained news-thread stop in ~a day; if it recurs, root-cause before another blind restart. Crypto arms unchanged (base -$385, gated -$378, ml7 -$247 vs a BTC-hold near its high). No parameter changes.

## 2026-07-06 17:11 UTC — eqmom's first book: the equity 12-1 momentum arm is trading

eqmom_main bought its first book at the Monday US open: 49 trades into a 49-name long-only 12-1 momentum portfolio (semis/AI-infra heavy: AVGO, AMD, MU, KLAC, LRCX, ANET, APP, GEV, plus energy/industrial names), equity $9,978.95 after entry fees and the first RTH hours, -$105 vs an SPY-hold that is up since the arm's history began. All four strategy books are now live forward: xsmom weekly (2 rebalance decisions), eqmom monthly (1), ml7 daily-tranche (3/7 slices on). BTC round-tripped violently today (~$10,014 -> ~$10,324 within hours), whipsawing the crypto arms' vs-benchmark gaps (base -$331, gated -$326, ml7 -$200) while their books barely moved — exactly the high-vol regime the gate is designed to sit out. Kill-watch: nothing near trigger, all arms under 4 rebalances. No parameter changes.

## 2026-07-06 01:11 UTC — first forward weekly re-rank (xsmom) + gate held cash

xsmom base executed its first forward weekly rebalance on the Monday UTC week roll: 23 trades to a fresh 7d-rank book (long ADA/AVAX/BCH/CRV/ETH/SOL/SUI/XLM, short AAVE/ATOM/AXS/ETC/GALA/ICP/INJ/TRX), equity $10,084 (+0.84% abs). The gated arm evaluated its gate at the same roll and stayed in cash (BTC 30d vol still >= its 1y median). BTC-hold rallied to $10,322 (+3.2%), so vs-benchmark both trail: base -$238, gated -$310 — two rebalance decisions in, kill-watch (>=4) not yet armed. ml7's third tranche fired after the 07-05 close (14 trades, 28 net positions, 3/7 tranches live, $10,006, -$180 vs BTC). eqmom's first book due at today's US open. All six runnables healthy.

## 2026-07-05 01:11 UTC — ml7 daily tranche rotation confirmed live

ml7's second tranche fired after the 2026-07-04 UTC close: 14 trade actions, book now 22 net positions across 2 of 7 tranches, equity $9,999.82 (-$70 vs BTC-hold). This is the first automated tranche rotation — the daily cadence machinery works forward; subsequent routine tranches will not be logged. All runnables healthy (ticks 13m fresh, intel 1m). xsmom base $10,068 (-$130 vs BTC-hold), gated still flat in cash (-$198 vs hold). news_main additionally running (user-started). No parameter changes.

## 2026-07-04 17:11 UTC — forward book checkpoint (first >1% move)

All five runnables healthy, ticks <1h fresh, intel scanning. xsmom_base swung +1.1% since the prior 4h check to $10,088.90 (+0.89% abs, 16 trades) — its long alts caught the same bounce that took BTC-hold to $10,182, so it is still -$93 vs benchmark (improved from -$159). Gated arm remains flat in cash by vol-gate design (-$180 vs a rallying BTC — the cost of the gate this week). ml7 $9,998.69 (-$52 vs BTC) on its first tranche; next tranche fires after the 2026-07-04 UTC close (~7h). eqmom untraded until Monday's US open (first book). Kill-watch clear — no arm near -10% after >=4 rebalances (books have 1-2). No parameter changes.

## 2026-07-04 — ml7 forward arm (audited h7 ML strategy) shipped + LIVE

The audit-confirmed h7 ML book (headline above) built into the consolidated Trading extension as a live forward paper arm, following the xsmom house pattern end-to-end.

- **Feature parity is the whole game, so it is a module + a test.** `server/ml7_features.py` extracts the smoke's 28-feature panel builder verbatim (UTC-day resample with gap masking, vol-normalized returns, funding daily sums -> 7d means, cross-sectional ranks, BTC-residual z, dow one-hots, y_sign_7); `tests/test_ml7_features.py` asserts byte-equal output (`np.array_equal`) against the smoke's own code on a shared fixture. Two real parity traps found and closed in flight: OKX `bar=1D` is UTC+8-aligned (ml7 uses `1Dutc`; plain 1D would shift every bridged bar 8h) and pandas 3's default CSV float parser is 1 LSB off (`float_precision="round_trip"` in the cache loaders).
- **Frozen model.** `ml7_trader.py --train` (CLI only, never at boot): full crypto_of history resampled to daily + June-2026 bridge from OKX closed 1Dutc bars + OKX funding -> 86,076-row panel -> 3 HGB classifier seeds {0,1,2} on 85,778 labeled rows through 2026-06-26; fit 86.6s, total 163.4s; `ml7_model.joblib` + manifest (train_end / features / seeds / sklearn 1.9.0). Live scoring = mean predict_proba. Retrain quarterly via `--train`.
- **Live loop.** Hourly ticks; on a new closed UTC day: extend the 400-day rolling caches from OKX (closed bars only, funding summed per UTC day), rebuild features, score, rebalance ONLY tranche `epoch_day % 7` to top/bottom quintile at +-(1/7)/2/k of equity, 3.5bp/side on traded delta; the other six tranches ride (the audited turnover model). Funding neither credited nor charged (audit: ~1%/yr, immaterial; funding_off arm passed alone). Ledger/routes/auto-resume identical to xsmom (`account_ml7_main.json`, `/ext/trading/paper/ml7/*`, system strip + stop_all). Scoreable universe currently 37/40 (FTM/MKR/RUNE are OKX-dead) so k adapts to 7 a side, matching the smoke's per-date behavior.
- **Verification.** 141 extension tests green (124 prior + 17 new: parity x3, mixed-epoch resample, tranche targets/gross, single-slot accounting, fee-on-delta, daily stamp, model-missing guard, resume stamp, cache round-trip); py_compile sweep + JXA parse clean. `--once` placed the first real tranche (day 2026-07-03, slot 1: long XRP/GRT/TRX/FIL/THETA/BTC/AVAX, short AAVE/ICP/ETC/SOL/NEAR/CRV/AXS; gross $1,428.53 vs $1,428.57 target; fee $0.49; equity $9,999.51). Dashboard restarted via `/lifecycle/restart` with the live e3recur training run preserved (PID 85981 verified alive after); ml7 routes 200, model "ready (trained through 2026-06-26)", arm STARTED via the route (auto stamp set), first managed tick landed (equity $10,000.84, no double-trade on the stamped day), and all four prior runnables auto-resumed without a gap (xsmom 43->44, 43->44, eqmom 11->12, intel watch up).
- **Docs.** New `developer_documentation/architecture/backend/trading_ml7.md`; `trading_extension.md`, `trading_page.md`, `paper_trade_xsmom.md` updated; strategy verdict + build recorded as `trading_model_plan.md` section 11. UI: ml7 strategy card (validation panel with the honest lag1 numbers ~13%/yr Sharpe ~1.0, seed-variance + upper-bound caveats, forward record starts 2026-07-04), Overview portfolio cards + system strip + trust-table row.

## 2026-07-04 — trading extension consolidation (market + paper_trade + market_intel -> trading)

Per user: one self-contained Trading extension with a clear, plain-language UI; the old three deleted in the same change.

- **Server.** `extensions/canonical/trading/server/` (24 modules, 8.6k lines) absorbs all three servers unchanged in behavior; one `register.py` mounts `/ext/trading/market/*` (9 model-serving/backtest routes), `/ext/trading/paper/*` (20 trader routes), `/ext/trading/intel/*` (7 radar routes), `GET|POST /ext/trading/settings`, `GET /ext/trading/system` + `POST .../system/stop|stop_all`, and the `/ext/trading/models` analytics page. `news_trader.DECIDE_URL` and every path constant updated; no legacy prefixes kept.
- **Shared channel registry.** `scraper.py` now owns `installed/trading/data/settings.json`: model default, scan intervals, news-fee assumption, and channels `{id, type rss|reddit|gnews|index, value, enabled}` (defaults = the old RSS set + fear-greed). reddit reads `old.reddit.com/r/<sub>/hot.json` with 403/429 -> `blocked` health; per-channel health (ok/blocked/error + last fetch) surfaces in Settings and as chips on the intel tab. News sentiment AND intel briefs both read through this one scraper (intel's own gnews fetch deleted).
- **Control policy.** Nothing auto-starts at boot except `resume()` of user-started runs (ledger `auto` / state `watch.on` stamps, the existing pattern). Recorder + autotrader stay CLI-only. The Overview system strip is the one place to see and stop everything.
- **Page.** One 1602-line `/ext/trading` page (t- design language, all inline, JXA parse-clean): Overview (what-is-this, system status, active-strategy cards with sparklines + vs-buy-and-hold, latest pump alerts, "Can I trust these numbers?" table), Strategies (news / xsmom / eqmom cards, each with an honest "How it was validated" panel: xsmom positive all OOS years Sharpe 0.44/0.93-test-adjudicated, eqmom beat SPY 7/10 + 5.6pp/yr survivorship-matched upper bound, news = no documented edge, live experiment; experiment arms + insight charts folded into an ADVANCED sub-section of the news card), Market Intel (trending board / pump radar with the exit-liquidity warning / event log / channel health), Research (chart-model backtest relabeled "test the chart model on past data" + link to `/ext/trading/models`, the ported 896-line old market page), Settings (defaults + channel manager).
- **Migration evidence.** Arms stopped via old API, auto-stamps rewritten, ledgers md5-identical before/after the `mv` (base 654ac8c5..., gated 7c5f148f..., eqmom 543267fd...; events.jsonl 92 lines 8201d961...); dashboard restarted via `/lifecycle/restart` (detached relaunch; live e2patched trainer PID 72435 verified alive after); all four runnables auto-resumed and ticked within a minute (histories 42->43, 42->43, 10->11); recorder killed + relaunched from the new module path, 10/10 pairs ticking to `installed/trading/data/market/live/`. Old routes 404, new routes 200, catalog = one `trading` entry. Platform dataset cache moved to `installed/trading/data/extension_data/` (the `extensions/data.py` contract location).
- **Tests/docs.** 124 extension tests green (99 migrated + 8 register/system/settings-route + 13 scraper channel/reddit/persistence + 4 pre-existing registry/data); full py_compile sweep clean. New `developer_documentation/architecture/backend/trading_extension.md` + `architecture/frontend/trading_page.md`; `market_*`/`paper_trade_*`/`market_intel` docs updated in place for the new paths/prefixes; `market_page.md`/`paper_trade_page.md` replaced by `trading_models_page.md`/`trading_page.md`; `documentation/api/rest_api.md` + `documentation/extensions/{manifest,marketplace}.md` updated.
- **Bug found + fixed in flight:** `/market/veritate_data_report` had been 500ing since the `EXTERNAL_DIR`->`EXTENSION_DIR` rename (`veritate.py` referenced the dead name); now routes through `data.source_dir()` and returns the inventory (verified live: 40 files, 12.05 GB for crypto_of).

## 2026-07-03/04 — architecture research campaign (efficiency levers E1-E4)

Goal: a byte-level model that is foundationally cheaper to train and run, holds conversations, and writes knowledge into memory at use time instead of by corpus re-feeding. Method: ranked levers from a 2-agent literature sweep, one falsifiable A/B per lever, everything through the dashboard, every outcome to the research ledgers.

- E1 Muon vs AdamW (canonical 10M, fineweb_edu, 12k steps, single delta): **Muon wins 1.60x** (bytes-to-AdamW-final-quality), ahead 115/120 evals, ~3% step overhead. Warmup lag until ~step 400: judge nothing early. Adopted as default optimizer. Runs: `e1adamw_10m_qat`, `e1muon_10m_qat`.
- E2 patched trunk (SpaceByte-style, global blocks on seq/4 boundary slots, muon both arms): 15.0M params at ~dense per-byte FLOPs, +28% realized tok/s (90k vs 70k). First launch died at step 100 to a transient MPS AcceleratorError (async garbage-index): hardened the forward (F.embedding lookup for the boundary table, int32 sort, out-of-place masks; preflight 24c) and relaunched clean. Verdict vs `e1muon` at equal steps + equal wall-clock pending (~step 8300/12000 at time of writing).
- E3 constant-state trunk (gated linear recurrence, scalar-per-head decay, chunkwise): built + verified (chunk algebra exact to 4e-11 vs naive scan, causality bitwise, dump suite green at real shape). Queued behind E2.
- E4 surprise-gated memory trunk (Titans MAG class): built + verified (closed-form inner write gradient exact vs autograd: rel 1.8e-07 fp32 / 7e-16 fp64; chunk-causal bitwise; cross-window persistence plumbing real). Knowledge-injection harness ready (`SMOKE_RESULTS/e4_knowledge_injection_smoke.py`), untrained-model null verified (win rate 0.50, 0 recall). Queued.
- Hooks incident: the E2 run was silently missing 7/14 dump artifacts + neuron_memory.json. Root cause: fixed slot count vs 12-45-byte dump prompts (T < slots). Fixed (`min(slots, T)`), verified all dumps green on real checkpoints for all three variant trunks; `save.py` now prints `DUMP FAILED:` to the run log; preflight 24d added (real-shape dump smoke is gating for variants). e2patched hook backfill for already-written steps pending run completion.
- Kills this campaign (ledgered): backprop-free learning rules (no LM parity at any scale), ternary as a training lever below 3B (stays as export path), MoE-on-MPS (unproven kernels), dynamic shapes on MPS (23x, measured).

## 2026-07-03 — strategy-modular dashboard + plugin cleanup + 3 new research probes

Per user: clean/intuitive paper-trading dashboard, strategies modular + selectable from a dropdown, plugins cleaned up, keep experimenting.

- **Strategy selector shipped.** The Paper Trading page now opens with one Strategy dropdown — News sentiment (LLM) / Weekly momentum (long-short ranks) / Chart model backtest — each strategy's panels show alone; selection persists; running strategies keep trading in the background. Each strategy is its own server module (news_trader.py / xsmom_trader.py / policy+veritate via /market API).
- **Momentum panel shipped.** New xsmom section: arms board (state incl invested/flat vol-gate chip, return, vs-holding-BTC, trades, equity), overlay equity chart vs the $10k line, per-arm tabs with the signed long/short book + recent trades, Start/Stop wired to /ext/paper_trade/xsmom/*.
- **Dead code deleted.** Legacy 2-arm A/B (/ab/* routes, _ab_view, AB constants, page call) removed (~100 lines; superseded by /exp; frozen ledgers left on disk). Stale "mkt_crypto_80m is mis-served" tooltip corrected (stride bug was fixed 2026-07-02).
- **Market plugin defaults fixed.** autotrader CLI default mode vol_harvest -> directional (the old default emitted straddles the spot broker maps to FLAT, so a default launch never traded — plumbing bug found in the 2026-07-02 survey); fee defaults unified at 20bp round-trip across policy.DEFAULTS, autotrader, backtest UI (was an inconsistent 5bp "aggressive maker" nobody gets on market orders).
- **Dashboard restarted** (no training active, all arms stopped) to mount /xsmom/* + drop /ab/*; xsmom arms migrated from the standalone process to dashboard-managed threads with full ledger continuity (32 ticks preserved; base 16-position book intact, gated correctly flat). Verified live: routes 200, /ab 404, page serves the new markup, 63 tests green (26 paper_trade incl 10 new xsmom + 37 market), JS parse-checked.
- **3 research agents dispatched** (running): (1) daily-to-weekly ML feature-panel sweep — does ML beat plain 1w momentum after costs at daily+ horizons; (2) daily pairs/stat-arb with a real short leg at 2.6-3.5bp — cointegration + beta-hedged dispersion, never tested with shorts; (3) canonical 12-1 equity momentum vs SPY at 2026 retail costs — the most-documented factor in finance, untested here. Verdicts get logged when they land.
- **VERDICT pump-fade (short the detected pump): CLOSED — the reversion edge does not exist where shorting exists.** 22,762 events, 160 shortable perps names, 2023-2026: on liquid names (median $71M/24h) detected pumps CONTINUE (+0.15%/24h, +0.47%/72h pooled, P<=0.006) — the 2026-06-26 negative drift was a property of dead-book UNSHORTABLE mid-caps. Liquidity and reversion are mutually exclusive populations (0.3% of shortable events under $1M/24h). Strategy net: -73%/-30%/+19%/-46% by year; the one positive year (2025) = random-short alt-bear beta, not signal; 2026 detection timing is ANTI-selective. Shorts PAID funding during pumps (-0.10 to -0.12%/24h trade, 2-3x the fee load); squeeze MAE p99 23.8%, max +196%. trading_model_plan.md section 14. Retry: forward events.jsonl only. The radar stays a radar.
- **h7 daily-ML audit, static pass CLEAN (hardening reruns in flight):** features trailing, train-only scaling, clean boundaries, fees on traded delta, funding ALREADY charged. Caveats: same-close entry (lag arm running), 2026-majors survivorship (ambiguous sign for L/S). Verdict pending on purge/seeds/quantile/entry-lag arms.
- **Market Intel extension SHIPPED + LIVE (user request, agent-built end to end).** New extensions/canonical/market_intel/: OKX all-market scanner (302 USDT pairs, $50k floor) + CoinGecko trending/meme (cached, rate-limit safe), per-coin trailing-baseline z-scores from rolling snapshots, pump flag = price_z>=2.5 AND vol_z>=3.0 (the validated probe recipe) w/ 6h cooldown -> events.jsonl forward dataset; local-LLM briefs (why moving / news-driven / meme / pump-risk) via /teacher/complete w/ no-news+spike auto-high-risk; page = Trending board / Pump radar / Event log with a permanent exit-liquidity warning (research law: detection is a radar, NOT a buy signal). 30 new tests (99 total green), component doc, catalog entry. Mounted + watch thread live (5-min scans, auto-resume at boot); FIRST SCAN caught real dual-spike events (KAT price_z 5.5/vol_z 6.4, MEME 3.2/3.2), model briefs ok. Watchdog cron covers it.
- **Auto-resume validated for real:** the market_intel mounting restart brought all 3 paper arms back up with zero manual action.
- **VERDICT (3) equity momentum: PASSED-WITH-ASTERISK -> built + launched forward.** 12-1 top-decile monthly on ~496 current S&P names: beats SPY 7/10 years, Sharpe 0.875 vs 0.540 — but the agent's random-decile null measures survivorship at ~+4.9pp/yr; the defensible claim is +5.6pp/yr OVER that null (above all 20 null seeds), an upper bound. Costs irrelevant (turnover 3.8x/yr ~ 23bp/yr). Only clean validation = forward. DONE: universe FROZEN 2026-07-03 (equity_universe.json, 496 names), built eqmom_trader.py (monthly 12-1 top-decile long-only, 3bp/side, Yahoo marks, SPY bench), /ext/paper_trade/eqmom/* routes, strategy-dropdown entry + shared book panel, 6 new tests (32 green). LIVE — correctly refused to trade today (July-4th market holiday); first book buys Monday's open. trading_model_plan.md section 13.
- **Auto-resume shipped.** xsmom + eqmom arms stamp `auto` in their ledgers; register() resumes flagged arms at boot — dashboard restarts no longer gap the forward records. Verified: restart + all 3 arms running (xsmom base/gated, eqmom main).
- **VERDICT (2) stat-arb: CLOSED — dies on signal, not cost.** With shorts legal and turnover 1-14x/yr, fees are finally irrelevant (Sharpe barely moves 2.6->5bp) — and it still fails: full-period Sharpe 0.163, inside shuffled-entry noise every year (2023 WORSE than random timing). Root cause measured: in-sample cointegration does not persist OOS (selection persistence 39%->17% YoY; qualifying pairs 31->3 by 2026; 70-80% of exits are timeouts, not convergence). Beta-hedged dispersion variant: Sharpe -0.7 to -0.9, DOA. trading_model_plan.md section 12. No retry at this universe/frequency.

## 2026-07-02 — trading campaign: fee-floor attack, signal compositions, one forward survivor

Mandate: "figure out how to make money daily; review the algo, iterate and refine until it works." Ran 8 falsifier-driven experiments (6 new smokes + 2 web-research agents), all CPU, all logged with stats JSONs in SMOKE_RESULTS/ and doc sections in developer_documentation/market/.

### What changed the map
- **Fee floor (web agent):** US-legal perps now 1.9-3bp/side all-in (Kraken US perps via Bitnomial launched 2026-06-15: BTC 2.6/ETH 1.9bp flat; Coinbase nano perps ~3bp; CME MBT via Robinhood 2.6-2.9bp). Binance.US spot repriced to 0 maker / 2bp taker (ghost-town book). Every prior backtest assumed 10-20bp. Shorting majors is now cheap and legal. Realistic all-in with spread/impact: 3.5-5bp taker, and passive saves only ~0.5bp net of adverse selection on momentum-style signals.
- **Dataset landmine:** crypto_of CSV `time` switches ms->us at 2025-01-01 (Binance format change). `market/data.py::normalize_time` handles it; raw-CSV consumers must normalize or corrupt all 2025+ bars. All new smokes handle it.
- **Stride regression fixed:** `mkt_crypto_80m` had no stamped `bar_stride` and the removed LEGACY_STRIDE fallback would have served it at stride 9 (trained 5). Stamped `bar_stride: 5` into its config.json.
- **OKX live recorder restarted** (was dead since 2026-06-15 after 4.7h): book_imbalance/OI/funding accumulating forward again for a future L2-signal test.

### Experiments (verdict, artifact)
1. **OFI x magnitude composition** (ofi_magnitude_composition_smoke.py): the two real signals composed. Gating direction trades on predicted-move size multiplies gross/trade 3-7x exactly as hypothesized, BUT gated trades become isolated round trips (turnover/trade 1.3-1.6 sides), so break-even ~1-1.5bp/side. FAILS at 2bp+. 1h OFI edge is DECAYING (AUC 0.538 2022 -> 0.511 2026); 15m is not (2026 its best year). Nulls clean.
2. **Turnover-clustered construction** (ofi_portfolio_construction_smoke.py): deadband/hysteresis/persistence variants. Persistence (2-bar confirmation) = best cost-surviving direction result ever here: net-positive 4/5 years at 3.5bp — but 2026 negative and h2-2025 negative, consistent with the AUC decay. NOT deployable; research-closed.
3. **Cost-killed families re-test** (costkilled_retest_smoke.py): funding is a +1.85%/yr TAILWIND to XS momentum L/S (old 10%/yr drag assumption wrong). 1d momentum +16.3%/yr at 2.6bp but 2025 negative; **1w momentum positive ALL years at ALL fees** (Sharpe 0.44). 1h XS reversal: +120.7%/yr Sharpe 1.36 at ZERO fee OOS — strongest raw signal of the entire campaign — break-even 1.16bp/side, dies at real fees. Seasonality + breakout: signal-dead.
4. **Reversal maker execution** (reversal_maker_exec_smoke.py): the falsifier that mattered. The ~1.2% of reversal signals a resting limit CANNOT fill carry the whole +121%/yr; the fillable subset is -2.1%/yr GROSS. The wall is adverse selection, not fees; passive execution cannot rescue it. New durable invariant: only signals with positive FILLED-SET gross are worth maker-execution retests. Also: the reversal signal itself decayed to sub-fee in 2026, and capacity is ~$13k. Family closed.
5. **Options/vol monetization** (vol_options_scoping_smoke.py): magnitude signal (the one always-real axis, Spearman 0.35-0.43 every year) is NOT monetizable via options: IV already prices it (reactive anchor kills every cell), BTC VRP is fat (IV/RV ~1.5 now), IBIT straddle round-trip 3-11% of premium, and the signal lives at 15m-1h while the shortest expiry is 1-2d. Short-vol variants = generic VRP harvest with -8 to -17% single-day tails. Closed.
6. **Carry re-validation** (carry_revalidation_smoke.py): 2026 funding regime compressed; $10k top-3 rotation nets $1.51/day maker vs $1.23/day T-bills. Below venue risk. Retry: sustained majors funding >15%/yr.
7. **marketof A/B finally scored** (marketof_ab_score_smoke.py): the trained-but-never-evaluated order-flow byte-model A/B. Real taker-flow channels add ~nothing: direction delta -0.3pp mean (7/18 cells better), return-byte CE flat, magnitude +0.002 (under noise). The byte model matches classical GBM on the same features — MODEL ARCHITECTURE IS NOT THE BOTTLENECK; signal content of public data is. (Its val-loss "advantage" was an artifact of constant channels in the noflow corpus.)
8. **1w momentum regime filter** (xsmom_regime_filter_smoke.py): 9 pre-registered candidates, train-only selection. btc_vol_low (calm regime) clears the full bar on test — Sharpe 0.93, all years positive, maxDD 10.1%, fee-robust — but train couldn't distinguish it from its own complement and placebo gates pass 21.5% of the time (observed = 92nd pct of placebos). Verdict: promising, test-window spent, FORWARD VALIDATION ONLY.

### Built + launched
- **`xsmom_trader.py`** (paper_trade extension): weekly XS momentum L/S forward paper trader, arms `base` + `gated` (ledgers account_xsmom_{base,gated}.json, $10k each), 40-major universe, long top-8 / short bottom-8, 3.5bp/side, hourly marks to OKX, closed-candle signals only, no funding credit (conservative). 10 new tests (26 total green). Routes `/ext/paper_trade/xsmom/*` land on next dashboard restart (did NOT restart the live server); standalone process running now. First live ticks verified: gated arm correctly FLAT (BTC 30d vol > 365d median today), base arm opened 8L/8S at $3.50 fees. Component doc: developer_documentation/architecture/backend/paper_trade_xsmom.md.
- Judge the arms by: equity vs $10k AND vs BTC-hold, plus base-vs-gated divergence. Weekly cadence means evidence accrues in months, not days — do not overread the first weeks.

### Honest bottom line
Direction on public data remains efficient at retail cost even after the fee collapse — every rescue (composition, construction, maker execution, options) failed its falsifier for a structural reason now understood and documented. The one honest candidate left standing is slow (weekly), modest (backtest ~13%/yr, Sharpe ~0.9 gated / ~7.5%/yr Sharpe 0.44 unfiltered), and unproven forward — which is exactly what the paper run exists to settle. "Money daily" from short-horizon prediction stays falsified at this capital/cost tier; the daily-visible number is the forward equity curve now accruing.

Goal handed over: train a 2.5B model optimized for coding, distilling a corpus from a language teacher so it follows instructions then writes only Python. "No pushback, figure it out, experiment, track every step."

### What is running
- `pycoder_3b` LIVE via dashboard (`veritate_3b` trainer = exactly 2.52B params, dense byte-level, L32 h2560 ffn10240 vocab256). Stage A = language-first pretrain on FineWeb-Edu (5.34 GB, on disk). WSD schedule (flat LR, usable if stopped anytime), total_steps 400k.
- Throughput retune (live-confirmed): batch 2 -> 16 took tok/s **210 -> 432** (step 1; steady-state higher). act-ckpt stays ON because the dashboard `_build_argv` can't emit a false boolean (`--no-flag`), and flipping the trainer-manifest default is an upstream-mirrored change wrong for low-mem boxes. Batch is the dominant lever anyway. Memory ~50 GiB of 217 GiB budget — huge headroom.
- Teacher-distilled Python corpus generation running in background: qwen2.5-coder:32b, Python-only system prompt, chat template, ~64 B/s (3-4 workers, contended). `generate_chat_corpus.py` extended with `--categories` + `--system` (backward-compatible). Output `trainers/corpus/py_distill_v1`.
- Loss sanity: byte0 6.05 -> 3.38 by step 25 (batch 2 run before retune). Pipeline proven end-to-end; teacher output is clean runnable Python.

### Three research agents (findings; sources in each agent transcript)
1. **Throughput (M3 dense 2.5B):** act-ckpt HURTS a dense model (+33% FLOPs for memory we don't need); bitsandbytes not importable on MPS (fp32 AdamW, fine at 180 GiB free); torch.compile/channels-last/fused-adam give ~0 on MPS. Best OOM-safe config = act-ckpt OFF + batch 8 (~2-2.5x); via dashboard we got batch 16 + ckpt-on = ~2x. **Dominant problem is horizon, not tok/s.**
2. **Corpus (Phi recipe, byte-efficient):** DON'T teacher-distill the pretraining volume (~25 GB byte-equiv of Phi-1's CodeTextbook = ~7 years at 10 MB/day). Pretrain on FREE streamed real code (codeparrot/codeparrot-clean ~50 GB Python; the-stack-dedup python; starcoderdata) + FineWeb prose, mix ~60% code / 30% prose / 10% Pile, ChatML frame from step 0. Spend the ENTIRE teacher budget on a small (~0.5-1 GB) instruction->Python SFT set (docstring->function), hard-filtered (ast.parse + sandbox-run), ROUGE-L<0.7 dedup, multi-sample, function-name/topic templating, multi-teacher routing (32B code / 72B QA / 14B breadth), 3-8 epochs with 5-10% replay. Skip logit/KL distillation (byte-vs-BPE vocab mismatch kills it). Anchor: phi-1-small 350M = 45% HumanEval.
3. **Feasibility (the decisive one):** **From-scratch dense 2.5B coder on this box is INFEASIBLE.** Two independent walls: (a) compute — only measured M3 training point is ~21.3k tok/s on 88M MTP; params-scaled dense 2.5B ~ 750-3000 tok/s -> Chinchilla-optimal 50B tokens = ~250 days to 2+ yr; (b) corpus — local teacher tops out ~0.08-0.75 B byte-tokens/MONTH, 2-3 orders of magnitude below any useful coder (Qwen2.5-Coder 0.5B/1.5B/3B each saw 5.5T tokens). **Reachable instead:** byteify an open-weight Qwen2.5-Coder-1.5B/3B (Bolmo method, ~10% of pretraining compute, preserves code ability) — the ONE lever that fixes both walls; OR a 0.4-0.8B narrow-Python model (the 800M trainer exists, MTP-equipped). Other levers: Muon optimizer (~2x), MTP at >=2.5B (small code win at crossover scale), curriculum + short-context packing. Cheapest <24h de-risk: continued-pretrain the 800M trainer on the distilled corpus, measure HumanEval; if it can't clear ~15-20% pass@1 in a day, the from-scratch 2.5B is dead on arrival.

### Honest verdict (preflight 9a)
The literal ask — from-scratch dense 2.5B coder distilled from a local teacher — cannot reach a useful Python model on one M3 Ultra (compute months-to-years; teacher corpus 2-3 orders too small). The from-scratch `pycoder_3b` run continues as the requested baseline + pipeline proof + real loss curve, and corpus distillation continues (the explicit ask). The path that actually yields a ~2.5-3B byte-level Python coder on this box is **byteify Qwen2.5-Coder-3B (Apache-2.0) + distilled SFT** — recommended as the real bet. Awaiting operator steer on whether to pivot the single trainer slot to it (or to the cheap 800M de-risk) vs. keep the from-scratch baseline running.

### checkpoint — step 500 (~4.1h)
- `pycoder_3b`: byte0 loss 6.05 -> 2.57, grad_norm 38 -> 1.9, **tok/s 556 steady** (batch16), still in warmup (2000). First ckpt at step 2500 (~20h). Trajectory healthy.
- Corpus quality (live confirm): pass 1 = 324 prompts -> 317 valid / **3 invalid (0.9% filtered)** / 4 timeout. The Python-only system prompt + qwen2.5-coder yields almost-all-valid Python; the `ast.parse` filter is cheap insurance. Pass size ~248 KB.
- Corpus throughput wall (live): **27 B/s under training contention** (was ~64-76 B/s solo), ~2.6h/pass. A 0.5 GB SFT set = weeks-to-months even at best stream rates. Confirms feasibility verdict: teacher-corpus scale is the binding constraint; spend it only on the small SFT layer, never pretraining volume.

### checkpoint — step 2500 (~20.6h), first ckpt + first eval + first sample
- `step_2500.pt` saved (30 GB incl. optimizer). **val_loss 2.17** nats/byte on held-out FineWeb (random 5.55). No overfit (<1% of one epoch). byte0 train (single-batch) 1.28.
- CPU greedy sample (ROE rule 2, off the MPS): `"The capital of France is"` -> `"...is a strong to the strong to the strong"`; `"Water is made of"` -> `"...made of the strengthened the strengthened"`. Honest read: real English words + spacing + local grammar from a 256-way byte vocab, but greedy repetition loops and zero semantics/long-range coherence at 41M tokens. Pipeline proven end-to-end (train -> ckpt -> CPU inference -> valid English bytes). No code yet (none in FineWeb). Sampler at /tmp/sample_pycoder.py (throwaway).

### model_type classification fix (step ~2950)
- Bug caught: launching via the raw `/trainers/run` API did NOT set `model_type` (the dashboard form does; trainers drop `--model_type` via parse_known_args; `trainer_runner` only exports `VERITATE_MODEL_TYPE` when the arg is present). So `pycoder_3b` defaulted to `language` -> wrong hook gating + wrong eval-deep gate for a coder.
- Reviewed save/hooks path: `model_type` in {language(default), code, statistical, other}. Non-language SKIPS `LANGUAGE_DUMPS` (grades/reading_comprehension/math/grammar/reasoning/concepts/writing_health/generation); arch probes (probe/lens, classroom, surprise, quant_kl) always run; no code-specific dump exists. eval-deep route refuses language benchmarks for non-language types.
- Fixed zero-loss: set `training_args.model_type="code"` in `models/pycoder_3b/config.json` (`_ensure_config` only writes when missing, so the edit sticks; save path reads it fresh). VERIFIED at step_5000 hook dir: all 8 language dumps skipped, arch probes (classroom/lens/probe/quant_kl/surprise) present. step_2500 (pre-fix, "language") had the full language suite.
- Preflight updated: rule 24a (launch via /trainers/run MUST set model_type, + the silent-default trap + live-fix), rule 24b (act-ckpt-vs-batch on high-mem box, `_build_argv` can't send false bool, no bitsandbytes on MPS).

### checkpoint — step 5000 (~41.5h)
- byte0 train ~1.12, **val_loss 1.1118** (down from 2.17 at step 2500). Real progress on the English base. tok/s ~512-556. WSD stable phase (flat LR 1.5e-4).

### checkpoint — step 10000 (~83h, 164M tok, 2.5% of run): from-scratch plateau, on the record
- val trend: 2500=2.17 -> 5000=1.11 -> 7500=1.05 -> 10000=1.02. **Flattening to a ~1.0 plateau** (deltas 1.06 -> 0.06 -> 0.03). Flat LR WSD stable phase.
- Greedy samples FLAT in quality 5000->10000 despite val drop: `"The capital of France is the most important to the construction of the c"`, `"Water is made of the standard standard standard"`. Locally grammatical, greedy-loops, zero semantics/facts, no code.
- VERDICT CONFIRMED WITH DATA: a from-scratch dense 2.5B byte-level model on this box converges into a mediocre English byte-LM (~val 1.0) that produces grammatical-but-meaningless greedy text, and plateaus pre-coherence. 3.5 days of compute -> not a coherent text model, nowhere near a coder. Empirical proof of the feasibility agent's compute+corpus walls. Path to an actual ~2.5-3B coder remains warm-start-from-Qwen2.5-Coder (byteify) + the small distilled SFT; filtered Python corpus at 8.2 MB, growing ~2-3 MB/day, enough for the cheap 800M-de-risk SFT when the trainer slot frees.

---

## June 2026: trading and sentiment research (condensed 2026-07-08)

The detailed day-by-day chronology of the June trading and news-sentiment campaigns
(the Robinhood-strategy hunt of 2026-06-15, the LLM-sentiment edge of 2026-06-16, and
the 48h forward validation of 2026-06-17) has been retired from this log to keep it
focused. Those are a separate research line and their outcomes are recorded where they
belong:

- One-line summaries survive in the **timeline** block at the top of this file.
- The validated and falsified verdicts are in `successes.md` / `failures.md`.
- The living market research and trading-model state are under
  `developer_documentation/market/` and the trading extension docs
  (`developer_documentation/architecture/backend/`).

Headline of that line of work: no long-only, fee-paying, public-signal crypto strategy
showed a net-of-cost edge; the one measured signal (LLM-scored news sentiment predicting
next-day BTC, and later 1-week cross-sectional momentum) moved to forward paper validation.

## 2026-07-14 — corpus library expansion (technical)

**Built (all deterministic, seeds fixed, sha256 recorded in corpus_catalog.json):**

| stem | train | val | zip (staged) |
|---|---|---|---|
| chat_500mb | 513,802,730 | 10,486,351 | chat/chat_500mb.zip 178 MB |
| chat_5gb | 5,138,023,139 | 104,858,273 | chat/chat_5gb.zip 1.74 GB |
| agent_150mb | 154,141,038 | 3,146,243 | agent/agent_150mb.zip 17 MB |
| agent_1500mb | 1,541,407,297 | 31,457,972 | agent/agent_1500mb.zip 169 MB |
| mcp_15mb | 15,415,068 | 315,681 | mcp/mcp_15mb.zip 2 MB |
| mcp_150mb | 154,141,336 | 3,146,487 | mcp/mcp_150mb.zip 18 MB |
| mcp_1500mb | 1,541,407,629 | 31,457,682 | mcp/mcp_1500mb.zip 184 MB |

Staging root: `~/Library/Mobile Documents/com~apple~CloudDocs/Mirach-Corpuses/`
(+ `trading_datasets/crypto.zip` 6.8 GB, sha256 2dc354aa…; `manifest.md` maps
zip → placeholder URL → release steps). Bins also copied into `trainers/corpus/`.

**Chat rebuild caveat:** the original chat_500mb/chat_5gb catalog hashes were
minted against a PG cache that no longer existed; the cache was restored as 30
Gutenberg texts (`trainers/corpus/_pg_cache/`, fetch list in the staging
manifest) and the catalog now carries the new hashes. `chat_50mb`/`agent_15mb`
stay untouched on GitHub (published, hash-stable, under the 100 MB raw limit).

**New builders** (`veritate_mri/tools/`): `build_agent_corpus.py` (Hermes
frames per framing.md over the runtime toolbox — calculator/fs_read/fetch/
retrieve; both system-turn styles: framing "You may call:" and runtime
`prompt_block()` "Available tools:"; calculator results evaluated from the same
expression, fs/fetch answers derived from generated content, 10% error-recovery
turns, 15% no-tool turns reusing the chat Q/A pools). `build_mcp_corpus.py`
(45% JSON-RPC transcripts with initialize/tools/list/tools/call/resources/
errors/ping over 8 fictional servers and 4 protocol versions; 20% protocol Q&A
from a 28-pair hand-written pool + passages lifted from the native mcp_docs
bin; 35% Hermes conversations over MCP server tools).

**Platform:** corpus_sync `zip_bundle` format (`_extract_zip_bundle`: member
match by basename, any folder prefix, `.part` atomic writes, zip removed in
`finally`; val member optional and composable with `val_split_ratio`);
`coming_soon` in `_entry_skeleton` + install refusal; disk precheck counts
size_val for bundles; index.js `CORPUS_COMING_SOON` set deleted in favor of the
catalog flag; UI expected-size adds val for zip_bundle. Trading side untouched
by design — `extensions/data.py` already extracts zip/tar.gz and deletes the
archive in `finally`, and `url: null` is its native coming-soon.

**Tests:** `tests/mri/test_corpus_library_zip.py` — extractor both-members +
zip-deleted, missing-train-member error, coming-soon install refusal, builder
determinism (agent, mcp), frame validity (tool_call JSON contract, JSON-RPC 2.0
lines). 7/7 green; tests/mri 149 passed / 4 pre-existing test_models_route
failures (reproduce on stashed tree).

**Docs:** `developer_documentation/corpus/library_ladder.md` (ladder + cap
rationale), `architecture/backend/corpus_library.md` (5 formats + release
flow), `architecture/frontend/settings_tab.md` (catalog-driven gate).

---

## 2026-08-12 — wren1_0 chat SFT launch

**Run:** `wren1_0`, resumed from `wren_base` step 146,014 (270,510,336 params,
hybrid trunk, layers=16 -> 20 blocks). 8,000 steps, `loss_mask=assistant`,
base_lr 5e-5 -> 5e-6, wsd decay_frac 0.25, warmup 200, batch 48 x seq 1024 x
n_chunks 4. Mix: mixed_chat 0.300, chrg 0.200, wren_identity 0.180,
hansard 0.120, scotus 0.080, ukinquiry 0.070, veritate_chat 0.050
(1,895,835,636 train bytes, val = mixed_chat_val). Launched via
`POST /trainers/run`. Steady 12,880-12,904 tok/s vs the base run's 12,857
ceiling; 15.2 s/step, ~33.8 h projected.

**Two bugs caught at launch, both would have been silent.**

1. `shape_from_checkpoint` counted `blocks.N.*` entries as the layer count.
   VeritatePatched builds `N_LOCAL_ENC(2) + layers + N_LOCAL_DEC(2)`, so
   wren_base's `layers=16` writes 20 entries. The first launch built a 24-block
   model, `load_resume_state(strict=False)` loaded the 20 real blocks, and
   54,649,152 params stayed at random init with nothing in the log. Caught by
   diffing the printed param count (325,159,488) against the checkpoint
   (270,510,336). Fix: `trunk_block_overhead()` + `PATCHED_TRUNKS`, and
   `load_resume_state(require_complete=True)` now reports every stranded tensor
   and refuses a plain resume that leaves any. wren_base's `shape.layers: 16`
   was correct all along; the earlier todo calling it a bug was wrong, reverted.
2. The unknown-flag fatality shipped earlier rejected 9 live dashboard schema
   fields (`model_type`, `layers`, `hidden`, `ffn`, `heads`, `rope_base`,
   `alpha`, `variant`, `init_from`) — every UI launch would have died. Shape
   fields are now parsed and ASSERTED against the resolved shape rather than
   ignored; the rest joined SCHEMA_IGNORED_FLAGS. A test scrapes TRAINER_SCHEMA
   out of `web/index.js` and fails on any unhandled field.

**Hook cost control.** Measured the full dump suite at ~137s/checkpoint across
three wren_base checkpoints, of which generation 33s, reading_comprehension 31s,
writing_health 24s, reasoning 20s, math 20s (= `save.HEAVY_DUMPS`); everything
else totals ~9s. New `--hooks full|light|off` + `--hooks_full_every N`. This run
uses `light` + every 4th, which bought ckpt_every 250 / eval_every 200 for ~1%
overhead instead of ~4% at ckpt_every 500.

`use_act_ckpt` left ON. mem_planner reports `tier=none fits=True need=206.5GB
budget=217.6GB`, but the budget assumes the whole 256 GB machine (~170 GB was
actually free) and the planner does not model n_chunks. Not worth an OOM on a
34 h run; queued as a measured experiment for the next pretrain (~30% on offer).

**Naming:** `wren1.0` is rejected by `NAME_RE` (`[a-z0-9_]` only). On disk as
`wren1_0`.

**Tests:** 1,115 passed, 6 skipped, 8 xfailed.

## 2026-08-18 — IDEA 19 campaign + ideas backlog (autonomous session)

Plan: (1) guard-distilled corpus from wren1_0@1250 (no_repeat_ngram=16 replies to
mixed_chat user prompts), (2) IDEA 6 closeout from the finished chin200m curve,
(3) IDEA 10 accept-length gate measured offline from this week's reply transcripts,
(4) wren1_2 SFT via /trainers/run per the launch checklist, (5) 30-prompt ladder
vs wren1_0@1250. Falsifier: bare-greedy loop 0.20 -> <0.05 with grounded/identity/
median held. Logging here as each step lands.

**IDEA 6 closed** (successes.md): chin200m/wren_base val 0.7798 -> 0.7073 at 145k,
13% under chat200m's 0.812, form A/B held. Under-training confirmed as the ceiling;
section removed from ideas.md.

**IDEA 10 gate measured** (ideas.md updated): offline simulation over 67 real
transcripts. Grounded traffic 2.62-6.54 accept/draft (~2.2x fewer weight streams),
clears the >=2.5 ship gate at every m; plain chat 0.84-2.99, marginal and
loop-flattered. Engine implementation queued RAG-path-first; not started.

**IDEA 19 phase A running**: 100,118 unique mixed_chat prompts harvested, generating
2,000 guarded replies from wren1_0@1250 (ban n=16). ETA ~2h.

**wren1_2 prepared**: seeded models/wren1_2/checkpoints/step_0.pt from wren1_0@1250
(the ship checkpoint; fork route only forks latest, which is the measured-worst
step_3000 — platform gap noted). Launch args staged: 500 steps, base_lr 2e-05,
wsd decay last 125, ckpt_every 100, mix mixed_chat 0.60 / transcripts 0.20 /
wren_noloop 0.08 / veritate_chat 0.04 / identity 0.04 / idk 0.04 (small-corpus
0.20 rule held). Launch via POST /trainers/run once corpus generation frees the GPU.
Baseline for the falsifier: wren1_0@1250 on the fixed 30-prompt set (loop 0.20,
grounded 0.25, identity 1.00, median 168 B).

**trainers/ deleted** (user-approved, 71 GB freed). Verified first: scan() tolerates
the missing dir, native trainer lives in veritate_mri/, every trainers/corpus file
was a byte-duplicate of data/corpus except four stale ones (June 0-byte rag_ui
husks; a superseded Jul 27 veritate_chat). CLAUDE.md + documentation.md corpus
locations updated. LEGACY_CORPUS_ROOT stays in code for installs that have one.

**noloop generation relaunched** — the first run was stopped by a session restart
before writing bins.

**venv correction**: the raw venv/ is gone (user removed it); the correct environment
is .veritate_venv/ at repo root (torch 2.11.0, MPS). noloop generation relaunched
under it. Any tooling that referenced ./venv/bin/python must use .veritate_venv.

**Server restarted correctly**: veritate.py VENV_DIR now points at .veritate_venv
(it was silently rebuilding the raw venv/ the user deleted). Dashboard back on 8001,
runner idle. **IDEA 11 unblocked**: the "missing" format-adherence set exists as
ifeval_form.json (280 obedience-only items, deterministic checkers) — the blocker
note was stale. Experiment needs two matched 200M runs; compute sizing is the
user's call.

**wren_noloop corpus done**: 1,929 examples / 0.9 MB, 96.5% keep (43 unclosed,
28 residual loops dropped). Well-formed ChatML, coherent replies.
**wren1_2 launched** via POST /trainers/run: 500 steps, base_lr 2e-05 wsd,
mix mixed_chat 0.60 / transcripts 0.20 / wren_noloop 0.08 / small-corpus 0.20.
Monitor armed on val rows + terminal state. Next: 30-prompt ladder + grounded
loop check + ifeval form on the best checkpoint vs the wren1_0@1250 anchor.

**wren1_2 launch #1 failed and was fixed**: apply_resume_overrides requires a
config.json on the resume target, which a bare checkpoint seed lacks. Wrote the
config the way fork_model does (name/step/forked_from set, corpus_bin cleared,
resume true) and relaunched clean. Note for the fork-from-step gap: any manual
seed needs checkpoint + config, not just the .pt.

**wren1_2 run complete**: 500/500 steps, exit 0, 6 checkpoints. Val 0.548 (100)
-> 0.543 (200) -> 0.550 (300) -> 0.549 (400) -> 0.558 (500); mild late rise,
consistent with small-corpus passes. Ladder launched: wren1_2 x6 + wren1_0@1250
anchor, bare greedy, fixed 30-prompt set. Verdict criteria: loop < 0.05 (from
0.20) with grounded 0.25 / identity 1.00 / median ~168 B held.

**IDEA 19 mechanism 1: FALSIFIED.** wren1_2 ladder: best loop 0.13 (@200/300) vs
anchor 0.20, target <0.05 -- under half the gap, and grounded halved / identity
slipped exactly where loop improved. Harness validated by wren1_2@0 scoring
byte-identical to wren1_0@1250. Kill recorded in failures.md; ideas.md updated;
unlikelihood loss is next in the queue. Ship model unchanged: wren1_0@1250 with
the serving-default guard.

## IDEA 7 Track A opened (2026-08-18)

**Cliff measured** (wren1_0@1250, hansard_val, 3 arms): slide flat 0.89-0.95 bpb
(but 1 forward/byte); stream 1.44 bpb first-64-B-after-wrap (+62%), 1.01 late;
stream == stream0 to 4 decimals. Root cause read from model_recurrent.py: GLA
per-slot decay exp(cumsum(-softplus(a))) x256/window -> state 1e-15 by window
end; carry-off training never rewarded retention. conv tail also carries padding
zeros on part-full windows (secondary).

**wren1_3 launched** (runner confirmed running): wren1_0 recipe with ONLY
state_carry=chunks + bptt_window=2 changed, 1,000 steps, lr 2e-05, ckpt/eval 200.
Each step = 4,096 contiguous bytes, state threaded across boundaries, gradients
flow 2 chunks. Falsifier: re-run cliff_measure -- stream[1-64] must close toward
slide AND stream must beat stream0; chat ladder must hold. Escalation if it
fails: state_rule=delta, then pinned (non-decaying slots).

**wren1_3 launch #1 crashed, platform bug found and fixed**: both activation-
checkpoint wrappers (trainer lambda + mem_executor._checkpointed) dropped kwargs,
so state_carry=chunks (blocks called with state=/return_state=) crashed at step 1.
Fixed with kwargs pass-through; 3 new tests in tests/training/
test_act_ckpt_state_carry.py (wrapper composition + grads flow through wrapped
state carry). Ruff installed into .veritate_venv; repo clean. Relaunched with
use_act_ckpt=false anyway: de-risks the run and doubles as todo-10's
act-ckpt-off speed measurement on this shape.

**wren1_3 attempt 2 diagnosed and fixed — two platform defects, one architecture defect:**
1. Non-finite GRADIENTS with finite loss: anomaly-traced to the slot-mask multiply
   (model_patched.py). Under state carry, the next window's loss backprops an
   unbounded gradient into padded slot rows; mask-multiply turns inf*0 into nan.
   Reproduced deterministically with trained weights on MPS AND CPU (not precision,
   not device: fp32 failed identically; tiny random weights do not repro).
2. Fixes: torch.where for the slot mask (exact-zero grads at padding); k/v/la masked
   out of the state stream inside RecurrentMixer (padding writes nothing, decays
   nothing; in-window outputs bit-identical -- golden check 0.00e+00); trainer now
   skips the optimizer step on non-finite grad norm (step 1 had stepped on nan and
   corrupted the weights; guard validated by the relaunch rather than a unit test).
3. Tests: tests/training/test_state_carry_padding.py (4: state no-op, padding
   invariance, streaming==training equivalence, finite grads). Suites 647 passed,
   ruff clean. Post-fix repro: 0 non-finite grads in bf16 and fp32.
wren1_3 relaunched (attempt 3).

**wren1_3 attempt 3 -> the deeper defect, found and fixed.** The grad-norm guard
held (weights protected) but ~10/12 steps skipped. Bisected a 48-row batch to ONE
culprit row; isolated the trigger to bptt_window=2 (carry with bptt=1 clean);
instrumented seam hooks (fired 64x, all finite -- the carry seam was innocent) and
an inf-detector that caught the true origin: padded slot rows re-inflate INSIDE
the window walk (q@state + FF biases, never re-masked between blocks), drift out
of the trained regime, and their FF backward goes infinite by mid-stack (first
inf at blk4), then dies to nan at the next mixer's proj -- poisoning shared weight
grads. Fixes: (1) re-mask the slot stream after EVERY global block in
forward_streaming (valid rows never read padding, so behavior is unchanged --
suites + equivalence tests green); (2) carry_seam_clip: norm-clip + nan-zero
gradient hook at the carry seam (never fired in validation; kept as insurance for
the truncated-BPTT explosion class, wired to args.grad_clip). Validation: culprit
row finite in bf16+fp32; 20x16-row live-condition sweep (1.3M tokens) zero
non-finite; unit tests cannot reach this pathology (needs trained weights), so the
sweep is the pinning artifact. 647 tests pass, ruff clean. wren1_3 attempt 4
launched.

## IDEA 7 arm 1: VALIDATED (2026-08-19)

wren1_3 completed 1,000/1,000, zero non-finite steps. Falsifier: carried state
absmax 1e-15 -> 1.27; stream beats stream0 in every bucket; wrap cliff 1.44 ->
1.30 (~26% of the gap to slide 0.89); mid-window stream 0.99 ~ slide 0.91. Chat
ladder: loop 0.20 / closed 1.00 exactly match the ship anchor; identity 0.83 and
grounded 0.12 each one question inside n-noise; median 148 B vs 168. Val carry
shock 0.72@200 -> 0.55@400 (adapted). Graduated to successes.md; ideas.md carries
the escalation (longer adaptation, delta, pinned; then the streaming generation
loop). wren1_0@1250 stays the chat ship model; wren1_3 is the streaming-lane base.

## Persistent-memory pivot + arm 1b launch (2026-08-19)

User directive (permanent, recorded): persistent memory is THE research focus.
"Tell an AI something once and it doesn't forget" — stateless serving with
re-injected context is the frustration to eliminate; the net should update its
own state/weights from what it processes, brain-like. IDEA 20 opens the program.

Actions: launched IDEA 7 arm 1b — resumed wren1_3 1000 -> 3000 (triple carry
dose, recipe otherwise frozen; falsifier: wrap bucket keeps closing from 1.30
toward slide 0.89, chat ladder holds; ~8.5 h ETA, milestone monitor armed).
Fanned out three literature surveys (fast weights / test-time training;
Titans-class neural memory + surprise gating; continual learning + sleep
consolidation at small scale) to ground IDEA 20's experiment ladder before
writing it.

## IDEA 20 opened: persistent memory program (2026-08-19)

Three literature surveys returned and converged: (1) wren's GLA state is already a
fast-weight associative memory — the delta rule makes it in-place-editable (up to
D orthogonal associations, revisable without residue) for ~1 extra matvec; (2)
Titans-style surprise-gated MLP memory is proven at 170-760M and serializes per
user; (3) weight consolidation has a measured recipe (generic replay 5-25%,
constant low LR, 10-30 augmentations/fact both directions, spaced repetition,
merge fuse) and measured skips (EWC, ROME/MEMIT-class editing, LoRA-as-cure).
IDEA 20 written to ideas.md with the three-tier mapping (T1 carried state / T2
serialized state + delta or surprise-gated slow lane / T3 nightly sleep
consolidation) and ladder E1-E5. Own-ledger caveat folded in: delta NaN'd at 10M
pretraining; retry as low-LR adaptation with the new step-skip guard + capped
beta. E1 (arm 1b) training now; E3 (streaming generation loop + state
serialization) is the serving unlock that every tier needs.

arm 1b progress: resumed clean at step 1000, LR re-warmed to 2e-5, ~13.8k tok/s.

## E3 part 1 shipped: streaming generation loop (2026-08-19)

`/generate?fast=stream` (PyTorch backend): unbounded-context generation over
forward_streaming. Full windows commit into the carried states, one forward
each; the partial window is recomputed per byte (avg cost half a window, vs a
full window every byte today) and padded with a non-boundary byte to keep slots
CHUNK-aligned. Prompt never truncated. State commits only from full windows,
which sidesteps the known part-full conv-tail defect entirely. Gates: recurrent
mixer + slots%64==0 (seq multiple of 256; wren's 1024 qualifies). 5 new tests
(tests/mri/test_stream_fast_streaming.py): argmax parity with stream() inside a
window, generation past seq with commits, 3-window prompt consumed whole,
carried state changes next-window logits, clean error on dense models. mri
suite 477 passed, ruff clean. documentation.md inference section updated.
Remaining for E3: per-conversation state serialization (save/reload states
keyed by a session id + a caller protocol that sends only new bytes).

## IDEA 20 refinements + experience log shipped (2026-08-19)

User refinements folded into IDEA 20 (ideas.md): consolidation substrate = the
model's OWN thought and actions (trace consolidation m2 added as an arm vs
fact-SFT m1); sleep is self-contained (rehearsal only from the model's own past,
never new imports); the model may write notes but memorizes them in sleep
(never load-bearing md infrastructure); E6 anti-copy falsifier (generate from
absorbed state vs in-window context, measure verbatim overlap at task success);
E7 skill ingestion (drop a resource, model studies by authoring its own
augmentations, sleep internalizes). Digital-human-brain framing recorded as
project identity.

Shipped: the experience log — data/experience/YYYYMMDD.jsonl, both serving
backends record every exchange (inference/experience.py; EXPERIENCE_ROOT in
readers/paths.py; wrapped stream_c + stream_pt in backends_routes). 6 tests
(tests/mri/test_experience_log.py): lossless byte round-trip, one record per
exchange, disconnect still records, kill switch, never-raise. mri suite 483
passed, ruff clean. documentation.md updated.

## E3 part 2 shipped: persisted conversation state (2026-08-19)

`/generate?fast=stream&state_id=<id>`: carried recurrent states + pending window
buffer serialize to data/stream_states/<id>.pt after every call (completion,
early stop, and client disconnect alike); the next call sends only the NEW
bytes and continues byte-exactly — split-call parity with a single continuous
call is pinned by test (crossing a window commit, and with a real second turn).
State is bound to the exact checkpoint that wrote it (mismatch errors;
state_reset=1 starts over). Loopback-only, id whitelist regex. Tier T2(a) of
IDEA 20 is now real: tell wren something today, the state file holds it
tomorrow — retention quality is what E1/E2 (arm 1b + delta) are improving.
mri suite 486 passed, ruff clean. arm 1b at step 1200: val 0.5615 (was 0.5695
at the 1000 resume point), first checkpoint banked.

## E6 anti-copy baseline measured (2026-08-19)

anticopy_probe.py (scratchpad) on wren1_3@1000, CPU, greedy, 2.5KB reference +
authoring task, 256B generations. In-window: copy_8gram 0.044, engagement
0.028, coherent on-topic proposal. Absorbed-into-state (window cleared):
copy_8gram 0.000 but engagement 0.000 — output is topic-adjacent boilerplate
that loops; no measurable content survived into the state at 1,000 carry steps.
Reading: the yardstick works (copy vs engagement), and the baseline documents
that today's state carries statistics, not content — the E6 verdict waits for
higher-dose (arm 1b) and delta-rule checkpoints. Note: at 200M the in-window
copy rate is already low (small models can't hold long verbatim spans); the
copying pathology E6 targets grows with model scale, so this probe matters
MORE at the next size, and the baseline harness is now one command.

## E2 probe v1 invalid; redesigned as discrimination (2026-08-19)

Generation-form E2 measured zero everywhere — then the positive control
(facts in the SAME window) also scored 0/8: the model completes semantic
associations ("granite is gold"), not arbitrary bound codewords. That is the
role-binding wall (failures.md) re-surfacing, so v1's zeros measured task
failure, not state retention. v2 asks the sensitive question instead:
discrimination margin = NLL(foil|state) - NLL(true|state) over the codeword
sentence, foil = another fact's codeword; for revised facts the foil is the
OLD codeword, so residue shows as a negative margin. K sweep now includes
K=0 as the instrument control. gla baseline running (wren1_3@1000).
Implication for the delta arm: its falsifier line uses margins, not exact
recall — exact-recall codeword binding at 200M is blocked by the wall, which
scale or IDEA 8's bind-then-read must lift, not the state rule.

## E2 v2 gla baseline measured (2026-08-19)

wren1_3@1000, 24 facts (7 revised), discrimination margins. Result: no
reliable retention signal at any depth — plain win rates 0.53-0.59 and revised
0.29-0.71, all inside the n=24/n=7 binomial noise band (chance = 0.5), margins
bounce sign across K. Even the K=0 in-window control is weak (0.53 win,
+1.46 bits from a few outliers): arbitrary-binding discrimination is hard for
the same reason generation was impossible (binding wall). Two clean findings:
(1) save->reload margins are EXACTLY equal to live at every K — E3's state
persistence is lossless, measured; (2) the instrument has dynamic range
(bits), so the delta arm's comparison will be paired same-fact margin deltas,
and n should rise to ~64 facts for the verdict run. This is the "before"
picture the delta arm must beat.

## Arm 1b mid-run cliff read at step 2000 (2026-08-19)

CPU measurement (device note: arm-1 numbers were MPS; compare gaps, not
absolutes). Wrap bucket stream 1.2864 vs slide 0.9099 -> gap 0.377 bits.
At step 1000 the gap was 0.407 (1.2997 vs 0.8929). So the second 1,000 carry
steps closed ~0.03 bits where the first 1,000 closed ~0.14: the dose curve
has flattened hard. stream still beats stream0 everywhere (1.286 vs 1.478
wrap). Reading: gla carry dose is exhausted as a lever; the remaining wrap
gap belongs to the write mechanism — the delta arm is now the main event.
Letting the run finish for the WSD decay phase (final quality + chat ladder),
then evaluating and launching wren1_4.

## Chat streaming UI + sleep platform pieces (2026-08-19)

User directives: verify "what did you just say" recall through the state (no
transcript re-feed), and sleep-when-idle ("index its thoughts, train on them").

Shipped: (1) chat tab wired to the streaming path — fast mode "stream (carried
state)" sends ONLY the new turn with a localStorage per-conversation state_id,
rotated on chat clear and stale-state errors; (2)
tools/build_experience_corpus.py — experience log -> experience_{train,val}.bin
(dedupe, min-reply filter, torn-line tolerant; 4 tests, ruff clean); rehearsal
is just the corpus mixer pulling the model's own base corpus (self-contained);
(3) scratchpad/sleep_trigger.py — idle-gated consolidation launcher (idle ->
index -> low-LR dashboard launch), deliberately NOT armed until the E4 recipe
validates; (4) scratchpad/chat_state_recall.py — the "what did you just say"
behavioral test with fresh-state leak controls, gla baseline in flight.

Note: training tok/s dipped (13.7k -> ~5.1k at step 2610) while CPU probes run
beside it — co-tenancy cost, quality unaffected, ETA stretches. Probes are
short; letting both finish.

## "What did you just say" baseline: 1/6, zero leaks (2026-08-19)

chat_state_recall.py on wren1_3@1000, streaming path, turn 2 sent ALONE with
the state file as the only bridge, per-item fresh-state leak control. Result:
1/6 recalled, 0 leaks — "Where did I just travel to?" -> "You traveled to
Norway." is the platform's first behavioral cross-turn recall through carried
state with no transcript re-feed. The other five answered "I don't know"
(the sft_idk abstention showing, which is the right failure mode — no
hallucinated answers). This is the user's own acceptance test; the number to
move is 1/6. Delta arm and consolidation both target it.

## Arm 1b closeout + wren1_4 launch (2026-08-19)

Full 3000-checkpoint eval sweep: cliff gap 0.383 (vs 0.407 @1000, MPS both) —
dose falsified, recorded in failures.md; E2 discrimination at chance both
doses, reload lossless again; recall 0/6 (1/6 @1000, n-noise); anticopy
absorbed engagement 0.0. Chat ladder: loop 0.20 closed 0.97 median 183B
grounded 0.38 identity 1.00 vs anchor (0.20/1.00/168/0.25/1.00) — held, with
identity recovered and grounded up; wren1_3@3000 is the new streaming base and
worth a look as chat ship candidate after the memory campaign (single-ladder
n, not a ship decision). Seeding wren1_4 (delta) from wren1_3@3000 and
launching arm 2.

## Delta-rule underflow found and fixed; wren1_4 relaunched (2026-08-19)

wren1_4 attempt 1 "completed" in minutes: all 1,000 steps skipped on
non-finite LOSS. Diagnosis (CPU, hooks): first nan inside the second recurrent
block's delta math; chunk trace pinned it to the state-update line computing
the decay ratio as exp(a_last)/exp(a_t) — with trained decays a chunk's cumsum
passes ~-88, both exps underflow to 0 in fp32, 0/0 = nan. The gla path always
used the safe difference form exp(a_last - a_t) (why gla never nan'd), and
this closes the failures.md "delta not trainable" mystery: same text-dependent
underflow, sporadic at 10M random init (step 2094), immediate on trained 200M
decays. One-line fix in model_recurrent.py; forward now finite fp32+bf16;
regression test tests/training/test_delta_underflow.py forces strong decay
through a_proj.bias for both rules + a carried-state window walk. 668 tests
pass. Dead wren1_4 deleted (zero steps trained), re-seeded from wren1_3@3000,
relaunched; first-steps watcher + milestone monitor armed.

## Second delta defect: pre-mask exp inf poisons backward; attempt 3 up (2026-08-19)

Attempt 2's forward was finite but every step skipped on non-finite GRAD.
Cause: the delta chunk computes exp(cl_i - cl_j) for ALL pairs, masks after —
the upper triangle is exp(+large) = inf, discarded by tril in the forward but
saved for backward, where 0 * inf = nan grads. Same bug class as the padding
mask-multiply. gla masks with -inf BEFORE exp (why it never tripped). Fix:
delta now masks the exponent pre-exp. Regression extended with a backward
finiteness test for both rules; verified red on the reverted code, green on
the fix. 670 tests pass. wren1_4 re-seeded and relaunched (attempt 3),
first-steps watcher + milestone monitor armed.

## Delta@600 content read: still chance; recall curriculum built (2026-08-20)

E2 margins on wren1_4@600: chance at every K, same as gla. Interim reading
(not the verdict — that is the 1000 evals): the write rule alone may not be
the binding constraint, because NOTHING in the corpus rewards binding recall —
chat/prose rarely restates a fact verbatim windows later, so the loss never
pays for storing addressable content, whatever the rule permits. Arm 3 fuel
built regardless of the verdict: tools/build_recall_corpus.py — synthetic
codeword examples stated once, buried under 1-6 windows of real filler,
restated at the end (second statement predictable only from memory; the byte
loss itself rewards retention); a third of examples revise the codeword
mid-filler and the final statement uses the newest value (rewards in-place
revision, delta's specialty). 3 tests, ruff clean. Arm 3 shape if 1000 stays
flat: continue the delta adaptation with recall_curr mixed at ~10%.

## Arm 2 verdict + arm 3 launched (2026-08-20)

wren1_4@1000 full sweep: wrap gap 0.380 (= gla 0.383), val 0.583 (worse than
base 0.549), chat degraded (identity 0.67, closed 0.83, grounded 0.12; loop
improved 0.10), E2 chance, recall 0/6. Configuration falsified -> failures.md.
THE FINDING: anticopy absorbed arm jumped 0.06 -> 0.47 copy with first nonzero
engagement — the delta state transports retrievable verbatim content across a
cleared window; capacity exists, retrieval-on-demand was never trained
(chat/prose corpora contain no binding-recall dependencies). Arm 3 launched:
wren1_4 continued 1000 -> 2000 with recall_curr mixed at 10% (4,000 examples,
13.9MB built from hansard filler), chat mix preserved. Falsifier in the run
description; milestone monitor armed.

## User corrections + flagship planning notes (2026-08-20)

User: the 800M was never trained to optimal — the role-binding wall evidence at
800M is from an UNDERTRAINED model and is not conclusive; treat the wall as
established only at well-trained 122M/200M. Well-trained large model is a
future task, after the memory experiments nail down. User wants: (a) sleep
without external runtimes (confirmed: the whole loop is platform-native), (b)
a 500M-vs-1B recommendation, (c) IDEA 21 — grow wren's trained net into the
larger flagship instead of retraining from scratch.

## Naming convention (user, 2026-08-20)

Major version = base/size change: wren2 is the 500M flagship (grown or fresh,
a new base flips the major), wren2_x its adaptations. Recorded.

## Arm 3 verdict + server restart (2026-08-20)

Full sweep on wren1_4@2000 (failures.md holds the entry): wrap gap 0.358 —
best yet and the first real movement; content transport stable (absorbed copy
0.50 / engagement 0.042); BUT binding still absent (plain margins chance,
revisions resolve to the FIRST stamp under both phrasings — the state carries
familiarity + primacy, not noun->word binding), recall 1/6, and the single
template overfit into behavior ("The codeword for poroge is poroge" loops on
ordinary prompts; identity 0.67, closed 0.87, median 123B). Three state arms
now agree the binding wall lives inside the state mechanism at 200M —
independent corroboration of IDEA 8. Curriculum v2 retry conditions parked in
failures.md. Priority shifts to E4 sleep: exact facts belong in weights; the
state's proven role is transport + gist + working memory.

Ops: the long-lived server task (from the 08-18 venv fix) died during a pip
phase (exit 241) — dashboard was down briefly, no run active. Relaunched
detached via veritate.py; watcher armed.

2026-08-20 — chat extraction purge (user-ordered). Removed the Chat tab
(index.html block + chat_tab.js/css), the standalone /chat page (hybrid.html +
app.py route), and the /hybrid/* chat endpoints with their chat-only helpers
(hybrid_routes.py: /hybrid/chat, /hybrid/chat/stream, /hybrid/health,
/hybrid/models, /hybrid/kb/upload, memory compaction, context meter, system
budgeting, remote-model picker, KB upload). auth PUBLIC_PREFIXES trimmed to
/static. Kept: /v1/chat/completions + /v1/chat/mri, ChatML framing + routing,
BM25 retrieval (hallucination detector), Generation tab incl. fast=stream +
state_id, experience log, all corpora/builders. Tests: test_chat_compaction.py
deleted, hybrid route tests dropped from test_openai_chat.py/test_mri_optin.py,
delta-stream tests rebased onto _local_stream_items. CHAT_HANDOFF.md written at
repo root; documentation.md updated (auth surface, inference bullet, tabs).

2026-08-20 — sleep controller shipped (IDEA 20 T3 operational layer, user
directive: dynamic sleep + wake button + checkpoint hygiene, all on the
Generation tab). training/sleep.py + routes/sleep_routes.py (/sleep,
/sleep/wake, /sleep/now) + watcher thread in app.py (60 s tick, skipped in
--minimal). Usage-scaled dose (new exchanges × sleep_steps_per_exchange,
clamped 50–500), idle gate off the experience-log mtime (no serving hooks),
recipe reused from the model's own config.json training_args with only sleep
levers overridden, dense sleep ckpts (every 25) + auto-prune (intermediates
deleted at run end, finals thinned to sleep_keep_finals, non-sleep ckpts never
touched). 12 sleep_* settings in runtime/settings.py DEFAULTS; disabled by
default until E4 validates. Gen-bar chip: awake countdown / sleeping progress
+ eta + wake button. tests/training/test_sleep_controller.py (8, green);
ruff clean; node --check clean. documentation.md backend section updated.
claude_preflight.md: scope made a standing rule (push back once on any
expansion beyond train/eval/run, even from the user; chat extraction noted).

2026-08-20 — E4 night-1 attempt 3 running: earlier relaunch died with the
08-19 server incident; relaunched 16:02, step 60/300 at 16:20, loss 1.13→0.59,
lr 5e-6 flat, ~15.3 s/step, terminal ~17:15. Baseline val bpb locked for the
forgetting kill line (wren1_3@3000): mixed_chat 0.824, veritate_chat 2.071,
hansard 1.059 (scratchpad/val_bpb.py, 24×1024 B fixed windows per bin).
Post-sleep gate: mixed_chat ≤ 0.840 (+2%).

2026-08-20 — cardinal optimization track: wren1_3@3000 exported to v13 fp16
(POST /export/wren1_3, 541 MB) and ported to cardinal (config.json +
veritate.bin + train.csv, md5-verified); dashboard picked it up warm without a
restart and generated a coherent identity reply via /v1/chat/completions;
active C slot restored to wren1_0_int8 after. Decode on cardinal (CLI
chat_greedy, 200 B, traceless): wren1_3 p50 9.48 ms/byte, wren1_0 9.53,
wren1_0_int8 6.63 (int8 1.43x) — the ~35 ms/byte successes.md figure is
dashboard decode with MRI trace on. Finding: v13 int8 COMPUTE already shipped
(hybrid_matvec/matmul_i8 scalar+AVX2+SDOT, per-row weight scales + dynamic
maxabs act quant, engine tests pin SIMD to scalar bitwise) — documentation.md
"computes fp32 end-to-end" and IDEA 21's "int8 kernels must be built" are
stale. Micro-benchmark on cardinal (core-pinned, FFN shapes 200m/400m): scalar
int8 1.9-2.0x vs scalar fp32, BITWISE-OK vs plain-C ref; AVX2 int8 2.3-3.3x vs
AVX2 fp32, 1.8-1.9x vs AVX2 fp16. Engine suite: mac arm64 34/34 green;
cardinal 38/40 — 2 pre-existing failures (also in cardinal's pytest lastfailed
before this session): v9 dense loads SIGILL in prep_b (vpbroadcastq %zmm0) —
plain-C prep_b lives in matmul_vnni.c which builds with -mavx512vnni, clang 18
auto-vectorizes it AVX-512 and it runs un-gated at load; v13 unaffected.

2026-08-20 — seed packs extended to all five verticals. The earlier request was
1,500 seeds PER VERTICAL, not 1,500 total; `conversation` alone was a misread.
Wrote `code`, `technical`, `business`, `medical` at 1,520 seeds / 40 topic
groups each via eight parallel agents (two per vertical, 20 groups x 38 seeds
apiece), merged through a validator that enforces what the test suite enforces
plus the cross-fragment duplicates neither half could see. Ships 7,600 seeds
over 200 selectable topics, zero duplicates within or across packs, ASCII only.
Vertical and genre are orthogonal — pack supplies the subject, genre supplies
the behaviour and its gate — so no new genre was needed. At the measured ~150
openers/seed that is ~1.1M conversations (~2.5 GB) of headroom.
tests/corpus/test_seed_packs.py now parametrises over whatever packs are on
disk (16 -> 37 tests), so a future vertical is held to the same bar the day it
lands; added a check that seeds are subject phrases rather than capitalised
sentences, since they interpolate into "Subject area: {seed}" where a sentence
reads as an instruction. Fixed a latent bug that only bites with >1 pack:
INTERVIEW_TOPIC_STORE was one flat list, so switching vertical overwrote the
other's selection — now keyed by vertical. All five verticals went live on the
running dashboard without a restart (the route reads from disk).
Suite: 921 passed, 9 skipped, 1 xfailed (corpus + mri).

2026-08-20 — SIGILL fix (go-ahead granted): VERITATE_BASELINE_CODEGEN macro in
veritate_engine/src/veritate.h pins un-gated load helpers to SSE4.2 codegen
(target no-avx512*/no-avx2/no-avx); applied to prep_b + prep_b_keep_raw
(matmul_vnni.c) and prep_b_int4 (matmul_int4.c). prep_b_ternary was already
safe (defined in the baseline TU matmul_ternary_scalar.c). New regression test
tests/engine/test_kernel_isa.py disassembles the built binary and asserts no
zmm/ymm in those symbols (linux x86_64 only; skips where inlined by LTO —
observed for prep_b_keep_raw, whose copy inherits the baseline caller's
codegen). Verified: mac arm64 34 passed 3 skipped, behavior unchanged;
cardinal x86_64 40 passed 1 skipped — v9 model LOAD no longer SIGILLs
(prep_b/prep_b_int4 disassemble clean) and hot-path kernels are untouched
(no-op on AVX-512 boxes; prep output is integer/IEEE-identical). REMAINING:
the two v9 golden tests still fail on cardinal one layer deeper —
model.c's dense forward hard-calls matmul_int8_vnni_mt_prep/_vnni_prep
(gdb: SIGILL vpbroadcastq %zmm1 in matmul_int8_vnni_mt_prep <- attention),
a portable symbol with only AVX-512-VNNI (x86) and NEON-SDOT (arm64)
implementations; dense v9/v11/v12 cannot run on AVX2-only x86 without a new
fallback prepped-kernel family + runtime selection (rule 25 obligations) —
needs its own mandate. v13 hybrid unaffected throughout.

2026-08-20 — conversation pack expanded 1,520 -> 3,500 seeds (50 groups x 70).
Six parallel agents: four deepened the existing 40 groups 38 -> 70, each given a
brief listing every seed already in its groups and told to write AROUND them
(different relationships, channels and failure modes) since the obvious
situations were taken and a near-paraphrase buys no openers; two wrote 10 new
groups at 70 each — giving_advice, asking_for_help, explaining_things,
changing_your_mind, everyday_decisions, shopping_and_buying, getting_around,
etiquette_and_rules, memories_nostalgia, downtime_boredom. The merge validator
caught 4 cross-shard collisions no single agent could see (bees in pets vs
hobbies_crafts, nativity in learning_school vs celebrations, damp patch in
home_living vs worries_fears, repeat prescription in health_body vs
errands_admin); the later occurrence was replaced by hand and re-validated.
Pack set now 9,580 seeds over 210 topics, zero duplicates within or across
packs. Went live on the running dashboard without a restart.
Suite: 1,297 passed, 12 skipped, 8 xfailed (skips are environment-gated:
missing chat200m bin, gitignored trainers/corpus, linux objdump).

2026-08-20 — E4 night-1 verdict (wren1_5@300): fwd 6/50, rev 6/50 closed-book
(baseline 0/50 both), dose curve 0→0→2→6 rising. Safety green: mixed_chat
+0.26% (limit +2%), ladder loop 0.10, closure 0.97. Identity 0.83 / grounded
0.25 dips noted as watch items. Full entry in successes.md. Launching night 2:
resume wren1_5 → total_steps 600 absolute, identical recipe (spaced
repetition). Battery on completion: e4_qa_probe {400,500,600}, val_bpb,
ladder. Also shipped: sleep review box below the Generation chat (status +
event history from data/sleep/history.jsonl; sleep-now / wake buttons);
history() + _log_event in training/sleep.py; 9 controller tests green.

2026-08-20 21:59 — OVERNIGHT AUTONOMOUS RUN START (user directive: work,
decide, test until 2026-08-21 09:59, then send summarized report). Open lanes
at start: night 2 at step ~570/600 (battery watcher armed); cardinal sleep
benchmark agent mid-run; sleep UI awaiting dashboard restart (I may restart
between runs — user push to cardinal stays user-gated); branch logic for
night 3 pre-registered in handoff.md.

2026-08-20 22:30 — NIGHT 2 VERDICT: fwd 45/50, rev 47/50 (curve 6→26→38→45).
Identity recovered 1.00; grounded 0.25 (watch). Forgetting +1.49% cumulative
(kill line 0.840 vs 0.83626 — thin headroom). Full entry successes.md.
Dashboard restarted in the idle window: sleep box + usage ledger live
(/sleep serves history + activity_by_hour; real pattern shows 16-17h/20-21h
busiest). Night 3 launched (600→900, recipe unchanged) with per-checkpoint
tripwire: val_bpb at 700/800, auto /trainers/stop if mixed_chat > 0.840;
final battery (exams 700/800/900, val, ladder) armed. Cardinal benchmark
agent still mid-run.

2026-08-20 ~23:20 — Cardinal sleep benchmark landed (agent, 2.5 h on-box).
Headline: recipe-batch sleep is >=920 s/step on cardinal (>=65-80x Mac) — 0/15
steps in 77 min, bf16 CPU emulation + 1-core phases dominate; RAM 13.0 GB
peak; serving survived (p50 flat, worst 5.7x); on-box export .pt->bin 9.8 s;
the staging gap is the 2.17 GB .pt (7.4 min scp). The run resumed REAL
weights (Mac step_0.pt staged, md5-verified). Benchmark also caught 4
controller bugs — all fixed tonight with tests (14 green, training dir 196
green): bookkeeping-key launch refusal, retry-storm (now 60-min cooldown +
"failed" event + panel line), missing corpus-size gate (seq*n_chunks+2),
_model_step now reads latest checkpoint not config "step". Deferred levers
recorded in handoff cardinal track (fp32 CPU sleep A/B, thread cap, batch/seq
override, probe-traffic exclusion). Cardinal restored: sleep off, bin
md5-verified, dashboard serving. Mac server still pre-fix (sleep disabled
here, harmless); fold in at next idle-window restart. Night 3 mid-run,
unaffected.

2026-08-21 ~00:00 — E4 CAMPAIGN CLOSED. Tripwire fired exactly as designed:
step 800 mixed_chat 0.84217 > 0.840 -> auto /trainers/stop mid-night-3.
Dose-response complete: fwd 0(base)->6(300)->45(600)->45(700)->46(800);
rev 0->6->47->49(700)->47(800). Peak = wren1_5@700 (fwd 45/50, rev 49/50,
mixed_chat 0.83632 green). Recall plateaued while forgetting climbed: the
+2% budget binds at ~700 steps at lr 5e-6 — that is the measured ceiling of
this recipe. Retention harness moved into the repo:
veritate_mri/tools/e4_retention_quiz.py + data/eval/e4_facts.json (ruff
clean, smoke-tested); quiz dates 2026-08-27 and 2026-09-19 in handoff.
Ladder@700 running to certify the closeout checkpoint.

2026-08-21 ~00:20 — Closeout certified: ladder@700 loop 0.17 / closed 0.97 /
median 161B / identity 1.00 / grounded 0.25. E4 closeout entry written to
successes.md (dose-response table). Dashboard restarted; serves the fixed
sleep controller (verified /sleep). Per-fact analysis: residual is
RARE-WORD OCCUPATIONS — 4/5 fwd misses @700 are job facts (jobs fwd 21/25 vs
residences 24/25; rev 49/50 near-saturated); model answers in trained format
with wrong occupation (doorman-for-milliner, archivist-for-farrier); 3 job
facts never landed fwd across all 8 checkpoints; 2 landed-then-lost. Recipe
note: augmentation count should scale with object-word rarity. Grounded
probe: @700 = 2/8, all misses are two-entity selection failures (wrong
entity's attribute or refusal-to-select) — running wren1_3@3000 per-item A/B
to separate interference from n=8 noise.

2026-08-21 ~00:25 — Grounded dip CLEARED: per-item A/B (wren1_3@3000 vs
wren1_5@700, same 8 prompts) shows 7/8 identical outcomes; exactly one item
flipped (blue/red folder, 3/8 -> 2/8) = n=8 noise, not interference. The
two-entity selection failures (Amelia/Marcus, Leeds/York, password, eggs)
are missed by the BASE model too — a pre-existing capability gap, not
sleep damage. E4 safety record is fully clean: no metric shows systematic
regression from consolidation.

2026-08-21 ~00:35 — Platform green post-restart (805 passed, 9 skipped,
2 xfailed). Bonus probes on wren1_5@700: ACCEPTANCE TEST ("what did you just
say") = 3/6 recalled, zero leaks — up from 1/6 (wren1_3@1000 baseline);
running wren1_3@3000 (direct fork parent) for clean attribution of whether
consolidation or the parent's improvement bought it. Anticopy absorbed:
copy 0.0241 / engagement 0.0 (looping sample) — unchanged profile, E6
verdict still waits for delta/higher-dose arms.

2026-08-21 ~00:45 — ATTRIBUTION RESULT (secondary discovery): acceptance
test ("what did you just say", 2-turn state recall, leak-controlled):
wren1_3@3000 (direct fork parent) = 0/6 — all abstentions ("I don't know");
wren1_5@700 (slept child) = 3/6, zero leaks. Sleep consolidation itself
moved cross-turn state recall 0 -> 3 of 6. Reading: fact-QA consolidation
taught answer-from-what-you-hold over the sft_idk abstention reflex, and
the gain shows up through the carried state with no hallucination cost on
the fresh-state control. Goal remains 6/6. Logged to handoff + ideas.md
acceptance line + memory.

2026-08-21 ~01:15 — User-approved deletions executed: models/wren1_1 (v2 SFT,
falsified), models/wren1_2 (guard distillation, falsified), models/wren1_4
(delta arm, falsified twice; user accepted losing the delta checkpoint) —
~54 GB freed. Fleet now: wren_base, wren1_0, wren1_3, wren1_5. Publications
agent delivered 3 drafts to publications/ (arXiv-style paper, blog post,
technical note) — honest limitations included, retention quizzes flagged as
pending. wren1_5 thinning deferred pending growth-source decision.

2026-08-21 ~01:40 — E4 m2 arm LAUNCHED (user-ordered: settle whether raw
transcripts consolidate before wren2 starts). New repo tool
tools/build_fact_chats.py: same 50 facts as natural multi-turn conversations
(user tells, assistant echoes so the mask sees fact bytes, distractor turns,
reverse framing at ~1/3 natural frequency) -> fact_chat bins 368KB/19.5KB.
wren1_6 = fresh fork of wren1_3@3000 (step_0 seeded). Night 1: 300 steps,
identical dose/recipe to m1 night 1 (lr 5e-6 flat, assistant mask,
ckpt_every 100). m1 reference @300: 6/50 fwd, 6/50 rev. Battery watcher
armed (exams 100/200/300, val, ladder). Decision rule: m2 within ~2x of m1's
curve -> raw-transcript sleep is enough, sleep-as-shipped is complete; far
below -> extraction pre-pass needed. fact_sft also added to the corpus
catalog as native (deterministic rebuild documented in the entry).
Earlier same hour: user approved + executed deletion of wren1_1/wren1_2/
wren1_4 (~54GB); publications drafts delivered to publications/.

2026-08-21 ~02:00 — wren2 growth machinery SHIPPED (agent): training/grow.py
gained checkpoint-to-checkpoint function-preserving growth + CLI; proof in
tests/training/test_grow_function_preserving.py (9 tests green, 56
trainer-adjacent still green). Measured: max |dlogits| <= 4.8e-7 fp32 across
width/heads/ffn axes, exactly 0 for depth; grads finite on all params incl.
new. Key exactness decisions logged in the file header (RMS-exact duplicate
scaling, tie-preserving lm_head, zero-proj new heads/blocks, GLA o_norm/
D**-0.5 cancellation). Refuses delta/pinned/looped/MoE variants. Recommended
wren2 path: wren1_3@3000 (gla/hybrid — verified compatible) -> trainer_sizes
"400m" 24L/1280/5120/20h ~472M, head_dim 64 preserved, no head_dim growth
needed. Optimizer state dropped by design -> continue with warmup > 0.
documentation.md "model growth (IDEA 21)" section added. m2 (wren1_6) at
step 60/300 mid-run, on pace.

2026-08-21 ~02:40 — Growth shipped to the dashboard (user-ordered): POST
/models/grow + /status + /options (server-side target filtering via the
tool's own validate_growth), Training-tab "grow model" modal with params
before->after and the not-an-upgrade caveat, full continue-flow handshake
verified (step-0-only discovery, fresh-optimizer guard, strict load).
27 tests green (8 new route tests). Reviewer caveats recorded: heads come
from the size preset on resume (UI restricted to size keys; raw-API explicit
shapes with novel head counts strict-fail safely); variant refusal is
config-based at POST, weight-based in the worker. documentation.md growth
section extended. User directives captured to memory: wren2 = conversation/
writing ONLY (no code), clean-data bar, dense ckpts, seq/context expansion
wanted, git push before launch.

2026-08-21 ~03:10 — Context (seq) growth axis SHIPPED: pos_emb/slot_pos_emb
extension (copy-of-last init, never interpolation), bit-exact preservation
asserted delta==0.0 on in-domain inputs, slot-cap subtlety traced and
documented (previously-truncated boundary overflow becomes real capacity),
stride derived from checkpoint, refusals for shrink/non-multiple. Route/UI:
target_seq + seq selector (1x/2x/4x), seq-only growth supported. 34 tests
green (4 new tool + 3 new route), trainer regression 56 green, ruff clean.
export.py verified: grown-seq models export to v13 unchanged (header carries
seq+slots). wren2 can now be grown to 400m shape AND seq 4096 in one call.
