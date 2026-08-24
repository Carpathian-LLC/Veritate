# handoff

## WREN2 WATCHER MANDATE (2026-08-23 21:xx) — for the agent assigned to wren2 only

A second agent is working cardinal-01 (inference speed, sleep on weak hardware, UI) and must not
touch this run. If you are the wren2 watcher, this section is your whole job.

**Hard rule: never touch the wren2 process, its checkpoints beyond the prune policy below, its
`train.csv`, or launch anything on the Mac's GPU.** No training launches, no model loads, no
evals on this box while wren2 runs. If wren2 dies, do not restart it -- report and wait.

State at handoff: pid 34932, step 17,340 / 144,000 (~12%), train ~0.72-0.81 (noisy per row),
val 0.745226 @ 17,250 (prior: 0.7432 @ 17,000, 0.7322 @ 16,750, 0.7346 @ 16,500), ~6,193 B/s,
13 checkpoints, 999 Gi free. LR flat 3e-4 wsd until the decay tail. ETA ~15 days.

1. **Checkpoint prune, every 10 min.** List `models/wren2/checkpoints/step_*.pt`. KEEP step_0
   (grown seed), every step divisible by 5000 (milestones), and the 8 newest. Delete the rest.
   Run only while the trainer lives. Unpruned this fills ~6.7 GB/h and the disk dies in ~6 days.
2. **Health watch.** Trainer liveness (pid 34932 / `veritate_trainer.py --name wren2`); new Python
   `.ips` crash reports in `~/Library/Logs/DiagnosticReports` (the MPS tiled-bmm segfault class);
   `train.csv` still advancing (a row every ~20 steps, ~3.5 min); val trend every 250 steps.
3. **Judge val by TREND, not level** -- LR is flat, so single-row rises are noise. Escalate only on
   a sustained rise across 4+ val rows, a dead process, or a new crash report.
4. **MPS constraint (hard):** bs x heads x seq^2 must stay under 2^31 attention elements.
   bs16@seq2048 is the proven-safe shape. Late-phase plan (campaign end, not now): grow 2048->4096
   via `training/grow.py --seq` (bit-exact pos extension) + a short bs<=4 extension run.


2026-08-23 session-end handoff (user decommissioned the session). Ledgers hold the full numbers; this is state only. READ THIS SECTION FIRST — two watchers died with the session and one must be re-armed promptly.

## SESSION END 2026-08-23 — immediate actions for the next session

1. **RE-ARM THE WREN2 CHECKPOINT PRUNE WATCHER (do this first).** wren2 is mid-pretrain with `ckpt_every 250` (~43 min per ~4.8 GB checkpoint ≈ 6.7 GB/h). The rolling prune loop ran as a session-bound background shell and DIED with the session; unpruned, the disk (~1.0 TB free) fills in roughly 6 days. Policy to reimplement (session script, not in repo): every 10 min, list `models/wren2/checkpoints/step_*.pt`; KEEP step_0 (grown seed), every step divisible by 5000 (milestones), and the 8 newest; delete the rest; run while the trainer lives. 12 checkpoints on disk at handoff time, disk healthy.
2. **Re-arm a wren2 health watch** (also died): trainer-process liveness, new Python .ips crash reports in ~/Library/Logs/DiagnosticReports (the MPS tiled-bmm segfault class), and val trend from models/wren2/train.csv (val logs every 250 steps). At handoff: step 17,160/144,000 (~12%), train 0.79, val 0.7432 (down from 0.75 plateau), ~6,230 B/s, ETA ~15 more days. LR is flat 3e-4 wsd until the decay tail; judge val by trend, not level. MPS constraint (hard): bs×heads×seq² must stay under 2^31 attention elements; bs16@seq2048 is the proven-safe shape. Late-phase plan: grow 2048→4096 (bit-exact pos extension via training/grow.py --seq) + short bs≤4 extension run at campaign end.
3. **Cardinal**: the user pushed and was updating the box at session end. The weak-machine sleep speed benchmark (fp32-vs-bf16 arms) was CANCELLED mid-run by the user — its partial numbers are lost; rerun is cheap (design in worklog 2026-08-23; arms: bf16 baseline repro, fp32, batch, threads; dashboard-only training on cardinal, 20-min/step abort). A stray benchmark trainer child was pkill'ed clean; cardinal dashboard was up and serving afterward. After the box updates, its settings/state migrate to per-model sleep automatically; sleep is DISABLED there — arming cardinal's public model is a user decision and should wait for the extractor wiring (see 5).
4. **Mac sleep settings at handoff**: sleep_enabled False, sleep_models [] (user was experimenting, ended clean). While wren2 trains, any sleep attempt correctly returns "trainer busy" — nothing can sleep on the Mac until the run ends.
5. **Memory product chain status**: extractor SHIPPED (tools/extract_facts.py, closed-loop precision 1.000 / recall 1.000 on the 50-fact benchmark; v2 gap = first-person facts); per-model sleep SHIPPED (enrollment via `sleep_models`, own-conversations-only isolation, turn-taking, per-model state; recorder bug fixed in backends_routes so records now carry the model dir name); `sleep_use_extraction` setting exists but is UNWIRED — next build step: make the sleep corpus lane call build_fact_bins() when it's on, then validate end-to-end with an E4-m3 night (extracted-from-chat flashcards vs m1's curve) once the GPU frees. m2 (raw transcripts) is falsified (failures.md); wren1_6 kept as its artifact.
6. **UI state (all live on the restarted Mac server)**: Settings → Models row = two side-by-side boxes, Warm models + Sleeping models (master enable + per-model checkboxes, multi-enroll verified); Generation tab sleep panel (between prompt and letter timing) is status/history only, per-model rows with sleep-now/wake, click feedback flashes the controller's verdict (green success / orange reason); app updater auto-wakes a sleeping model instead of blocking (real training still blocks). Browser cache bit the user repeatedly this session: after UI edits, tell them to hard refresh.
7. **Standing dates/gates**: retention quizzes for wren1_5@700 on 2026-08-27 and 2026-09-19 (`python -m tools.e4_retention_quiz wren1_5 700`, facts in veritate_mri/data/eval/e4_facts.json; acquisition reference fwd 45/50 rev 49/50). E4 campaign is CLOSED (successes.md). Checkpoint thinning of wren1_5 (keep 0+700) still a user call. publications/ holds three E4 drafts awaiting user review.
8. **Session-bound artifacts that did NOT survive**: scratchpad scripts (val_bpb.py with the config-aware shape fix, ladder.py, e4_qa_probe.py, launch payload JSONs, watcher scripts). The repo-tracked equivalents that matter (retention quiz, extractor, grow tool) are safe. If a new session needs val/ladder probes, rebuild from the ledger descriptions or the originals referenced in worklog.

## CARDINAL NIGHT 2026-08-23 (inference + weak-hardware sleep)

Cardinal is now a real git checkout of `dev` (`~/Veritate`, was an untracked copy; `git init` +
reset, no local edits were lost -- it differed from origin only by the files that session had
changed). Deploy loop: push dev on the Mac, then on cardinal
`git fetch origin dev && git reset --hard origin/dev`, restart with
`setsid nohup .veritate_venv/bin/python veritate.py --skip-build --no-browser --port 8001`.
Reach it at `ssh -p 2222 cardinal-01@127.0.0.1`, dashboard tunnelled to localhost:8011.

**Answered: the "50% CPU while generating" question.** Not psutil, not `DEFAULT_THREADS` (unset),
not the BIOS clamp. The ENGINE calibrates its worker count on a 1,2,4,8 ladder and stops when a
rung improves less than `HYBRID_CALIB_KNEE` (13%); on cardinal 4->8 measures 10.4% so it picks 4
of 8 cores. Numbers and the instability (wren1_0 cached 8, wren1_3 cached 4) are in successes.md.
`engine_threads` now pins it (0 = calibrate). The C-side fix (knee per step class, or a
sustained-load re-check) is NOT done and needs an engine rebuild on each arch.

**The real cause of the lag was sleep, not the engine.** A sleep child took 7 of 8 cores, so a
served request ran at 44.6-54.6 ms/byte with a 2.9 s first byte. Preemption (below) removes it.

**Shipped tonight** (all on dev, all with tests, repo ruff-clean, 1536 tests green):
- `runtime/serving.py` beacon + `sleep.yield_to_serving` / `resume_if_quiet`: the sleep child is
  SIGSTOPped for each request and resumed after `sleep_resume_s`. Serving under an active sleep
  run measures 17.9-18.2 ms/byte / 12-23 ms first byte, same as an idle box. Verified at the OS
  level (child state R -> T for the request's duration), not inferred from latency.
- `sleep_reserve_cores` / `sleep_nice` via `_cpu_budget` / `_nice` run modifiers; `trainer_runner.pid()`.
- `sleep_batch_size` override; val-bin floor (earlier commit) is what finally let sleep launch here.
- bench ramp gained a wall-clock budget AND a per-rung deadline; without the latter a first rung
  slower than the whole budget ran unbounded, which is the normal case on this box.
- UI: sleep panel copy no longer claims idle-only, redundant enroll chip removed, a parked run
  reads "paused for you"; Settings gained pause-while-answering, cores-kept-free, engine-workers.
- Fixed: `memory` corpus topic was in neither the test vocabulary nor the dashboard modal, so
  `fact_sft` could not be browsed. Repo-wide ruff violations cleared.

**Known-good serving numbers on cardinal** (wren1_3 fp16, greedy, engine-direct): 9.9 ms/byte
engine p50, 18.3 ms/byte end to end. The gap is NOT the Python layer (0.02 ms/byte, 0.1%) -- it is
boundary steps: word-initial bytes run the GLA global-block stack and cost 50 ms vs 10 ms, 54% of
decode. int8 is 1.46x on identical weights and is the first lever (successes.md).

**Two dead ends, recorded so nobody re-runs them:**
- bf16 emulation is NOT why CPU sleep is slow. Raw bf16 matmul on this AVX2-only box really is 424x
  slower than fp32 (0.33 vs 141.68 GFLOP/s, successes.md), but `hardware.resolve_precision("bf16",
  "cpu")` already returns None, verified on the box -- every CPU run is fp32 already.
- The engine/serving stack is not the bottleneck either: 0.02 ms/byte, 0.1% of decode.

**OPEN / next session:**
1. `perf_trace.py` is BROKEN: imports `FRAME_PAYLOAD_BYTES`, which is now per-instance
   (`sub._frame_payload_bytes`). Not fixed -- I used a scratchpad probe instead.
2. Sleep throughput on this box is the honest limit: a batch-4 step (16,384 tokens) had not
   finished in ~5 min. Because a SIGSTOP loses no work, the right weak-box shape is the LARGEST
   batch that fits (amortizes Muon's batch-independent Newton-Schulz cost), i.e. leave
   `sleep_batch_size` at 0. Confirm with the bench ramp now that it terminates.
3. **int8 was exported and measured, then REVERTED.** 1.72x end to end (17.5 -> 10.2 ms/byte with
   8 engine workers), but greedy parity is FALSIFIED (failures.md): fp16 and int8 from the SAME
   checkpoint agree on 1 of 5 replies. Cardinal is back on its original fp16 bin
   (md5 35b040bd, verified). Two backups left in `models/wren1_3/`: `veritate.bin.fp16.bak`
   (identical to what is served) and `veritate.bin.int8.bak` (keep -- it is the artifact a quality
   eval would grade). Shipping int8 needs that eval, not a parity check.
   TRAP: cardinal's stored wren1_3 bin is NOT an export of its own step_0.pt (35b040bd vs 6c1856ec).
   Re-export BOTH arms from one checkpoint before comparing anything.
4. Cardinal settings are RESET to defaults with sleep OFF. `engine_threads` is pinned to **8**
   (measured 17.51 -> 15.2-16.2 ms/byte, ~10%, output identical); everything else is stock.
5. Browser cache: after any UI deploy tell the user to hard refresh.
6. **The real weak-box bottleneck is IDEA 23 (ideas.md): CPU training runs single-threaded.**
   CPU time grew at 45% of one core over 5 min with a 7-core budget, no contention, fp32, not
   suspended; three consecutive samples showed 1 running thread of 13. Utilization is low and
   variable (168-306% at other moments), so this is a sustained serial PHASE, not a serial run. Next step is a `torch.profiler` pass
   over two steps grouped by op -- a profile, not a guess. This is the last thing between preemptive
   sleep and a model that measurably improves overnight on commodity hardware.
7. A full sleep cycle was NOT observed completing on cardinal: a batch-4 step did not finish in ~20
   wall min, so the run was woken deliberately. The lifecycle either side of the step IS validated
   (launch -> suspend -> resume -> wake -> finalize -> history -> cooldown, all verified on the box).

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
- **Sleep controller SHIPPED 2026-08-20 (disabled by default), PER-MODEL 2026-08-23**: training/sleep.py + /sleep, /sleep/wake, /sleep/now + 60 s watcher thread. Per-model sleep: enrollment is the `sleep_models` list (old `sleep_model` string migrates automatically in settings._ensure_settings); enrolled models take turns on the one trainer — per idle window the model with the most pending OWN exchanges above sleep_min_exchanges sleeps, the next waits for finalize. Own-conversations-only: experience records now carry the model dir name (backends_routes fixed 2026-08-23; older records attribute only via uniquely-owned artifact basename, ambiguous ones train nobody); the corpus build runs over a per-model filtered view in data/sleep/filtered/. Usage-scaled dose (own exchanges × steps_per_exchange, clamp 50–500), idle gate off experience-log mtime (box-wide), recipe from the model's own training_args (sleep levers overridden: constant sleep_lr, no warmup, assistant mask, ckpt every 25), auto-prune per model (run intermediates deleted, finals thinned to sleep_keep_finals, non-sleep ckpts never touched), per-model cooldown on a failed sleep. State keyed by model in data/sleep/state.json (flat pre-per-model file migrates on load). Routes take a `model` param (omitted = the only enrolled model, else 400). `sleep_use_extraction` setting reserved for the experience builder's extraction mode (unwired until that lands). UI: gen-bar chip one-line summary + review box with one row per enrolled model (state, pending, last slept, next eligibility, own sleep-now/wake); history + activity sparkline stay global. Tests: tests/training/test_sleep_controller.py (20) + tests/mri/test_sleep_routes.py (5) green. Note: the live server predates this; per-model panel appears after the next restart.
- **E4 night-1 VERDICT (wren1_5@300, 2026-08-20) — first nonzero closed-book recall; full entry in successes.md**: fwd 0→0→2→6/50, rev 6/50 (baseline 0/50 both), dose curve rising at run end. Safety green: mixed_chat 0.82608 = +0.26% of the +2% budget (kill line 0.840 untouched), veritate_chat +0.07%, hansard +1.02%; ladder loop 0.10 / closure 0.97 / median 83 B. Watch across nights: identity 0.83 (anchor 1.00), grounded 0.25 (anchor 0.38). 10 facts landed, 2 bidirectional; cross-binding (right towns, wrong people) is the dominant remaining error.
- **E4 night 2 VERDICT (2026-08-20 ~22:20) — VALIDATED, full entry in successes.md**: fwd 6→26→38→**45/50**, rev 6→27→42→**47/50** across steps 300-600. Identity recovered to 1.00; grounded still 0.25 (last watch metric). Forgetting cumulative: mixed_chat +0.26% (night 1) → **+1.49%** (night 2) of the +2% budget — kill line 0.840 not crossed (0.83626) but headroom thin. Night 3 (→900, same recipe) launched with a per-checkpoint tripwire: val_bpb at 700/800; trainer stopped automatically if mixed_chat > 0.840.
- **WREN2 TRAINING LIVE (2026-08-21 ~11:45)**: grown from wren1_3@3000 (270M→598M, 24L/1280/5120/20h) at seq 2048, continue-pretrain 144,000 steps × 65,536 B = 9.44 GB on the conversation/writing-only mix (data plan in worklog 2026-08-21), lr 3e-4 wsd, warmup 500 (fresh optimizer), loss_mask off, ckpt/eval every 250 with a rolling prune watcher (keep seed + 5000-milestones + newest 8) and a persistent health monitor. ~6,300 B/s → **ETA ~17 days**. **Hard platform constraint discovered**: torch MPS segfaults (SIGSEGV, tiled_bmm) when an attention bmm exceeds 2^31 elements — bs×heads×seq² must stay < 2.1e9 (bs16@seq4096@20h = 5.4e9 crashes; bs16@2048 = 1.34e9 fine). Late-phase plan: grow 2048→4096 (bit-exact pos extension) + short bs≤4 extension run at campaign end. m2 arm falsified same morning (failures.md): raw-transcript sleep 0/50 vs fact-SFT 6/50 at matched dose — the tell-it-once loop needs the extraction pre-pass (chat → facts → build_fact_sft), self-contained. wren1_6 (m2 arm) retained for reference; sanity gate for growth passed (mixed_chat + hansard bit-exact, veritate_chat +0.23% = documented slot-overflow domain).
- **E4 CAMPAIGN CLOSED (2026-08-21 ~00:00) — tripwire stopped night 3 at the forgetting ceiling, exactly as designed**: step 700 = fwd 45/50 rev **49/50**, mixed_chat 0.83632 (green); step 800 = fwd 46 rev 47, mixed_chat 0.84217 > 0.840 kill line → auto /trainers/stop. Recall plateaued while forgetting climbed: the budget binds at ~700 steps at lr 5e-6. **wren1_5@700 is the E4 closeout checkpoint.** Retention quizzes (zero further training, closed book): run `python -m tools.e4_retention_quiz wren1_5 700` (repo tool; facts preserved at veritate_mri/data/eval/e4_facts.json) on **2026-08-27 (7-day)** and **2026-09-19 (30-day)**; acquisition reference fwd 45/50 rev 49/50. Checkpoints step_0..800 all kept — nothing deleted, thinning is a user call. Post-closeout analysis (2026-08-21, worklog): residual fwd misses are **rare-word occupations** (jobs 21/25 vs residences 24/25; 3 job facts never landed) — augmentation should scale with object-word rarity; the **grounded dip is CLEARED as noise** (per-item A/B vs anchor: 7/8 identical, one flip; the hard two-entity selections fail on the base model too). Safety record fully clean. **Secondary discovery**: the acceptance test ("what did you just say", 2-turn state recall) jumped **0/6 (parent wren1_3@3000, all abstentions) → 3/6 (wren1_5@700, zero leaks)** — consolidation itself unteaches the abstention reflex in favor of answering from held state (worklog 2026-08-21; ideas.md acceptance line). Anticopy unchanged (absorbed 0.0241/0.0 — E6 verdict still open).

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

Goal: wren2 (500M hybrid flagship) serves int8 on cardinal-01. CORRECTED after on-box verification 2026-08-20: v13 int8 compute already shipped (hybrid.c, parity-tested; wren1_0_int8 = 6.63 ms/byte vs 9.48 fp16 on cardinal) — the "engine gap" was stale documentation, now fixed. wren1_3@3000 ported to cardinal (fp16 v13, serving, 9.48 ms/byte p50). Remaining items (detail in ideas.md IDEA 21):
1. **AVX-512 SIGILL load bug — FIXED 2026-08-20** (`VERITATE_BASELINE_CODEGEN` pin on `prep_b`/`prep_b_keep_raw`/`prep_b_int4`; `prep_b_ternary` was already safe in a scalar TU; new tests/engine/test_kernel_isa.py disassembly regression test; verified arm64 34 pass + x86_64 load completes, no-op on AVX-512 boxes). **Deeper pre-existing limitation uncovered**: the dense v9/v11/v12 FORWARD on x86 is AVX-512-VNNI-only by construction — `matmul_int8_vnni_mt_prep`/`_prep` are hard-called (~15 sites in model.c) with no runtime fallback, so dense bins load but cannot forward on AVX2-only CPUs (the 2 golden tests on cardinal now fail there, past load). Fix would be a new prepped scalar/AVX2 kernel family + runtime selection under rule-25 bitwise contracts — its own mandate. LOW PRIORITY: the entire wren fleet and wren2 are hybrid/v13, which has its own dispatch and is fully green on cardinal; dense-on-AVX2-x86 matters only if a dense model is ever shipped there.
2. **QAT pilot demoted to conditional**: PTQ int8 is already greedy-byte-identical at 200M; run the pilot only if PTQ parity degrades at 500M, and target the v13 scheme (per-row + dynamic scales), not qat.py's v9 scheme.
3. **Sleep-on-cardinal benchmark (agent, 2026-08-20 night)** — full raw data in worklog + /tmp/sleep_bench on cardinal. Numbers: recipe-batch sleep step (200M, bs48, seq1024×4 chunks, bf16) is **≥920 s/step on the i7-9700T (≥65-80x the Mac's 14.3 s)** — 0 of 15 steps finished in 77 min; RAM peak 13.0 GB (fits 23 GB); serving stayed up throughout, p50 unchanged, worst 5.7x during the trainer's multi-core phases; **on-box fp16 bin export from a .pt is cheap (9.8 s)** — the real staging cost is the 2.17 GB .pt itself (7.4 min transfer). Four controller bugs the benchmark exposed were **FIXED same night with tests** (14 green): save()-bookkeeping keys tripping the trainer's unknown-flag gate (sleep couldn't launch anywhere), watcher retry-storm on failed launch (now 60-min cooldown + "failed" history event), no corpus-size gate (small night's bins crashed the child; now gated at seq×n_chunks+2), `_model_step` reading config "step" instead of the latest checkpoint (forked models would sleep 0 steps). OPEN weak-machine levers, deferred pending decision: fp32-compute override for CPU sleep (bf16 emulation on AVX2-only silicon is the dominant cost — needs a measured A/B), thread cap + nice for the sleep child, sleep batch/seq override (recipe change — user call), excluding probe/health traffic from experience recording (sleep currently consolidates its own monitoring chatter), staging .pt checkpoints with any model expected to sleep on-box. Cardinal left clean: sleep disabled, original bin restored + md5-verified, staged step_0.pt kept deliberately.

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
5. **Arm the sleep controller? (proposal, 2026-08-21)** E4 validated the mechanism and the controller bugs found on cardinal are fixed, so it can go live on the Mac whenever approved. Proposed settings: `sleep_enabled true`, `sleep_models ["wren1_5"]` (serve from @700), keep defaults except `sleep_max_steps 100` — the measured forgetting ceiling was ~700 total steps at 5e-6, and E4 spent that budget, so nightly doses must stay small until the 7-day retention quiz (2026-08-27) shows whether the bpb cost persists or anneals. Not armed; user call. (Also: git push wanted — engine SIGILL fix, sleep controller + fixes, retention tool, and the web sleep box are all local-only; cardinal update after that push.)
6. Overnight-run checkpoint disk: models/wren1_5/checkpoints holds step_0..800 (9 files ≈ 19 GB). Selection is made (@700); thinning is a user call.
