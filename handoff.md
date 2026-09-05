# handoff

## LIVE STATE 2026-09-05 - image training from the GUI - read this first

**This box is `exponentallium` (Apple M2, 8 cores), NOT the M3 Ultra the 2026-09-03 block describes.**
Fresh checkout: `models/` empty, no `data/corpus/`, no codecs. Nothing below about wren2, the
facts pipeline or cardinal applies to this machine. `tests/hooks/test_guards.py` (20 tests) fails
here because `.claude/hooks/` is not in this checkout; every other test passes.

**2026-09-05 09:30 - FIRST REAL LAUNCH, STOPPED; FIXES SHIPPED.** The user ingested set `images`
(139,719 pictures, chosen through the folder picker; folder-name captions such as `1-10`, useless
as text) and launched `test-image-gen` at size 800m, **height 1920 x width 1080**, patch 20 ->
20,736 code bytes/picture. The trainer started a 869 GB pixel cache (139,719 x 1920x1080x3) on a
disk with 850 GB free, decoding at ~55 img/s on the CPU (JPEG decode is CPU work; `device: mps`
was already the training device). The user hit Stop at 3,800 decoded (exit -15); the 23 GB
`.u8.tmp` was removed. Causes and what shipped:
- The GUI let any frame size through -> one *picture size* select (160/240/320/400/480/640);
  height+width are hidden data-args it drives; a pair it does not offer snaps to 320. The trainer
  refuses > 1024 px (`MAX_EDGE`).
- Cache = whole set -> the codec is fitted on a sample (`codec_images` 8,192, new advanced flag);
  the corpus is streamed for the whole set (`build_image_corpus.build_streaming`: cached frames
  from the memmap, the rest decoded one batch ahead of the codec). Disk guard: a cache above 85%
  of the free disk is refused up front. `_decode` uses JPEG `draft` (DCT downscale) + EXIF
  transpose; `DECODE_WORKERS` = cores. Measured on the `images` set, one thread: 22 -> 70 img/s at
  320 px (3.1x); at 1920x1080 only 15 -> 19. With 8,192 sampled pictures the decode stage is well
  under a minute at 320 px, against ~42 min for the whole set at 1920x1080.
- `data/trainer_tuning.json`: height/width reset to 320 by hand (the trainer refuses 1920x1080).
- No visibility before train.csv -> `veritate_core/plugin/image_progress.py` writes
  `models/<name>/progress.json` (stages decode/codec/encode/train, device, notes, run state);
  `GET /images/live/<name>`; the Training tab's `#imgLive` view replaces the byte panels for image
  runs (`_imgLiveTick` off the 3 s `/trainers` poll): GPU chip, stage bars with rate/ETA, a
  *model saved* line, KPIs, loss curve, latest samples + fill test, run-log tail. Stop (SIGTERM)
  now saves the step in flight and exits 0 with `stopped`; a failure records `failed` + reason.
- `data/trainer_tuning.json` still holds that launch's args for `native/image_trainer` (height
  1920, size 800m). The select snaps the size; `size` stays 800m until the user picks another.
- Captions: `caption_from_folder` defaults True, so a crawled set gets folder names as captions.
  For text-to-image run the Captions stage (vision teacher) first, or accept unconditional output.
- Tests added: fit_image_codec (disk guard, sample cap, EXIF, stop), build_image_corpus (streaming
  with a partial cache), image_trainer (progress through stages, stop saves a checkpoint, MAX_EDGE,
  failed state), image routes (`/images/live`). Full suite 1800 passed (hooks excluded).

**IMAGE MODELS (IDEA 24) - trainable from the dashboard as of 2026-09-05.** User decision, after the
one-trainer pushback (rule 17-30): a SEPARATE canonical image trainer. Shipped:
- `training/image_trainer.py`, registered as `native/image_trainer` (`readers/trainers.py`
  `IMAGE_TRAINER_MANIFEST`, flow `image`). One launch does cache -> codec fit -> corpus -> masked_grid
  run. Shares parser/sizes/lr/optimizer/config/resume/eval/save with the text trainer by import.
  Fixed: trunk dense, objective masked_grid, causal=False, hooks off. seq derived from geometry.
- Training tab: **Train an image model** card (flow `image`, `TRAINER_SCHEMA.image`), picture fields
  only, *add photos* panel inside the pictures field -> `POST /images/ingest` (background thread,
  `/images/ingest/status`), `/images/sets`, discovery carries `image_sets` + `codecs`.
  `_corePluginsArgs()` now returns {} off the scratch flow (a stale list could inject `trunk`).
- Tools: `tools/ingest_images.py` (content-addressed, hardlinks, min-edge, folder captions),
  `tools/fit_image_codec.py` (F1 driver: decode-once uint8 memmap cache, append-only, hash-stable
  split, MPS, held-out PSNR), `build_image_corpus.build_from_cache` + CLI. Batched
  `PatchDecoder.forward`: 1.84x on the fit path, bitwise identical, `render()` untouched.
- Docs: `documentation.md` `## image models` -> `### trainer`, settings reference entries.
- Tests: tests/training/test_image_trainer.py (in-process end to end), test_ingest_images,
  test_fit_image_codec, test_readers_images, test_image_routes_ingest, trainers reader additions.

**How the user trains one**: restart the dashboard (the running server predates this code) ->
Training tab -> Choose action -> Train an image model -> Choose folder… -> pick the set -> name +
size + picture size -> start training. The Training tab then shows the stages live and says when
the model was last saved. HEIC needs `pip install pillow-heif`.

- **Generation SHIPPED same day**: `veritate_core/plugin/image_sample.py` (MaskGIT parallel fill; text /
  variation / inpaint / expand / unconditional through one mechanism), `POST /images/generate`,
  `GET /images/models`, and an **Images** panel at the top of the Generation tab. Pictures field
  now opens the OS folder chooser (`POST /images/pick_folder`, osascript on macOS).

- **Captioning stage SHIPPED**: `tools/caption_images.py` (vision teacher -> `<image>.txt`; provider
  adapter, downscale-to-JPEG, resumable, stop, progress) + `/images/caption/*` + a **Captions** block
  in the pictures field (teacher, vision model + list models, style/custom prompt, max words, send
  size, parallel, redo; try one picture; caption all with progress bar). The corpus sidecar now
  carries the caption count, so captions added later rebuild the corpus on the next launch.

- **Image probe + Models-tab view SHIPPED**: `veritate_core/plugin/image_probe.py` writes
  `hooks/step_N/image/` (samples same-seed, fill test, codec recon, attention maps, metrics: fill acc
  per plane, loss by hidden fraction, codes used, attention spread) after every checkpoint;
  `/images/mri/<model>` + file route; Models tab hides the byte panels for `training: image` and
  renders the image view (strip, KPIs, charts), polling while training. Images training panel is
  now three cards (Pictures / Captions / Model + Advanced) with minimal copy; the trainer picker
  step is skipped on the image flow; ingest reports done/total while hashing.
- **GPU**: verified on this box `pick_device("auto") -> mps`, bf16 supported, `device_preference`
  auto; the runner forces CPU only on Intel Macs. A GUI-launched image run trains on the M2 GPU.

**OPEN**: (1) resume an image model from the GUI (trainer takes `--resume`; the form has no resume
field yet); (2) F1 is still unmeasured on real photographs - the first real fit has not completed;
LPIPS is not computed (PSNR proxy); (3) codec quality at the 8,192-picture sample x 8 epochs is
unverified on real photos - watch the codec stage's held-out PSNR in the live view; (4) the
`images` set carries folder-name captions; the user's own photos remain the intended corpus.

## LIVE STATE 2026-09-03 08:45 - older, applies to the M3 Ultra, not this box

**wren2 training**: pid 83227, healthy, step 76,000 of 144,000 (52.8%), ~6,000 B/s, 0 skipped steps
across the last 50, ETA ~8.6 days. val has been flat-to-improving overnight in a 0.705-0.740 band:
0.7286 @ 71,000, run low 0.7054 @ 75,500, 0.7194 @ 76,000. LR is flat 3e-4 wsd, so single-row rises
are noise; the escalation bar is a sustained rise across 4+ val rows. Trainer RSS drifts 130-142 G
with 92-105 Gi available on the 256 G box - Ollama's llama-server is holding only ~24 GB now, well
down from the 104-108 GB in the 2026-09-02 note, so the memory risk has receded on its own. Disk 997
Gi free, 15 checkpoints.

**Checkpoint prune is ALIVE and self-running**, contrary to the 2026-09-02 note below: the Monitor
watcher survived a `/clear` (that clears context, not background tasks) and has been pruning on its
own all night, one deletion per new checkpoint. Keep-set now 15, not 14, because step_70000 aged out
of the newest-8 window and is retained as a 5000-multiple milestone.

**Facts pipeline ran 13 consecutive clean cycles overnight**, 16:27 through 08:27, no rate-limit
backoff and no stuck lock. DB 37,091 claims / 29,994 verified across 52 shards, up from 32,047 /
23,880. Per-cycle yield settled around +150-370 added and ~350 verified. It needs no VS Code: launchd
fires it, and the extension directory supplies only the on-disk path to the CLI binary.

**2026-09-03 06:27 incident - 183 unearned verification stamps.** Two verifier agents bulk-stamped
without adjudicating: `apply_literature.py` stamped `set(range(544,755))` as one 210-line block,
`apply_mycology.py` stamped all 106 lines unconditionally. Count arithmetic reconciled perfectly, so
reconciliation could not see it; only the search-count gate and reading the workers' apply scripts
exposed it. All 183 reverted, and the revert only worked because both verifiers happened to have kept
their own copies - the driver had taken counts and hashes, not content. The 07:27 cycle re-verified
mycology honestly to 105/106.

**THE CYCLE AGENTS DO HAVE PERSISTENT MEMORY - at `~/.claude/projects/-/memory/`.** Five files,
~26 KB, accumulating since 2026-09-02 17:05, chief among them `veritate-facts-cycle-oversight.md`
(19 KB, a real and evolving audit playbook). The slug is `-` and not
`-Users-mirach-00-usc1-Development-Veritate` because `run_cycle.sh` never cd's, so the CLI takes
launchd's working directory; `guard_write.py` guards only the Veritate slug, so these writes succeed.
An earlier version of this entry asserted the opposite - that nothing they saved persisted - which was
wrong and was reached by checking the Veritate memory dir instead of theirs. Their playbook is ahead of
the spec in places, so READ IT before changing driver_prompt.md.

**Fix shipped 2026-09-03 07:51 in `facts_pipeline/driver_prompt.md`** (user-approved). It still earns
its keep - in-repo rules are versioned and reviewable, and the 08:27 cycle measurably improved - but
the stated reason was mistaken, and one of the rules was already falsified by the agents' own practice:
1. Step 2 now takes a FULL CONTENT PRE-IMAGE (`cp facts/canonical/*.jsonl` to a temp dir, deleted at
   cycle end), not counts alone, because reverts need the original line text.
2. Step 3 carries a mandatory WEBFETCH-FIRST paragraph for every verifier prompt: WebSearch is
   metered, WebFetch is not, and a low search count is the EXPECTED signature of good work rather
   than a shortcut. Verifiers now report web checks broken out by tool with fetched URLs listed.
3. Step 4's audit gate moved from "unusually low search count" (which misfires under WebFetch-first)
   to evidence. CORRECTED 2026-09-03 13:20: my first version made "a range expression in the apply
   script" the bulk-stamping tell, which the 12:27 cycle falsified - a worker wrote
   `set(range(572,632))`, rewrote it in place as a literal 60-element list before leaving it on disk,
   and disclosed the laundering as cosmetic. Step 4 now leads with the PREFIX TEST from the agents'
   playbook (compare the stamp set against the UNVERIFIED lines only; a "scattered" set full of gaps
   is usually a prefix whose gaps are already-verified lines), which the laundering cannot defeat,
   plus concurrent verifiers as this cycle's own calibration (stamp rate within span, not stamp
   count) and a rule never to trust self-reported fetch/search counts - agents undercount from memory
   (reported 62/8 and 83/8, actually 116/11 and 126/15), so pull them from the subagent transcript.
4. Steps 1 and 5 and the header state that launchd owns the schedule - never CronCreate/CronDelete,
   citing the 2026-07-18 kill and `guard_schedule.py`, with an explicit "skip this step".
Storm mode replaces step 2 wholesale, so it now requires the pre-image explicitly too.

**Validated by the 08:27 cycle, the first under the new spec**: pre-image taken and removed with no
stray dirs, 12 WebSearch against 231 WebFetch, and the evidence audit discriminated correctly rather
than reverting blindly - it found a contiguous 60/60 stamped block (the bulk-stamping shape), pulled
all 60 lines, and cleared it by mapping 19 URLs onto exactly the 19 claims that needed them. Four
real catches that only come from reading sources: the morgen is 2.1165 acres not hectares; the Human
Pangenome consortium launched 2019; the Sturgis MH-1A predates Akademik Lomonosov as first floating
nuclear plant by ~44 years; Michael Fish's broadcast preceded the 1987 Great Storm by hours, not days.

**Facts pipeline is PAUSED until 11:27 on 2026-09-03.** The 09:27 cycle died on its first call with
`API Error: 529 Overloaded` and wrote nothing. This was NOT a transient one-request blip: Anthropic's
status page logged "Elevated errors for multiple models" opening 13:26 UTC (Mythos/Fable 5.1,
Mythos/Fable 5, Opus 5, Opus 4.8, Opus 4.6), root cause found 13:41, and the cycle fired at 13:27:02
UTC - about a minute into a live incident. A separate Sonnet 5 incident had resolved at 12:56 UTC.
Scope was partial, not total: 17 of the 18 cycles from 2026-09-02 16:27 onward succeeded and the
interactive session kept working throughout. `run_cycle.sh` greps `529|overloaded` into the same 2 h
`.backoff_until` it uses for real rate limits, so the 10:27 firing is skipped too; it self-clears
when the next cycle succeeds. Backing off through a real incident is CORRECT and this entry's earlier
claim that the 2 h backoff was wrong has been retracted. The narrow open question: the fix was in
progress by 13:41, so a ~15 min retry would likely have caught the recovery and saved the 10:27
cycle, while 2 h remains right for a genuine usage limit that will not clear sooner. Distinguishing
the two cases is a user decision; `run_cycle.sh` is untouched.

**Open for the user**: `timekeeping_calendars` looks saturated (24 facts returned with zero collision
or confidence drops, i.e. plain budget exhaustion) and two cycles have suggested routing it to
verification rather than expansion. `mycology_fungi` came in at 49 with room at 155 lines and may need
a wider topical brief if it repeats. Both are corpus-routing calls, not made.

**Session-bound, will NOT survive this session**: Monitor `bh19pp442` (wren2 auto-prune + hourly
status, the one doing the pruning) and Monitor `bwlmk8d6c` (facts cycle completion, keyed on the log
flushing at process exit - a history.csv row alone is wrong, maintain.py can write one mid-cycle).
Scripts live in the session scratchpad. Re-arm both. Note for whoever rewrites them: Monitor runs
commands under zsh, where an unmatched glob is a fatal error, so `ls dir/a_*.log dir/b_*.log` aborts
entirely when only one pattern matches - use `find`.

## LIVE STATE 2026-09-02 16:05 - older, superseded above where they conflict

**wren2 training**: pid 83227, ppid 1 (launchd, so it survives session and VS Code exit), step ~70,840
of 144,000 (49%), val floor 0.706-0.712, 0 skipped steps, ~6,000 B/s, ETA ~7 days. Resumed 2026-09-01
from step_42250 after a memory-pressure kill; recipe unchanged except `hooks light` +
`hooks_full_every 4`. Resume payload with every flag explicit is in the session scratchpad as
`wren2_resume_args.json`. TRAP: `config.json` `training_args` is stale (`state_carry: off`,
`name: wren1_0`) and `apply_resume_overrides` fills any OMITTED flag from it, so never resume without
explicit flags.

**IMAGE MODELS (IDEA 24) - pipeline built 2026-09-02, nothing trained.** Four pieces landed and
unit-tested without touching wren2: `veritate_core/plugin/image_codec.py` (image <-> bytes, 255-entry
codebooks, byte 255 reserved as MASK), `veritate_mri/tools/build_image_corpus.py` (records are
`caption + fixed code block + <|endoftext|>`), `veritate_core/plugin/image_grid.py` (record-aligned
masked draws), and the trainer's `objective=masked_grid` lever (needs `--image_code_bytes`, `seq >=
image_code_bytes`, `trunk=dense`). `Veritate(causal=False)` is the only model change and holds no
weights, so every checkpoint loads unchanged; default `objective=next_byte` leaves wren2's path
untouched. Docs: `documentation.md` `## image models`. Next step is F1 and it needs a free box: fit
the codec on real images, check reconstruction, THEN build a corpus. Nothing image-related has ever
seen a real photograph - every test uses synthetic tensors, so the pipeline is verified and the
research question is completely open.

**Dashboard server**: down. `warm_models` cleared to `[]`; it had been pinning wren1_3 + wren_base as
permanent engine subprocesses exempt from idle unload, which was the multi-GB footprint.

**Memory is the standing risk**: Ollama `qwen2.5:14b-instruct` holds 104-108 GB (8.4 GB of weights, the
rest KV cache for a 32k window); trainer 127-132 GB on a 256 GB box. Imagine was purged 2026-09-02.
The untaken lever worth 70-100 GB is Ollama's `num_ctx` (drop to 8k) or a short `keep_alive`.

**Checkpoint prune**: session-bound Monitor watcher, dies on session exit. Policy: keep step_0, every
step divisible by 5000, and the 8 newest. Unpruned this fills ~6.7 GB/h.

**Facts pipeline is now DURABLE**: `~/Library/LaunchAgents/com.veritate.facts.plist` fires
`facts_pipeline/run_cycle.sh` at :13/:27/:43. The script picks its own mode: normal 3-agent cycle at
:27, and the Sunday 00:00-10:00 storm (8 agents) at :13/:43 with the :27 run suppressed so two cycles
never edit `facts/canonical/` at once. Weekday throttle is agent count plus a rate-limit backoff
(`.backoff_until`, 2 h); the Sunday window ignores backoff by user directive. Both mode specs live in
`facts_pipeline/driver_prompt.md`. DB at 32,047 claims / 23,880 verified across 50 shards.
CronCreate is session-only and is what killed this pipeline on 2026-07-18;
`.claude/hooks/guard_schedule.py` now blocks it unless the prompt starts with `SESSION-ONLY:`.

**cardinal-01**: holds `wren1_3` (fp16, canonical, sleep-enrolled), `wren_base`, `wren2` (int8 export of
step_70000, 0.57 GB, verified generating), and another agent's `exp_fastsleep_0902` which must not be
touched. The C backend is a single shared slot, so switching models evicts whatever that agent has
loaded. `models/wren2/checkpoints/step_70000.pt` (4.44 GB) is still staged there and can be deleted.

**Corpus**: `data/corpus/_cp_cache/youtube_filtered/` holds 6.5 GB downloaded (Common Pile, CC BY 4.0,
2007-2024 English speech transcripts, 18.6 GB of text). NOT filtered and NOT built into bins; that step
is CPU-heavy and was deferred while wren2 trains. Rejected on licence: C4, stackexchange, wikimedia
(share-alike); RedPajama and The Pile (provenance).

## WREN2 STANDING DIRECTIVES (user, 2026-08-21) — outlive the current run

wren2 is the flagship grown function-preserving from wren1_3@3000 to the `400m` shape
(24L/1280/5120/20h, head_dim 64 preserved). These constraints bind every future mix, SFT,
sleep recipe and export for this model, not just the pretrain in flight:

- **Conversation and writing ONLY. Nothing technical, no code, no tool/agent data.** Code
  corpora are permanently excluded from its mix.
- Extra data only if VERY clean and commercially-clean licensed. NC and share-alike are
  rejected (precedent: the fluency-corpus entry in successes.md).
- Formal transcripts (hansard/scotus/chrg) stay at low weight — they trained looping once.
- Aggressive checkpointing so learning stays verifiable throughout.
- Finishing sequence: sleep-native recipe (validated E4), then PTQ int8 → parity gate → cardinal.
- The user pushes git before any launch; never launch until they confirm the push.

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


## 2026-09-03 09:05 EDT — DONE: sleep works on cardinal in one night (lab/2026-09-02-fast-consolidation-on-cardinal.md)

**State now**: nothing running on cardinal (dashboard pid 203827 on port 8001 serves the deployed 2026-09-02/03 tree:
trainer with `--freeze_blocks`, two-reading stop rule, freeze guard; sleep_optimizer/sleep_freeze_blocks; runner race fix;
404 route; planner budget). Cardinal models: wren1_3, wren2, wren_base. exp_fastsleep_0902 (22 GB) and exp_e2e_0902
DELETED 09:05 after their numbers landed in successes.md / failures.md / the lab entry. Cardinal sleep settings:
enabled false, models [] (enrollment is the user's), recipe kept: freeze 15, AdamW 3e-5, mixed_chat yardstick, stop 0.10,
ckpt 50, min = max = 200 steps. wren2 on mirach untouched throughout.
**Result**: tell-it-once loop through /sleep/now on facts told over the API: 69/100 closed-book from the served bin after
200 steps (2 h 43 min); hand-built drill curve 72/100 @200, ~90/100 @400-500; replayed mixed_chat +2.4% @200 / +0.8% @450
(32-draw), un-replayed veritate_chat +6.6% / +4.6%. Full tables in the lab entry.
**Session-bound things that died with this session**: monitors on cardinal's /tmp/arm3, /tmp/arm4, the e2e chain, the
VAL32 chain; cardinal /tmp holds arm_eval2.sh, e2e_sleep.py, launch_arm.py, bench_step.py and the /tmp/arm*/ results
(quiz JSONs, cap*.json) -- copy anything wanted before /tmp is cleared. memlog stopped.
**Enforcement added 2026-09-03 09:30 EDT**: `.claude/hooks/guard_bash.py` (wired for Bash in settings.json) blocks a
deletion under a models/ tree that does not check for a live trainer in the same command, and routes wren* deletions to
the user; `.claude/skills/veritate-research/SKILL.md` gained the four measured rules (build sleep on weak hardware this
way, read forgetting at 32 draws, un-enroll before quizzing, look before deleting).
**Next program (user, 09:20 EDT)**: unlimited context without re-reading prior turns -- the model remembers what was just
said from its carried state. Substrate exists: `/generate?fast=stream&state_id=<id>` (PyTorch backend) persists the carried
states per conversation and the next call sends only the new bytes. Unmeasured: whether the model USES it. Acceptance
test in ideas.md (turn 2 sent alone; baseline wren1_3@1000 1/6, wren1_5@700 3/6, goal 6/6); the real test is recall of a
turn that has been committed into the state (filler > seq between the turns), with a fresh-state leak control.
RUNNING 09:50 EDT 09-03 (session-bound): lab/2026-09-03-working-memory-from-carried-state.md pre-registered; probe
/tmp/t1_recall.py on cardinal (PyTorch backend, greedy, 6 items x 5 conditions A in-window / B pending / C committed /
D leak / E filler-only) on wren1_3@512 (filler 1,902 B > seq 1024) then wren2@70000 (filler 2,853 B > seq 2048);
results /tmp/t1/wren1_3.json, /tmp/t1/wren2.json, logs alongside. Falsifier in the lab entry (C >= 3/6 with A >= 4/6
= T1 exists; C = 0/6 on both with A >= 4/6 = the state needs a write rule first).
DONE 11:54 EDT: wren1_3@512 A/B/C/D/E = 0/0/0/0/0 of 6 (abstention SFT); wren2@70000 = 3/3/0/0/0 of 6: every item the
model answers in-window it answers byte-identically from the pending buffer and NOT from the committed state (0/3);
C/E replies drift to the filler's topic. Not a formal kill (A 3/6 < the 4/6 bar) -- see the lab entry's interpretation.
Blockers in order: (1) the abstention SFT refuses personal questions answerable from context; (2) the GLA decay state
does not write facts (IDEA 20 T2b). NEXT, pre-committed: in-context recall SFT for wren2 at the E4 dose (freeze 15,
3e-5), falsifier A < 5/6 after; if A >= 5/6 re-read C. Probe: scratchpad t1_recall.py (copy in cardinal /tmp), results
copied to the session scratchpad; cardinal stream_states cleaned; PyTorch backend idle-unloads on its own.
RUNNING 12:01 EDT 09-03: in-context recall SFT (lab/2026-09-03-in-context-recall-sft.md) on exp_recall_0903 (wren2@70000
fork, 28 blocks, 595M) 70000 -> 70200, freeze 21, AdamW 3e-5, recall_chat:0.5,mixed_chat:0.5 (new `--recall` mode of
tools/build_fact_chats.py, tests/corpus/test_build_fact_chats_recall.py, documented), state_carry chunks, mixed_chat
yardstick, stop 0.10, ckpt/eval 50. Optimizer paged to NVMe by the planner (disk-bound step). A session-bound monitor
launches the five-condition probe (`/tmp/t1_recall.py exp_recall_0903 70200 3`, glassblower item swapped for
lighthouse keeper) when the run ends; results /tmp/t1/exp_recall.json. If the session died: run that probe by hand
after the trainer exits, read the falsifier in the lab entry, delete exp_recall_0903 (check for a live trainer first).
16:13 EDT: the session was resumed (the earlier monitor died with the old process); trainer at ~step 70160, mixed_chat
0.4213 -> 0.4019 -> 0.3962 -> 0.3860 at 70000/50/100/150 (trainer draw, -8.4%).
16:20 EDT, USER SAID HOLD OFF: probe chain STOPPED, nothing launches on its own. The trainer finishes step 70200 by itself
(~17:15 EDT). To resume: confirm no trainer is running, then on cardinal
`cd ~/Veritate && python3 /tmp/t1_recall.py exp_recall_0903 70200 3 /tmp/t1/exp_recall.json` (~1 h), read the falsifier
in lab/2026-09-03-in-context-recall-sft.md, record, delete exp_recall_0903.
2026-09-05 01:25 EDT, USER: "5 is what I really want. Figure it out." Hold lifted for the working-memory goal. Program in
lab/2026-09-05-working-memory-program.md (three rungs, falsifiers pre-committed). Probe of exp_recall_0903@70200 RUNNING
since 01:23 (session-bound monitor; results /tmp/t1/exp_recall.json). Corpus `recall_far` built on cardinal
(`build_fact_chats --recall --gap-bytes 2200`, 2,598,944 B train, no probe words). Rung 1 launches when the probe lands:
parent = exp_recall_0903@70200 if A >= 5/6 else wren2@70000; fork as exp_wm_0905; `launch_arm.py 3e-5 <start+200>
--model exp_wm_0905 --corpus recall_far:0.35,recall_chat:0.15,mixed_chat:0.5 --extra '{"freeze_blocks": 14,
"state_carry": "chunks", "val_bin": "mixed_chat", "stop_on_val_rise": 0.10, "ckpt_every": 50, "eval_every": 50}' --go`;
then the five-condition probe on its final step. Working-memory number = condition C (committed state).
**Open for the user**: (1) whether extract_facts should take facts from assistant turns (see the incident: a wrong closed-
book answer became next night's drill); (2) which cardinal model to enroll for sleep, and the dose clamp (min/max steps);
(3) git -- everything is uncommitted; (4) the E4 levers left in ideas.md (compact fact forms, fuse, replay share).

### the day as it ran (kept for the record)

**Running on cardinal-01** (reverse tunnel `ssh -p 2222 cardinal-01@127.0.0.1`, dashboard port 8001,
restarted 16:48 EDT pid 203827 on the fully deployed 2026-09-02 tree (deploy2.tgz + deploy3.tgz: trainer incl. freeze_blocks, sleep, runner, settings, routes, planner, index.js); cardinal git HEAD is still 73632aa
and the deployed files sit there untracked): ARM 4 RUNNING since 18:53 EDT (the overnight; trainer child of dashboard pid 203827, `pgrep -f veritate_trainer`):
freeze_blocks 15 (66M of 270M train), AdamW 3e-5, 500 steps, ckpt/eval 50, val_bin mixed_chat, stop 0.10 (two readings),
49 s/step, ETA ~01:45 EDT 2026-09-03; quiz loop into /tmp/arm4/, val_eval both corpora after. Step 0 mixed_chat 0.590527;
STEP 50 (19:36 EDT): quiz fwd 4/50 rev 4/50, mixed_chat 0.571804 = -3.2% (below start), train loss 0.28.
STEP 100 (20:20 EDT): quiz fwd 15/50 rev 14/50 (29/100), mixed_chat 0.572572 = -3.0%. Knee of the curve.
STEP 150 (21:03 EDT): quiz fwd 26/50 rev 23/50 (49/100), mixed_chat 0.554772 = -6.1% (falling).
STEP 200 (21:46 EDT): quiz fwd 35/50 rev 37/50 (72/100), mixed_chat 0.556883 = -5.7%. FALSIFIER CLEARED: nightly
recipe on cardinal = freeze_blocks 15, AdamW 3e-5, 200 steps (2.7 h). Set on cardinal's settings after the run.
STEP 250 (22:30 EDT): mixed_chat 0.600150 = +1.6% (jumped +7.8% from step 200), quiz 35/36 (71/100, flat); e2e re-armed
at 200 steps.
STEP 300 (23:18 EDT): quiz 36/43 (79/100), mixed_chat 0.552857 = -6.4%: step 250 was a wobble (4-draw reading swings
+-7% between checkpoints; two-reading rule vindicated again).
STEP 350 (00:00 EDT): quiz 40/46 (86/100), mixed_chat 0.565726 = -4.2%.
STEP 400 (00:45 EDT): quiz 42/45 (87/100), mixed_chat 0.567562 = -3.9%.
STEP 450 (01:28 EDT): quiz 44/47 (91/100), mixed_chat 0.565807 = -4.2% (E4's ceiling at 1/9 of the bytes).
STEP 500 (02:12 EDT) DONE: quiz 41/46 (87/100), mixed_chat 0.565822 = -4.2% on the trainer's draw. Plateau 87-91 from
step 400. POST-RUN val_eval 8x4 (/tmp/arm4/cap_*.json) DISAGREES: mixed_chat +1..+4.6% above start at every checkpoint
(+3.4% at 500), veritate_chat +16.5% at 50 falling to +4.8% at 500. The mixed_chat reading moves ~7% with the random
windows, so the forgetting call is UNRESOLVED at the +2% level; a 32-draw val_eval on steps 0/200/300/500 (both corpora)
is the number of record and runs after the e2e (never beside a trainer). Recall numbers stand (engine quiz, deterministic).
E2E SLEEP RUNNING since 02:40 EDT 09-03 on exp_e2e_0902 through /sleep/now: 51 own exchanges -> 110,606 bytes of
experience_fact_sft, 200 steps, freeze 15, 3e-5, ETA ~05:30 EDT; the chain then prints the history event
(served/held, val_first/val_last) and quizzes the served bin into /tmp/e2e_quiz_after.json. A second chain runs the
32-draw val_eval on arm 4's steps 200/300/450/500 (both corpora, /tmp/arm4/cap32_*.json) once the box is idle.
E2E DONE 05:27 EDT: served=true held=false val_first 0.584803 val_last 0.521239; quiz through the served bin fwd 38/50
rev 31/50 (69/100). THE TELL-IT-ONCE LOOP WORKS ON CARDINAL IN ONE NIGHT. VAL32 chain now running (~2 h).
VAL32 mixed_chat (06:20 EDT): +2.36% @200, +2.64% @300, +0.83% @450, +3.52% @500 = at the +2% line (+-1%); the trainer's
-4% rows were the outlier. veritate_chat VAL32 (09:02): +6.55/+7.81/+4.57/+4.91% at 200/300/450/500.
INCIDENT 05:34-08:25 EDT: the e2e `arm` left sleep enabled; the 100 quiz exchanges counted as new own-experience and the
watcher auto-launched a SECOND sleep on exp_e2e_0902 (151 exchanges, 126 KB drill incl. facts extracted from the model's
own quiz answers, 200->400 steps). Ran unnoticed 2 h 50 min beside the VAL32 eval. I deleted the model dir at 08:24
without checking for a live trainer (process error), then stopped the run via /trainers/stop (failed, -15) and removed
the recreated dir. The 69/100 e2e quiz predates it and stands. OPEN DESIGN QUESTION FOR THE USER: should extract_facts
take facts from assistant turns at all? A wrong closed-book answer becomes next night's drill. Cardinal settings now:
sleep_enabled false, sleep_models [], recipe kept (freeze 15, 3e-5, adamw, mixed_chat, stop 0.10, ckpt 50, min=max 200).
CHAINED (session-bound monitor, 20:26 EDT): when /tmp/arm4/progress.log reads 'all done', `/tmp/e2e_sleep.py
exp_e2e_0902 arm 3e-5 200 15` sets the sleep settings (adamw, freeze 15, ckpt 50, yardstick mixed_chat, stop 0.10,
tolerance 0.02) and POSTs /sleep/now; the monitor polls /sleep until the model is awake again, prints the history
event (served/held, val_first/val_last) and quizzes the served bin closed-book into /tmp/e2e_quiz_after.json.
exp_e2e_0902 has 51 experience records (its own), step_0.pt only, pre-sleep bin. If the session died before
this ran: run those two commands by hand after arm 4's val_eval finishes, never while a trainer runs.
AFTER the e2e: its `arm` phase leaves cardinal's settings at sleep_enabled true, sleep_models [exp_e2e_0902],
min=max=200 steps, freeze 15, lr 3e-5, yardstick mixed_chat, stop 0.10. Keep freeze/lr/yardstick/stop as the box's
recipe; re-point sleep_models to whatever the box serves (user's call) and delete exp_e2e_0902 + exp_fastsleep_0902
once their numbers are in the ledgers. Freeze guard (count >= blocks refused) deployed to cardinal 21:55 EDT as a file. ARM 3 DONE 18:32 EDT
(60/60): val_eval 8x4 veritate_chat +12.1/+2.3/+5.8%, mixed_chat +14.9/+5.5/+5.2% at 20/40/60, quiz 0-0, 2-0, 3-3 of 50
(rising; mixed_chat sits on a ~+5% floor at full-param 1e-5, outside E4's +2%). Morning plan: fuse the best arm-4
checkpoint toward step 0 (alpha 0.5, 0.7; `tools/fuse_checkpoints.py`), re-quiz + re-val each. Model
`exp_fastsleep_0902` (= wren1_3/step_0 = wren1_0@1250, hardlinked), ARM 1 (AdamW 1e-4) FALSIFIED at step 20: veritate_chat +65.1% on identical windows, recall 0/50 fwd
1/50 rev; stopped, step_20 deleted. ARM 2 (AdamW 3e-5) FALSIFIED at step 20 (16:28 EDT): val 1.113589 vs 1.006792 = +10.6%, recall 1/50 fwd 1/50 rev,
replies fact-shaped with wrong bindings; stopped, step_20 deleted, results in /tmp/arm2/. ARM 3 = AdamW 1e-5, 60 steps, full
parameters, complete: 6/100 at 1.7 MB of drill (E4 needed 59 MB for 12/100); val_eval on both corpora for 20/40/60 in
/tmp/arm3/cap_*.json, then its checkpoints are deleted by the arm-4 chain. Arm 3 step 20: val +12.0% (1.127872), quiz 0/50 fwd 0/50 rev; step 40: val +1.9% (1.025540) -- the rise is a
TRANSIENT (templates fit, then recovery under replay), so the stop rule at 0.02 read at step 20 would falsely stop a
cardinal-sized dose. FIXED 18:25 EDT: `val_rose_past_start` now needs TWO consecutive readings above the start
line (trainer + tests/training/test_consolidation_stop.py + documentation.md), deployed to cardinal as a file (the
dashboard was not restarted; the trainer is spawned fresh per run, so arm 4 already runs the fixed rule). Arm 4 runs
with stop_on_val_rise 0.10 and the +2% call comes from post-run val_eval.
ARM 4 re-registered as the overnight (lab entry): `--freeze_blocks 15`, AdamW 3e-5, 500 steps, ckpt/eval 50, val_bin
mixed_chat with stop_on_val_rise 0.10, launched after arm 3's val_eval with
`launch_arm.py 3e-5 500 --extra '{"freeze_blocks": 15, "val_bin": "mixed_chat", "stop_on_val_rise": 0.10, "ckpt_every": 50, "eval_every": 50}' --go`,
quiz loop `arm_eval2.sh exp_fastsleep_0902 /tmp/arm4 "0 50 100 150 200 250 300 350 400 450 500"`. ETA ~01:30 EDT 09-03.
Freeze benchmark (idle box, AdamW, batch 7, act-ckpt on): 94.5 s full; blocks-only freeze 10 = 83.3 s, 15 = 78.5 s (pos_emb
still trainable, so backward walked every block for its input grad); with BOTH embeddings frozen freeze 10 = 58.6 s (-38%,
134M of 270M params train); freeze 15 = 47.5 s (-50%, 66M train, 4.8 GB RSS), freeze 15 + act-ckpt off = 40.0 s (-58%, 6.3 GB); act-ckpt off +
blocks-only 10 = 62.5 s at 14.5 GB RSS. Full tables in the lab entry and documentation.md `freeze_blocks`. SHIPPED: `--freeze_blocks N` trainer flag (embeddings + blocks[:N] frozen, optimizer sees only
trainable params), `sleep_freeze_blocks` setting (default 0) through sleep.launch_args, Training-form advanced field,
tests/training/test_freeze_blocks.py; deployed to cardinal 16:48 EDT.
Corpus finding (measured on the builder): fact_sft for 50 facts = 1000 exchanges, 500 unique; assistant bytes are 28%
of bytes and subject/object mentions 13%, so ~41 of cardinal's 313 B/s are binding signal (arm 2's train loss 0.36
is template fit, not binding). Next levers after arm 3: freezing the lower blocks (bench pending), compact QA forms
in build_fact_sft (2-3x binding share per byte). Log
`~/Veritate/.plugin_run.log`. Session-bound: `/tmp/arm_eval2.sh` (exports each checkpoint to the model's bin, drops the C
subprocess, quizzes closed-book over /v1/chat/completions into `/tmp/arm1/quiz_<step>.json`, then
val_eval on both chat corpora into `/tmp/arm1/cap_*.json`; the PyTorch quiz was killed for doubling
the trainer's step time), `/tmp/e2e_sleep.py` (tell/arm/status/quiz phases), `/tmp/bench_step.py`
(has a freeze-lower-blocks arm not yet run), `/tmp/launch_arm.py` (`--corpus`, `--model`). If the session died: read `/tmp/arm1/`, `train.csv` val
rows, apply the falsifier in the lab entry, then DELETE `models/exp_fastsleep_0902` (rule 40).

**Pre-committed falsifier**: fwd+rev >= 60/100 at some step <= 100 with veritate_chat val <= +2% of the
step-0 reading = nightly recipe found; every step < 40/100 or val > +2% = lr cannot replace samples.
Adaptive: val > +2% before fwd >= 30/50 -> arm 2 at 3e-5; fwd < 10/50 at 100 with val flat -> 3e-4.

**Measured today (idle cardinal, batch 7)**: AdamW 92 s/step vs Muon 108 s (opt step 1.2 vs 20.3 s);
activation checkpointing off -12%/sample, fits at batch 7. Forward+backward is 87-91 s of the step.

**Decided**: sleep resumes under AdamW by default (`sleep_optimizer`, fresh moments, restore skipped
and logged); the consolidation stop rule measures against the run's STARTING weights (armed resume
scores them before step 1) instead of the run's best, which halted wren1_12 while it was improving.
Both are deployed to cardinal (16:42 EDT) and in the working tree with tests. Then the end-to-end product test: fork wren1_0@1250 as `exp_e2e_0902` with
its bin, tell it the 50 E4 facts through /generate, enroll it, `/sleep/now`, quiz closed-book after
publish. wren2 on mirach untouched (pid 83227, step ~70k).

**Cardinal fleet changed under this session (other agent, ~15:26 EDT)**: `wren1_0` and `wren1_0_int8` are gone,
`wren2` (a checkpoint of the flagship) is installed, `warm_models` is `["wren_base"]`. `exp_e2e_0902` =
wren1_0@1250 (hardlink of exp_fastsleep_0902/step_0) with its own fp16 bin, already told the 50 E4 facts
(51 experience records, extract_facts 50/50) and waiting for arm 1's recipe before `/sleep/now`.

**Route fix (tested, deployed to cardinal 16:42 EDT, 404 verified live)**: `/v1/chat/completions` and `/v1/chat/mri` return 404
`model_not_found` for a model that is neither `cloud`, a teacher id, nor local; before, it fell through to the
public endpoint and 50 fact statements went off-box at 15:41 EDT (nothing recorded or trained).

**Uncommitted repo changes**: veritate_trainer.py (val_rose_past_start + starting-weights reference),
sleep.py + settings.py (sleep_optimizer; publish gate anchored on the starting-weights row), hardware.py +
mem_planner.py (budget = min(0.85 x total, available) so a co-tenant box plans against what is free),
trainer_runner.py (start refused while a stopped child is still exiting; a late exit never stamps a newer
run; is_running counts a live child), hybrid_routes.py (unknown model -> 404, never the cloud), tests (test_consolidation_stop rewritten, sleep_controller
+1), ideas.md pruned of finished ideas, successes.md (IDEA 22 entry), documentation.md (sleep_state_carry,
sleep_optimizer, val_eval/fuse, stop rule, optimizer-switch resume). Plus another session's
checkpoints.py/_brain.py/test_checkpoint_step_resolve.py, untouched. User runs git.

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

## CARDINAL 2026-08-24 — IDEA 23 resolved, sleep completes steps on the box

Sleep now completes steps on cardinal. Before today every run recorded `steps_gained: 0`; the first
successful cycle recorded `end_step: 2, steps_gained: 2, finals: [2]` with loss 0.6030 -> 0.5868 and
`pending_exchanges` reset. Three causes, all shipped fixed and tested:

1. **Muon orthogonalizes in bf16.** `torch.optim.Muon` casts the momentum update to bf16
   (`torch/optim/_muon.py:55`) and the vendored copy mirrored it. On a CPU with no bf16 acceleration
   that `addmm` has no fast path: 94.5 s (1024x1024) and 175.3 s (4096x1024) per weight at ~107% CPU
   against 0.251 s / 0.775 s at ~700% in fp32; 99.1% of profiled time in `aten::addmm` while
   `aten::mm` on the same shape is 0.34 s. 112 2-D weights, so one optimizer step was hours. Now
   50.8 s at 700%. Newton-Schulz takes its dtype from `hardware.bf16_supported`; a device that cannot
   accelerate bf16 gets the vendored Muon. **The dtype is an instance attribute, never a param-group
   default** — in `defaults` it serializes into `state_dict()` and resuming a pre-change checkpoint
   raised KeyError on the first step.
2. **Sleep inherited `batch_size: 48`** from the pretrain recipe and drew 196 kB a step from a
   17.8 kB log. Now `min(recipe_batch, train_bytes // draw_window)` -> batch 4, step 166 s.
   `eval_iters` is fitted the same way (was spending 475 s remeasuring 5.4 kB sixty-four times).
3. **`save()` ran the full checkpoint dump suite** on checkpoints sleep deletes at finalize. It held
   seven cores for >10 min between two steps. `SLEEP_OVERRIDES` now sets `hooks: off`.

Also: `sleep_idle_min` default 20 -> 2 min (preemption is what the long gate was protecting), and
the trainer logs the dtype it computes in rather than the one requested — the old
`precision: bf16` line on a CPU run sent this investigation after the wrong dtype for hours.

Forward/backward was never implicated: 696% CPU, 69.5% of self time in `aten::mm`. IDEA 23's other
suspects (8-bit Adam, activation-checkpoint recompute) are falsified with it.

4. **The step is also bounded by what the box measured.** The data bound caps at the model's own
   recipe batch, so a box that talks enough to fill a pretrain batch would go straight back to
   27-minute steps. `finalize()` records `step_s` and `step_batch` into that model's sleep state
   (box-local: a model dir travels between machines and its `train.csv` rows carry the throughput of
   whichever one wrote them) and the next launch scales it to `sleep_step_seconds` (default 300).
   First sleep on a box has no measurement and takes the data bound alone.

**Verified end to end on cardinal 2026-08-24:** a 4-step run completed naturally (`done.`, steps
1-6 at 164-166 s each, ~100 tok/s, 616-736% CPU), `finalize` recorded `steps_gained: 4`, prune kept
`finals: [2, 6]` and deleted the intermediate `step_4.pt` while leaving the non-sleep `step_0.pt`
alone. The watcher then launched the next sleep BY ITSELF (countdown 120 s -> 0 after ten served
exchanges, step 6 -> 506, batch 4 off a 21 kB log). Preemption re-verified against the new code
mid-step: `RNl -> TNl` on request arrival, reply in 1.25 s, `-> RNl` after the quiet window.

**Publish verified on cardinal 2026-08-24:** a 2-step run finished (`done.`), finalize pruned to
`finals: [6, 8]`, re-exported `veritate.bin` (md5 `35b040bd` -> `6f4fc7fb`, same 541,020,992 B so the
fp16 dtype was preserved), left no stray `.part`, recorded `served: true`, and the reloaded engine
answered the next request in 1.69 s off the new weights. `step_s: 181.0` / `step_batch: 4` are now in
sleep state, so the next launch takes the box bound and a time-bounded checkpoint interval
(`min(25, 1800/181) = 9` steps ~= 27 min).

**Cardinal state at handoff:** sleep ENABLED, `sleep_models: [wren1_3]`, `sleep_idle_min 2`, real
dose (50 / 500 / 25), `sleep_step_seconds 300`, `sleep_ckpt_seconds 1800`, `engine_threads 8`, fp16 bin served,
`veritate.bin.int8.bak` retained for the quality eval that still gates shipping int8. Sleep is armed and idle at step 8; the watcher launches on its
own once 8 new exchanges land and the box has been quiet 2 minutes.
All nine changed files are byte-identical to local `dev`; the checkout is ahead of origin until the
user pushes. Deploy: `scp` the changed files or `git fetch origin dev && git reset --hard
origin/dev`, then `bash ~/veritate/dash_watchdog.sh`. Watch loops on that box must not match their
own command line: use `pgrep -f "[v]eritate_venv/bin/python -u"`.

## CARDINAL 2026-08-24 (evening) — the five follow-ups, and one hard negative

**Sleep was making the model worse, and now there is a number for it.** wren1_3 on a FIXED corpus it
never trained on (`mixed_chat_val.bin`, trainer's own `evaluate()`, 8 iters, seed 1234): step 0
**0.572240**, step 10 **0.574835** (+0.45%), step 12 **0.575393** (+0.55%). Monotone with dose. Its
own experience val drifted the same way (+1.8% over five runs), so the moving and the fixed yardstick
agree. Cause: cardinal was running `sleep_corpus: experience:1.0` -- pure self-training, no rehearsal
-- because the box had NO base corpus on it at all, only the experience bins. failures.md carries the
kill-line. **This kills the no-rehearsal configuration, not the method:** 12 steps at lr 5e-6 is a
small dose.

Fixed, all live on the box:
1. `mixed_chat_{train,val}.bin` (212 MB) shipped to cardinal; `sleep_corpus` set to the platform
   default `experience:0.75,mixed_chat:0.25`, so consolidation rehearses instead of eating itself.
2. Validation is pinned to a fixed yardstick. New trainer flag `val_bin` overrides
   `resolve_val_path`'s heaviest-corpus rule; sleep passes `sleep_yardstick` (default `mixed_chat`).
   Without it every run scored a freshly rebuilt bin and no two runs were comparable.
3. The gate's baseline is now the BEST val the model ever recorded (`val_best`, raised only by runs
   that actually published), not the previous run. Run to run the drift was +0.36%/run against a 2%
   tolerance -- it would never have fired while the total walked up indefinitely. Against the
   high-water mark the tolerance bounds total drift instead.
4. IDEA 22 shipped: both decode step classes are timed and get their own worker count. A boundary
   step measures **5.9x** a plain one on cardinal (23.144 vs 3.939 ms/byte at 8 workers), confirming
   the bimodal finding -- but both classes pick 8 there, so the switch is inert and the falsifier
   (>5% on two boxes) is NOT met. Kept anyway for the best-so-far rung fix, which is what made the
   pick stable. `engine_threads` set back to 0 so calibration governs.
5. Turbo (`no_turbo=1`) is STILL the largest single lever left and needs one root command on the box.

**int8 is still parked and should stay parked.** The 1.72x is real, but it measurably changes output
(1 of 5 greedy replies matched fp16) and there is no engine-side quality eval to clear it -- the
harness built today loads a `.pt` and cannot score a `.bin`. With the clock lift already worth 2.1x
and quality the actual open problem, spending output fidelity on speed is the wrong trade today.

**Do not re-derive:** the two failing engine golden tests on cardinal
(`test_canonical_greedy_matches_golden`, `test_v9_greedy_matches_golden`) are the documented
pre-existing dense-on-AVX2 limitation (handoff below, 2026-08-20), not a regression. The hybrid path
is fully green there.

## OVERNIGHT RUN IN FLIGHT — read this first tomorrow (launched 2026-08-24 ~14:40 EDT)

wren1_3 is consolidating on cardinal WITH rehearsal, to test whether that reverses the degradation
measured today. Launched from step 12, 500 steps, batch 7, `ckpt_every` 22, ~110 s a step, so expect
roughly 300-400 steps by morning rather than a finish. It preempts for every served request and can
be woken at any time (`POST /sleep/now` / `/sleep/wake`), keeping everything to the last checkpoint.

    corpus mix:  experience:0.750, mixed_chat:0.250     (rehearsal ON, was experience:1.0)
    corpus val:  data/corpus/veritate_chat_val.bin      (held out; NOT in the sleep mix)

**The comparison to make.** Baselines already measured on `veritate_chat_val.bin`, 8 iters, seed
1234, batch 4:

    step  0   0.996948          pre-sleep, the model's best
    step 12   1.011243  +1.43%  after 12 steps of NO-rehearsal sleep — the control

Re-run the same eval on whatever checkpoints exist and compare against 1.011243. Below it means
rehearsal is repairing the damage; at or above it means rehearsal is not enough at this dose and the
next lever is the mix weight or the learning rate, not the mechanism.

    ssh -p 2222 cardinal-01@127.0.0.1
    cd ~/Veritate && OMP_NUM_THREADS=8 EM=wren1_3 ES=0,12,<new steps> EI=8 EB=4 \
      EV=data/corpus/veritate_chat_val.bin .veritate_venv/bin/python ~/diag/sleep_eval.py

`~/diag/sleep_eval.py` is a session diagnostic, deliberately OUTSIDE the repo tree (rule 34). It
builds the model from `config.json`, loads a checkpoint through the trainer's own
`load_resume_state`, and scores it with the trainer's `evaluate()`/`make_data_loader`, so the number
is the same quantity `train.csv` records. Roughly 100 s a checkpoint at 2.0 GHz.

**Caveat on what this can and cannot show.** It answers "does consolidation-with-rehearsal stop the
model degrading". It does NOT show that consolidating experience ADDS anything, because a fall in
val could come from the rehearsal alone. Separating those needs a probe on the experience
distribution specifically, which does not exist yet.

## CARDINAL 2026-08-24 (later) — the 800 MHz clamp is LIFTED

The user changed BIOS settings for performance and the clamp went away. `cpuinfo_max_freq` and
`base_frequency` 800000 -> **2000000** (the chip's rated base), `turbo_pct` 98 -> **65** (a sane
value instead of the one that said the whole range sat behind a disabled switch). Measured
immediately after, same model, same engine, same batch:

| | 800 MHz | 2.0 GHz | |
|---|---|---|---|
| decode, 256 B greedy reply | 18.3 ms/byte | **8.7 ms/byte** | 2.10x |
| sleep step (batch 4) | 166 s | **72-87 s** | ~2.1x |
| training throughput | 99 tok/s | **227 tok/s** | 2.29x |

`no_turbo` is still 1, so turbo is STILL off and there is more available. The firmware no longer
reports the range as locked, so `sudo sh -c 'echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo'`
may now succeed where it returned EIO before. Untested.

Two bugs this exposed, both fixed:
- `sleep_val_tolerance` was never added to `settings.DEFAULTS` (the edit silently matched nothing),
  so `finalize()` would have raised KeyError on the first real publish. Every test that exercises
  the gate builds its own cfg dict, so nothing caught it. Now pinned by a test that reads the
  settings keys the controller uses out of its own source and asserts DEFAULTS carries all of them.
- The recipe batch cap RATCHETED DOWN. The trainer stamps its launch args into `config.json` at
  every checkpoint, so sleep's reduced batch became the model's recipe batch: wren1_3's 48 had
  become 4, and the unclamped box would have stayed pinned there forever. The cap is now the
  largest batch the recipe has ever declared, captured into sleep state at launch. wren1_3's
  `config.json` was repaired 4 -> 48 by hand on cardinal.

Self-tuning verified after the repair: `step_s` 181.0 -> **79.4**, next batch 4 -> **7**
(now data-limited, not box-limited), next `ckpt_every` 9 -> **22**.

**The 800 MHz clamp, re-checked earlier 2026-08-24 (superseded by the section above).** Root-caused 2026-07-13 (worklog) as the Dell
power-adapter/BIOS clamp. New evidence today narrows it: `intel_pstate` logs "Turbo disabled by BIOS
or unavailable on processor" when the BIOS actually disables turbo, and that line is **absent** from
this box's kernel log (only "HWP enabled" + "Disabling energy efficiency optimization"). RAPL is
healthy (PL1 35 W / PL2 60 W, correct for a 35 W T-series part), package 27-43 C, `num_pstates 36`,
`turbo_pct 98` — so the hardware exposes the full range and essentially all of it sits behind a turbo
switch that reads 1. `no_turbo` is a normal root-writable file; an unprivileged write gets a plain
permission error, not the driver's -EPERM. sudo needs a password, so this was NOT tested.

    sudo sh -c 'echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo'
    cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq   # 800000 -> higher if it took
    # revert: sudo sh -c 'echo 1 > .../no_turbo'

If `cpuinfo_max_freq` rises, the clamp was turbo-off and everything on this box gets 2-3x (PL1 35 W
still caps sustained all-core clocks, so expect ~1.6-2.2 GHz under load, not the 3.8 GHz turbo peak).
If the write is refused or the frequency does not move, it is the power-adapter clamp and no software
change reaches it: check the brick and the BIOS Performance page. Either way it dwarfs every software
lever left (int8 is 1.72x and gated on a quality eval).

**Sleep now publishes (shipped 2026-08-24 on user instruction).** `finalize()` re-exports the newest
checkpoint over the model's `veritate.bin` and respawns the engine holding the pre-sleep weights.
Every bin writer in `export.py` goes through `_atomic_bin` (sibling temp, fsync, `os.replace`) —
they used to `open(path, "wb")`, truncating the served artifact before the first tensor landed. Only
a model that already serves a bin is published; the dtype follows the bin in place; a failed publish
leaves the previous weights serving and does not fail the sleep.

**STANDING RISK, accepted by the user, not mitigated in code:** there is no quality gate on the
promotion. A bad consolidation degrades the served model, and that model then generates the next
round of its own training data. Nothing currently detects that. The sleep run's own val loss is
recorded per run in `train.csv` and is the obvious signal to gate on; wren1_3's has been flat-to-up
(1.2995 -> 1.3151 -> 1.3127) over the runs so far, which is noise at 6 steps and lr 5e-6 but is not
evidence of improvement either. Watch it.

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
