# chat200m plan (200m-class hybrid chat model)

Successor plan to `chat_model_80m_plan.md`. Pretrain LAUNCHED 2026-07-08 (run `models/chat200m_200m/`, dashboard-launched). This doc carries the remaining phases and the pre-registered gates. Verdicts land in `successes.md` / `failures.md`; chronology in `worklog.md` 2026-07-08.

## Goal

A 200-300M-class hybrid chat model with real common knowledge, ms-class C-engine inference (int8 v13 measured 1.7-1.9 ms/byte p50 on M-class P-cores at this shape), identity-stable under sampling, context-fact extraction trained in from step 0, long-tail facts via RAG.

## Pretrain (running)

- Shape: veritate_200m manifest, trunk=hybrid = 270,510,336 params (2 local enc + 16 recurrent global + 2 local dec, h1024, ffn 4096, heads 16, seq 1024).
- Config: muon, bf16, batch 24, n_chunks 4, bptt_window 2, lr 3e-4 -> 3e-5 WSD (sqrt tail, decay_frac 0.2 so a 2B -> 6B extension resumes pre-decay), 20,400 steps ≈ 2.006B tokens, measured 14.2-14.3k tok/s ≈ ~2 days.
- Mix: fineweb_edu 0.37, openwebtext10g 0.365, chat_v1 0.05, chat_v2 0.04, chat_v3 0.03, py_code_v1 0.06, chat_recall_v1 0.04, grounded_v3 0.025, chat_identity_v1 0.02. Rationale: every 80M lesson dosed from step 0, nothing narrow saved for late phases (recall-SFT and identity-SFT kill entries, `failures.md` 2026-07-06 / 2026-07-08).
- state_carry=chunks NOT enabled (10M validation still pending). If validated before an extension resume, reconsider there.

## Gates (pre-registered)

1. **Pretrain sanity:** val trajectory vs 80M-scaled expectation at each 2k milestone (80M: 1.695 -> 0.942 over 30k at 16k tok/s; 200m val 1.141 @1500 is ahead). Dump battery = 14 artifact families incl. generation.json per checkpoint (count artifacts, do not trust a DUMP FAILED grep).
2. **Knowledge eval at 2B (the go/no-go for more tokens):** build a capitals/simple-facts battery BEFORE step 20,400. If <2x improvement over chat80m, STOP and re-evaluate before spending a 6B budget.
3. **Post-pretrain batteries, all four surfaces** (the 80M lesson: skills are surface-specific): (a) needle conversation-copy; (b) alien-fact context-block extraction (grounded_v3 val protocol, invented entities); (c) identity bare AND with persona line, sampled at temp 0.5 (not greedy-only); (d) chat register battery. Each gate scored sampled, not greedy.
4. **Phases after pretrain:** midtrain anneal (chat-heavier ~45%) -> SFT (chat + identity + grounded, small LR). At EVERY phase boundary: full battery re-run including needle >= 0.9 short-range, identity >= 95% with persona, no chat regression, tail-averaged val. NO late narrow phases; any single-skill top-up must keep all other batteries green or roll back.

## Serving

- v13 C engine: fp16 and int8 both support the shape (dtype=2 additive). Export via the platform quantize path; verify generation on the deployed surface (Generation tab + /hybrid/chat) with the standard battery before flipping capabilities.chat.
- /hybrid/chat serve stack (persona line, rep defaults, multi-marker stop, seq-budgeted RAG prompts) is model-agnostic and already live.

## Related

- 80M repair round (separate, queued behind this run on GPU): combined SFT from chat80m step 48000 — grounded_v3 ~25% + chat_identity_v1 ~15% + chat mix, ~2-3k steps at 1e-5, gates per `failures.md` 2026-07-08 retry condition (a).
