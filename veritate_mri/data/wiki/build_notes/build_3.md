---
title: "Build 3: hook_spec contract, mega in the Models tab"
date: 2026-05-05
tags: [build, hooks, mri, mega]
summary: Trainers now declare their own hook trace points via model.hook_spec(). The mega 1B MoE trainer is wired in, so its checkpoints show up in the Models tab with the standard probe / lens / classroom / generation dumps.
---

## versions

- build: 3
- engine: v2.0.0
- mri: v0.1.1
- format: v0.1.0
- plugins: v0.1.1

## what changed

The Models tab used to silently hide any model that hadn't produced per-step hook artifacts. The MEGA 1B trainer fell into that bucket because its state dict isn't shaped like a vanilla Veritate. Build 3 inverts the contract: the model tells the dumper where to look, instead of the dumper hardcoding paths.

- **hook_spec() contract.** Every model now exposes `model.hook_spec()`, which returns an object the dump suite walks as if it were a canonical Veritate. Vanilla models return `self`. Non-canonical models (MoE, future workspace / HDC sidecars) return a thin adapter that proxies onto their internal trace points. Spec lives in [documentation/hooks/contract.md](../../documentation/hooks/contract.md#hook_spec-contract).
- **MEGA gets the standard hook suite.** `multimind_mega` now goes through `save.save()` like every other trainer. Per-step probe, lens, classroom, grades, concepts, surprise, quant_kl, and generation artifacts are written under `models/<name>/hooks/step_<N>/`. Mid-training generation samples are now visible in the Models tab.
- **MoE FFN trace.** Each MoEFFN layer captures a routing-weighted `(B, T, ffn)` per-token activation under `last_ffn_act` so the per-layer "top neurons" probe sees a vanilla-shaped tensor. The down-projection weight surfaced to the dumper is expert 0's; per-expert traces are out of scope for now.
- **Models tab no longer filters hookless models.** `/timelines` returns every model with a config + at least one `.pt` checkpoint, even if no hooks have been written yet. The dropdown labels hookless entries `no hooks yet`, and the panel renders an inline warning explaining that the per-step dumps were not written. Useful when training is mid-run before its first save_checkpoint, or when a custom trainer has not yet been ported to `hook_spec()`.
- **Mega inline comment corrected.** The previous comment claimed save.save() could not run on mega and that the trainer saved manually. With hook_spec() the correct path is save.save(); the comment now reflects that, with a pointer to the multi-expert caveat.
- **Ask AI buttons.** The dashboard can now call out to a configured chat endpoint to explain training metrics, the train health verdict, and the loss curve. Set the endpoint and key under settings, then use the inline "ask ai" buttons on those panels to get a short, numeric read on the data.

## what you need to do

Nothing for vanilla, M1, or M3 trainers — `hook_spec()` returns `self` for the canonical model, so behavior is unchanged.

For MEGA models trained on build 2:

1. Old MEGA `.pt` checkpoints continue to load. They have no `hooks/` directory because save.save() was bypassed — the Models tab will show them with the inline "no hooks yet" warning until the next training run hits a save.
2. Resume training (or start a new run) on build 3 to populate the hook artifacts.

For custom trainers built outside the canonical model:

1. Implement a `hook_spec()` method on your model. Return `self` if you mirror the canonical layout. Otherwise return a small `nn.Module` adapter that exposes the canonical attributes (`.layers`, `.blocks[L].n1`, `.blocks[L].attn`, `.blocks[L].ff.up`, `.blocks[L].ff.down`, `.tok_emb`, `.lm_head`, `.n_out`, plus a 2-tuple-returning `forward`). See `plugins/multimind_mega/mega_model.py` for a worked MoE example.
2. Until you do, your trainer's checkpoints will show in the Models tab with the inline warning, and the per-step dump suite will silently skip operations it cannot run on your shape.

## known issues

- MEGA hook traces describe expert 0's down-projection only. Multi-expert direct-logit-attribution will need a contract extension (probably `blocks[L].ff.experts[]`) and a UI panel to slice by expert. Not in this build.
- The hook view's `_BlockView.forward` calls the real mega block forward then re-fires `ff.up` purely to trigger the dumper's hook. Cheap but adds a no-op forward per block per probe call. Negligible at checkpoint time, not on the hot training path.
- Carryover from build 2: MEGA still trains in pure PyTorch (no engine kernels for ternary + MoE yet); ARM64 still not supported.
