# GPU-machine handoff: train the largest model this box can hold

**You are Claude, running on Sam's NVIDIA GPU machine, in a checkout of the Veritate repo.** This
document is a self-contained brief. Sam wants you to (1) find the largest Veritate model this GPU can
train *from scratch*, (2) confirm the stack runs on CUDA, and (3) launch that training. The Mac is
already pretraining an 800M core; your job is to burst a *bigger* model onto real CUDA silicon.

Read this whole file first. Then read the repo's gating docs (below) before any code edit. Ask Sam
before anything destructive or irreversible; he runs all git himself.

---

## Why you exist (context from the design conversation, 2026-07-14)

Training throughput = `(peak FLOPs x MFU) / (FLOPs per token)`. The Mac (M3 Ultra) is **FLOP-bound
with no tensor cores** — its ceiling is low and unbuyable, so it runs a modest 800M. Your GPU has
tensor cores and real matmul throughput, so it can train a **larger** model far faster. That's the
entire point of moving here. The winning architecture recipe is already settled on the Mac (see
"the recipe" below) — you are not inventing; you are *scaling the proven stack up to your VRAM.*

---

## Read-first (repo gating contract — do this before editing anything)

- `CLAUDE.md` (repo root) — agent entry point, doc map, the trainer rules.
- `claude_preflight.md` (repo root) — behavior/code-style/tooling rules; wins on conflict.
- `developer_documentation/agents/coding_roe.md` — lean-code rules 100-128; gating for any edit.
- `developer_documentation/agents/agent_roe.md` — the seed rule (a >5% claim needs a 2nd seed).

**Hard guardrails (from CLAUDE.md — do not violate):**
- **`trainers/` is a synced checkout from a canonical repo.** Do NOT create new `trainers/<name>/`
  dirs. The size set is fixed: `veritate_10m, 80m, 200m, 400m, 800m, 1b3, 3b, 13b, 50b, 70b, 100b,
  120b, 160b, 200b, 250b, 350b, 500b, 700b, 1t`. **Pick the largest existing one that fits** — do not
  invent a size. Any trainer edit must be backwards-compatible and destined for the canonical repo;
  state that intent and quote the diff, don't slip it in.
- **Platform code** (`veritate_core/`, `trainers/common/vanilla_trainer.py`, root) is editable locally
  — that's where any CUDA-port fix goes.
- **Do not touch anything on the Mac.** Separate machine, separate run.
- **Sam runs all git.** No `git add`/`commit`/`mv`. Never ask about version bumps.

---

## The stack is CUDA-native (good news)

This is not a port — CUDA is the *primary* target of the training code:
- `trainers/common/vanilla_trainer.py`: `device = hardware.pick_device()` auto-selects CUDA;
  `device_type="cuda"` is the default; the optimizer uses `fused=(device=="cuda")`.
- 8-bit Adam via `bitsandbytes` is wired (`--use_8bit_adam`, CUDA-only lib) — a real VRAM lever here.
- The trainer prints a **memory plan** at startup: look for a `BENCH_RESULT {json}` line
  (`bench.plan_result(...)`) — it estimates whether a config fits. Use it to confirm sizing before a
  long run.

It *should* run out of the box. **Still smoke-test small first** (Step 2) — verify CUDA, no NaN,
sane throughput, checkpoints/dumps write — before committing to a big run.

---

## Step 1 — detect the hardware

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,count --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -c "import importlib.util as u; print('bitsandbytes', u.find_spec('bitsandbytes') is not None); print('flash_attn', u.find_spec('flash_attn') is not None)"
```

Record: GPU model, VRAM per card, card count, torch/CUDA version, whether bitsandbytes/flash-attn are
present. **Single-GPU full-from-scratch training is the assumption** unless Sam confirms multi-GPU
(the trainer is single-process; multi-GPU would need DDP wiring — ask before assuming it).

---

## Step 2 — smoke-test the stack on CUDA (de-risk before scaling)

Run the smallest trainer for a few hundred steps and confirm it's healthy on CUDA:

```bash
python -u trainers/veritate_10m/trainer.py --name cuda_smoke --corpus fineweb_edu:1.0 \
  --optimizer muon --trunk hybrid --state_rule gla --precision bf16 \
  --total_steps 300 --batch_size 16 --seq 1024 --n_chunks 4 --bptt_window 2 \
  --base_lr 6e-4 --lr_schedule wsd --warmup_steps 100 \
  --ckpt_every 300 --log_every 25 --eval_every 150 --eval_iters 8 --seed 0 --model_type language
```

Pass conditions: device prints as `cuda`; a `BENCH_RESULT` memory plan appears; train loss drops
(not NaN/flat); throughput is sane (should be *much* higher than the Mac's ~4350 tok/s at 800M — a
tensor-core GPU on a 10M should be very fast); a checkpoint + dumps write with zero `DUMP FAILED`. If
anything is MPS-specific and breaks, that's a platform-code fix in `veritate_core/`/`vanilla_trainer.py`
(allowed) — fix it, note it for Sam to mirror, re-smoke-test. **Do not scale up until the smoke test is green.**

---

## Step 3 — size the model to your VRAM

**Full from-scratch training memory ~= per-param state + activations.**

| Setup | Bytes/param (weights+grads+optimizer) |
|---|---|
| Adam, mixed precision | ~16 |
| 8-bit Adam / Muon (lighter opt state) | ~10 |

Activations add on top (scale with batch x seq x hidden x layers); **`--use_act_ckpt` shrinks them
~sqrt** at a compute cost. Rough single-GPU full-training ceilings (leave headroom, then verify with
the planner):

| VRAM | Comfortable | Ceiling (8-bit adam + act-ckpt + small batch) | Largest existing trainer to try |
|---|---|---|---|
| 24 GB | ~0.8–1B | ~1.5B | `veritate_800m` / `veritate_1b3` |
| 32 GB | ~1–1.5B | ~2–3B | `veritate_1b3` → `veritate_3b` (tight) |
| 48 GB | ~2–3B | ~4–5B | `veritate_3b` |
| 80 GB (A100/H100) | ~5–7B | ~10–13B (tight) | `veritate_3b` → `veritate_13b` (verify) |

**Procedure:** pick the largest existing trainer size your VRAM row allows → dry-run it and read the
`BENCH_RESULT` memory plan → if it fits with headroom, that's your target; if it OOMs, step down a
size or turn on the memory levers (Step 4) and re-check. Do **not** exceed the planner's fit estimate
on a long run — a mid-run OOM wastes hours.

> Token budget is the *other* real limit: a model wants very roughly ~20 tokens/param to be
> well-trained (a 3B ~ 60B tokens). Size `--total_steps` so `batch_size x seq x total_steps` lands in
> a sane range for the wall-clock you're willing to spend; it's fine to under-train for a first proof,
> like the Mac's 800M does. Tell Sam the token budget you chose and why.

---

## Step 4 — turn ON the CUDA speed/memory levers (unlike the Mac)

Rule 24e: test the free no-quality-loss levers on every launch. On CUDA they're far more potent than
on MPS — use them:

- **`torch.compile`** — ~33% free throughput on CUDA (flaky on MPS, so the Mac can't rely on it; you
  can). Confirm whether the trainer already enables it; if there's a flag/hook, turn it on and verify
  it doesn't break the run.
- **`--precision bf16`** (tensor cores) and TF32 matmul (`torch.backends.cuda.matmul.allow_tf32=True`).
- **`--use_8bit_adam`** — bitsandbytes; ~10 vs ~16 bytes/param, lets you fit a bigger model. (The Mac
  run used `--no-use_8bit_adam`; here, prefer it on when VRAM-bound.)
- **`--use_act_ckpt`** — trades compute for memory; turn on when VRAM-bound to fit a larger size.
- **`--n_chunks 4`** — the +68% amortization lever (already default in the recipe). Use gradient
  accumulation + the largest `--batch_size` that fits to keep the GPU saturated.

---

## Step 5 — the recipe and the launch command

**The proven winning stack (do not change these — settled on the Mac):**
`--optimizer muon --trunk hybrid --state_rule gla --precision bf16 --lr_schedule wsd`
(WSD decay: `--wsd_decay_frac 0.15 --wsd_decay_kind sqrt`; `--activation gelu`; `--grad_clip 1.0`;
`--label_smoothing 0.05`; `--weight_decay 0.1 --beta1 0.9 --beta2 0.95`).

**Corpus mix** (the 800M's 9-stem knowledge+chat+code+recall+identity+grounded blend — a good default;
`--corpus name:weight,...`, weights sum to 1.0). Verify these corpora exist on this box first
(`/train/discovery` or the data dir); drop any that are missing and renormalize:

```
fineweb_edu:0.37,openwebtext10g:0.365,chat_v1:0.05,chat_v2:0.04,chat_v3:0.03,py_code_v1:0.06,chat_recall_v1:0.04,grounded_v3:0.025,chat_identity_v1:0.02
```

**Launch template** (fill `<SIZE>` = chosen trainer, e.g. `1b3` / `3b`; `<STEPS>`, `<BATCH>` from your
sizing). Prefer launching through the **dashboard** if the platform runs on this box (start the server,
`POST /trainers/run`, confirm it appears) — that's the house style; the CLI below is the fallback:

```bash
python -u trainers/veritate_<SIZE>/trainer.py --name chat<SIZE> \
  --corpus fineweb_edu:0.37,openwebtext10g:0.365,chat_v1:0.05,chat_v2:0.04,chat_v3:0.03,py_code_v1:0.06,chat_recall_v1:0.04,grounded_v3:0.025,chat_identity_v1:0.02 \
  --description "<SIZE> CUDA burst pretrain: scaled hybrid recipe, GPU-machine handoff" \
  --optimizer muon --trunk hybrid --state_rule gla --state_carry off \
  --activation gelu --precision bf16 --version v1 \
  --use_act_ckpt --use_8bit_adam \
  --total_steps <STEPS> --batch_size <BATCH> --seq 1024 --n_chunks 4 --bptt_window 2 \
  --base_lr 3e-4 --min_lr 3e-5 --warmup_steps 1000 \
  --lr_schedule wsd --wsd_decay_frac 0.15 --wsd_decay_kind sqrt \
  --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --label_smoothing 0.05 --grad_clip 1.0 \
  --ckpt_every 1500 --log_every 25 --eval_every 500 --eval_iters 16 --seed 0 --model_type language
```

(The Mac's live 800M uses `--no-use_act_ckpt --no-use_8bit_adam` because it has the RAM to spare; you
likely want them ON to fit a bigger size. Flip them off only if the planner says you have headroom and
you want max speed.)

---

## Step 6 (ambitious, optional) — the largest *effective* model: `trunk=hybrid_moe`

If Sam wants the biggest model possible, MoE is the lever: it decouples capacity from throughput —
many small experts, only top-k active per token, so a large-**capacity** model trains at small-**active**
FLOPs. It's already built (`trunk=hybrid_moe`, `MoEFFN` in `veritate_core/model_patched.py`) but has
**never been scaled/measured** (it's IDEA 3 / T1 in `ideas.md`). Swap `--trunk hybrid` →
`--trunk hybrid_moe` (keep `state_carry` compatible — `hybrid_moe` is in `STATE_CARRY_TRUNKS`). Treat
it as experimental: **smoke-test it small first**, watch the MoE load-balancing aux loss, and tell Sam
this is the unproven-at-scale path so he can decide dense-largest (Step 5) vs MoE-largest (Step 6).

---

## Step 7 — report back to Sam

Send back, so we can compare Mac vs GPU and feed the throughput research (IDEA 3):
1. GPU model + VRAM + card count; torch/CUDA versions; bitsandbytes/flash-attn present?
2. Did the stack run on CUDA out of the box? Any platform-code fixes you made (quote diffs for Sam to
   mirror to canonical).
3. Chosen trainer size + why (the `BENCH_RESULT` fit plan), dense vs MoE.
4. Measured throughput (tok/s) and, if you can, MFU — and the per-component step profile
   (attention / GLA-scan / MLP / optimizer share). This directly feeds IDEA 3's "profile before you
   pick a lever" (T0) on real tensor-core silicon.
5. The launch: is it running, where are the logs/checkpoints, ETA to first eval.

Do not start a multi-day run without confirming the size/fit with Sam if it's a big commitment. When in
doubt, smoke-test, show the numbers, and ask.
