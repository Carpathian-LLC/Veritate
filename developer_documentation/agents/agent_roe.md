# agent roe

Rules of engagement for any agent on Veritate. Read `claude_preflight.md` first; this is the second doc. Subagents read both before any task.

## mission

Byte-level, energy-efficient LLM on consumer hardware ($300-1000 box; Apple Silicon, x86, Vulkan iGPU, Hailo NPU). Reject ideas that don't advance one of: byte-level, energy-efficient, consumer-hardware.

## hard rules

1. No git commit / push to the Veritate repo without explicit user permission. Local edits free; pushes only when authorized.
2. Do not touch a live training run (PID, ckpt dir, MPS device). Inference smokes default to CPU.
3. Read `claude_preflight.md` every session before significant work. ROE compliance is gating.
4. Check prior findings in the relevant `developer_documentation/<domain>/` doc before designing an experiment in that space.
5. No version bumps, no build-note edits, no merges, no model exports without explicit instruction.
6. Concise outputs: bullets over paragraphs, lines under 200 chars.
7. Field-symmetry mandate: every per-token frame the dashboard renders is emitted by both training-time and inference-time. Adding a field touches both in the same commit.
8. No new files / folders / agents / scripts / worktrees without explicit instruction (preflight rule 3).
9. Lead with the verdict, not the process. Honest losses are real deliverables; no sandbagging.

## where things live (paths drift; update when stale)

| what | path |
|---|---|
| Veritate repo (x86 dev) | `c:\GitHub\Veritate\` |
| Veritate repo (M3 Ultra) | `/Users/mirach-00-usc1/Development/Veritate` |
| 85M base ckpt | `experiments/overnight/ckpt_final.pt` (val 0.46) |
| 800M ckpt-in-progress | `experiments/v2/run_800m/ckpts/` (do not perturb) |
| TinyStories val | `trainers/corpus/tinystories_val.bin` |
| FineWeb val | `trainers/corpus/fineweb_edu_val.bin` |
| Tool-SFT bins | `trainers/corpus/tool_sft_{train,val}.bin` |
| Canonical model class | `veritate.model.Veritate` (h=768 L=12 GELU absolute-pos) |
| 800M model class | `trainers/veritate_800m/trainer.py::Veritate800M` (RoPE + MTP) |
| Brain (inference) | `veritate_mri/inference/backends/pytorch.py::Brain` |
| Agent loop / eval | `veritate_mri/agent/{loop,eval}.py` |
| Smoke output dir | `SMOKE_RESULTS/` |
| Versions ledger | `versions.json` at repo root |

## model invariants (preflight rule 11a)

The model class knows what it is; inference code never branches on variant. Brain dispatcher is 3-way:
- Canonical `Veritate`: `pos_emb.weight` present → learned pos, single `lm_head`, no MTP.
- `VeritateRoPE`: no `pos_emb`, no `mtp.transforms.*` → RoPE only.
- `Veritate800M`: has `mtp.transforms.*` → RoPE + MTP. Byte-0 routes through `mtp.lm_head(mtp.norms[0](mtp.transforms[0](h)))`.

Forward returns `(logits, loss)`; loss may be None. New variant ships a shared contract method on the class (`model.project_byte0(residual)`); consumers call it blindly. Adding a variant must never require touching decoder, agent loop, speculative draft, or engine wiring.

Brain loader reads shape from `args` OR `config` key. v2 experimental ckpts use `config`; canonical/800M use `args`.

`Brain.stream()` runs full per-byte telemetry. `Brain.stream_fast(mode="kv"|"mtp")` runs fast paths with coarse `fast_byte` events; don't use fast mode when the dashboard needs per-byte FFN/attention/lens captures.

## training invariants

- `lr_at(step, total, warmup, base_lr, min_lr, schedule)` is the WSD shape. Decay phase uses `(step - warmup) / (total - warmup)`.
- `save.save(model, name, step, optimizer=opt, args=vars(args))` for ALL ckpt writes. `append_train_row` for ALL CSV rows. No trainer writes `.pt` or CSV directly.
- Model dir name: `<corpus>_<size>_<precision>_<version>[_<variant>]`. Variant is one lowercase token, no underscores. Override via `--output_name`.
- Trained 85M activation profile: pre-GELU σ ≈ 0.93 (not σ ≈ 3 like Llama-class). PTQ recipes assuming σ ≈ 3 mis-tune at this size.
- LN-fold ordering in QAT2: fake-quant the post-LN activation BEFORE multiplying by INT8/64-quantized ln_w, or get 33% more residual drift.
- Q&A and structured formats must appear from Stage A; the model cannot discover formats it never saw.
- ≥3 seeds before reporting any small-scale architecture comparison < 5%. Single-seed deltas at byte-level shape are noise.
- Apply resume overrides: 800M trainer reads training_args from `config.json`; CLI flags override saved ones; optimizer state preserved.

## engine invariants

- OS-specific primitives live behind a single shim in `veritate_engine/src/`. Per-arch kernels never include OS headers or call OS APIs directly.
- Every kernel produces bitwise-identical output to its scalar reference before shipping.
- Cross-platform function-pointer contract: ARM SDOT ignores `q_sum` bias param; x86 VNNI uses it. Shared signature converges naturally.
- Matmuls = ~95% of forward cost at byte-level. Attention is ~2% of total decode (only matters at pos > 200).
- Sparse-decode kernel buffer trap: static buffers sized off `V_FFN` overflow when runtime shape exceeds compile-time cap. Use runtime alloc or max-shape sentinel.
- INT4 packed matmul needs `vpermt2b` (AVX-512 VBMI). Until then INT4 is on-disk-only.

## dashboard invariants

- Never hardcode model-specific details. Layer counts, FFN sizes, region bounds scale from the loaded model.
- `plotTrainSeries` requires `s.points.length > 1` to draw a line; pass `dots: true` for single-point series.
- `yAutoFit` zooms y-axis to dataMax × pad; threshold lines above visible frame render as top-edge "↑ above chart" labels.
- Don't surface CLI commands as user-facing errors. Silently omit or emit a one-line plain-language note.
- Don't reference `/docs/` — private to development, not in shipped repo.
- `pytorch_load_mode: "always"` eagerly loads brain at startup. 800M ckpt = 2-3 min startup; use `"on_demand"` to avoid.

## smoke contract

Every smoke script:

```
SMOKE_RESULTS/<your_name>_smoke.py
SMOKE_RESULTS/<your_name>_stats.json   # produced by script
```

Exit 0 on success, 1 on errors-but-recorded. JSON has top-level `errors: []`. Real test result lives under any other key.

Header for every smoke:

```python
# Notes:
# - what's tested + falsifier in one line.
# - wall-clock estimate.
# SMOKE_RESULTS/<your_name>_smoke.py
```

Checklist before starting:
- Check prior findings in `developer_documentation/<domain>/` for your topic.
- State the falsifier in one sentence.
- Wall-clock < 4 hours (else split the smoke).
- Device = CPU unless specifically needing MPS and 800M is done.
- Script writes ONE JSON stats file. No plots, no extra outputs.

## multi-agent collaboration protocol

Active agents (table; keep current as new boxes join):

| agent id | box | hardware | primary role |
|---|---|---|---|
| agent-mirach | M3 Ultra | 256 GB unified, 80-core GPU | platform dev + research dispatch + 85M/800M smokes |
| agent-mintaka | Mac mini | hermes3:8b via Ollama | tool-use SFT data gen (W13/W15) |
| agent-malkaisan | x86 dev box | Ryzen 7 9800X3D, 96 MB L3, 32 GB RAM, 12 GB VRAM | engine/MRI dev (x86 + AVX-512 VNNI) |

When you take a new task:
1. Read this file's last entries end-to-end.
2. Write a "starting W<N>" handoff note (in your local working log).
3. Estimate wall-clock + restate the falsifier.

When you complete a task:
1. Record the outcome (resolve / falsify / partial) in the relevant `developer_documentation/<domain>/` doc with a one-line pointer to the smoke.
2. Push smoke code + stats JSON to `SMOKE_RESULTS/`.
3. Tag the next agent if applicable.

Standing requests:
- If a falsified entry's retry condition is now met, DO try it again; document outcome.
- If you discover a durable invariant or gotcha, add it directly to this file or `claude_preflight.md`. Don't ask permission.
- Keep the active-agents table current as boxes join.

## subagent dispatch contract

Hardware budget classes:
- **Day-class:** <= 24 h. Smoke tests, no-training profiles, kernel ports.
- **Week-class:** <= 7 d. Single model train + eval.
- **Sprint-class:** <= 14 d. Multi-condition train + eval, multi-seed.

No dispatch requires GPU access to a frontier cloud model unless explicitly marked "teacher needed."

Every dispatch prompt MUST include:
1. "Read `./claude_preflight.md` (repo root) first" + "Read `developer_documentation/agents/agent_roe.md` second."
2. The success criterion and the falsifier, stated explicitly.
3. Hardware budget + wall-clock cap. Subagent bails and reports if cap exceeded.
4. Outcome contract — one of:
   - SUCCESS — propose code changes if applicable, ping operator.
   - FAILURE — report the negative result with its retry condition.
   - INCONCLUSIVE — return a tight blocker description; no doc edits.
5. "No commits, no pushes, no model exports. Operator decides."

If a dispatch needs a methodology document, the experiment isn't well-formed.

## specialized agent roles (when invoked)

- **Anti-overengineering reviewer.** Mandate: least code wins. Every line is guilty until proven necessary. Flag wrappers with one user, defensive checks at internal boundaries, never-reconfigured "config", premature flexibility, generic-where-specific, comments restating code, error handling for impossible errors, hot-path indirection. Bias toward deletion; user overrides.
- **Code review.** Mandate: cleanliness + style match. Header block, snake_case, sparse imperative comments (no articles, no rationale), no TODO/FIXME/commented-out code, no PR refs or ticket numbers. Reviews only; no edits, no benchmarks, no research.
- **Education / research currency.** Mandate: track quantization, analog hw, efficient inference, CPU SIMD, latent reasoning, compilers, latency reduction. File a research note only when SOTA shifts, a chip ships Veritate could target, or a benchmark contradicts a project assumption. Filter aggressively; the user reads what you write.

## anti-overengineering invariant (preflight rule 19)

Two layers of abstraction is one too many. If a feature can be removed without breaking the goal, remove it. No ad-hoc code anywhere; one module owns each concern.
