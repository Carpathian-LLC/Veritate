# bigmodel byteification handoff

Scope + test results for a proposed training option: convert a large pretrained subword coding LLM (Qwen2.5-Coder) into a byte-level (vocab=256) Veritate-family model, then continue-train it. Written as a handoff so the decision is made on numbers, not optimism.

Date: 2026-07-01. Target hardware: single M3 Ultra (256 GB unified, 80-core GPU, PyTorch MPS).

## TL;DR verdict

- The **mechanism works and is cheap to set up.** Qwen uses byte-level BPE, so all 256 byte values already exist as tokens; warm-starting the new byte embedding/head from those rows is exact and free. Validated live (256/256 rows mapped).
- **It does NOT beat the compute wall on this box.** Byteification moves the wall, it does not remove it. Recovering the model's coding ability after the tokenizer swap needs tens of billions of tokens of continue-training; at byte-level throughput on this machine that is measured in **years**, not the sub-3-month target.
- It is **cleanly buildable** as a new trainer (~4-5 files) but the result is **PyTorch-only** (RoPE+GQA is not engine-exportable per preflight 40b), which tensions the project's energy-efficient-inference mission.
- **Honest recommendation:** build it only as a long-horizon byte-level *research* artifact at the 0.5B size, eyes open that it will not reach competitive HumanEval within a quarter. For an actual working ~3B Python coder today, Qwen2.5-Coder-3B runs via Ollama at HumanEval 52.4 for $0; that is outside the "train our own" lane, stated for completeness.

## what byteification is (plain)

A model like Qwen reads text in chunks ("print", "def", "_name"), ~150k possible chunks. Our models read one byte at a time, 256 values. Conversion = keep the model's transformer body (the part that knows how to code), discard its input embedding table and output head (the chunk<->number interface), bolt on 256-wide byte versions, warm-start them from the model's own single-byte tokens, then continue-train so the body adapts to reading/writing one byte per step.

## empirical results (this box, Qwen2.5-Coder-0.5B)

PoC script: throwaway `/tmp/byteify_poc.py` (surgery + before/after) and `/tmp/byteify_train.py` (recovery continue-train). Not committed.

1. **Baseline (original BPE Qwen2.5-Coder-0.5B) codes well.**
   - `"def fibonacci(n):"` -> correct recursive fibonacci + a second variant.
   - `"# reverses a string\ndef "` -> `reverse_string(s): return s[::-1]` + palindrome + pangram.
2. **Warm-init is exact: 256/256 byte rows** copied directly from Qwen's byte-level-BPE base-byte token embeddings (via the GPT-2 `bytes_to_unicode` map). No approximation needed.
3. **Byteified but NOT retrained -> degenerate** (` \n \n \n`, `100d\n# 100d`). Expected: the body was trained on multi-byte chunks per position; one byte per position is an unseen input distribution. Warm-init gives correct embedding vectors, not a working model.
4. **Recovery continue-train (byteified 0.5B on the 10 MB distilled Python corpus, seq512 bs16 lr2e-4, MPS):** loss descending from the degenerate start; ~2215 tok/s solo on the freed GPU. Early-recovery samples: SEE LIVE SECTION BELOW (updated as evals land).

## compute cost / go-no-go (agent A findings, cited)

- **Warm-init:** free, exact, one-shot (direct base-byte-token copy). Mean-of-constituents (FVT) degenerates to the direct copy here; hypernetwork approaches (Zett) are wasted effort.
- **Recovery budget (measured, literature):**
  - Bolmo (arXiv 2512.15586), the only true byteification precedent: byteified OLMo, HumanEval pass@16 71.1 -> 74.7 (coding improved), at ~49B adaptation tokens ~ 216B raw bytes (<1% of pretraining). CRITICAL: Bolmo is NOT a naive vocab=256 swap; it adds BLT-style patch modules (~4.4 bytes per transformer step). Its cheap budget depends on that patching. A naive 1-byte-per-step swap does not inherit it.
  - Cross-tokenizer byte distillation with negligible continued-pretrain (2604.07466) = the falsifier: MMLU 60.5 -> 39.1, ARC-C 45.7 -> 30.9. Converting-and-not-training-enough collapses the model.
  - BLT (2412.09871) from-scratch byte crossover floor: byte beats BPE only after ~150B bytes @450M, ~1T bytes @3.9B.
- **Wall-clock on this box** (byte-level = 1 token per raw byte; ~1980 B/s @0.5B, ~458 B/s @3B, scaled from measured 550 tok/s@2.5B):

  | Qwen size | bytes/day | days to 40B bytes | days to 216B bytes (Bolmo-equiv HQ) |
  |---|---|---|---|
  | 0.5B | 0.17B | ~234 d | ~1263 d (3.5 yr) |
  | 1.5B | 0.07B | ~561 d | ~3030 d (8.3 yr) |
  | 3.0B | 0.04B | ~1010 d | ~5455 d (15 yr) |

- **Verdict: NO-GO to a useful coder on this box at any size.** HQ recovery is off by 10x-100x from the <3-month target. The only budget that fits (~10-15B bytes @0.5B in ~2 months) is below every measured recovery threshold, i.e. a model that has not recovered. No reachable byteified tier beats Qwen2.5-Coder base (HumanEval 0.5B=28.0, 1.5B=43.9, 3B=52.4), available now via Ollama at $0.
- **Falsifier:** below ~150B bytes of adaptation (BLT sub-1B crossover), a byteified model does not reliably beat a same-size from-scratch byte model; 150B bytes ~ 2.6 years at 0.5B here. No reachable budget clears it.

## integration design (agent B findings, if built anyway)

Buildable cleanly as a NEW trainer. New code (~4-5 files); the training loop, save path, and multicorpus are reused as-is.

- **`trainers/byteify_qwen/`** (manifest + thin `trainer.py` shim). NOTE rule 34a: a new trainer dir must be mirrored to the canonical trainers repo or `/trainers/git/sync` erases it. Manifest carries a `base_model` HF-path arg and `model_type: code` default. Size comes from the loaded Qwen config, not a `sizes` preset.
- **`veritate_core/model_qwen.py::VeritateQwen`** (NEW class, platform code). None of the three existing classes host Qwen's arch: canonical `Veritate` (learned-pos/GELU/tied/MHA), `VeritateRoPE` (RoPE but MHA/GELU/tied/no-QKV-bias), and there is no live `Veritate800M` class (the 800m trainer trains canonical `Veritate`; the `load.py` Veritate800M/85M branches are dead). Qwen needs RoPE + GQA + QKV-bias + SwiGLU + untied head. Must expose the model-invariant contract methods (preflight 11a) so Brain never branches on it.
- **`veritate_core/plugin/byteify.py::byteify_qwen(base_model)`** (NEW helper, ~70-100 LOC): HF load, name-map copy of all transformer weights (q/k/v+bias, o, gate/up/down, both RMSNorms, final norm), replace embed+head with 256-wide, warm-init the 256 rows.
- **`hook_spec()` adapter REQUIRED** on `VeritateQwen`. Even at `model_type=code`, `dump_probe`/`dump_classroom` always run and access `blk.ff.up`/`blk.ff.down`/`blk.attn.qkv`. A SwiGLU/GQA block lacks those names, so `save.save()` AttributeErrors on the first checkpoint without an adapter view (~30-50 LOC).
- **`veritate_core/load.py`** +1 branch keyed on an `arch=="qwen"` config marker (the one legal place to branch).
- **Engine/export (preflight 40b):** RoPE+GQA is NOT `.bin`-exportable; runs PyTorch-only via `Brain`. No fast C-engine decode, no INT8 engine path, no low-energy deployment. State this plainly: it is a quality/research artifact, not a shippable efficient-inference model.
- **Early falsifier to verify FIRST (~5 min):** build a tiny random `VeritateQwen` (H=32, 2 layers, vocab=256, n_kv<n_heads), set `VERITATE_MODEL_TYPE=code`, call `save.save(...)`, assert `hooks/step_1/` writes with no exception. Cheaper than discovering the adapter gap after a multi-hour run.

## recommendation

1. Do not expect a useful Python coder from byteification on this hardware. The compute wall is the same one that stopped the from-scratch 2.5B (`pycoder_3b`, plateaued at val ~1.0; see overnight_run_log.md 2026-06-27).
2. If byte-level conversion is wanted as research for its own sake: build the `byteify_qwen` trainer at the **0.5B** size (cheapest), run the `hook_spec` smoke first, and treat it as a multi-quarter background experiment. It will demonstrate the mechanism and produce a byte-level model, but not a competitive HumanEval within a quarter.
3. The distilled Python SFT corpus (`trainers/corpus/py_distill_v1`, ast.parse-filtered, ~10 MB and growing) is reusable for any of these paths.

## live recovery test (updated as it runs)

Byteified Qwen2.5-Coder-0.5B, continue-train on the distilled Python corpus, MPS solo.

- loss: step 50 = 2.03, 100 = 1.04, 150 = 0.70, 200 = 0.60, 250 = 0.49, 300 = 0.40. ~2600 tok/s solo.
- step 300 sample: `"def fibonacci(n):\n"` -> `"        # Check if n is a palindrome\n            if not node.isdigit():\n"`. Structurally valid Python (indentation, comment, if + method call) after ~2.5M bytes.
- CONTRAST worth noting: the from-scratch 2.5B (`pycoder_3b`) at 205M tokens produced NO code, only looping English. The byteified 0.5B produces Python syntax at 2.5M bytes (80x less data, 5x smaller). Warm-start recovery of code STRUCTURE is dramatic and cheap.
- step 600 (loss 0.26): `"...reverses a string...<|assistant|>\n"` -> `"def reverse_string(s):\n    reversed_s = \"\"  # Initialize an empty string..."` (correct, instruction-following).
- step 900 (loss 0.17): `"def fibonacci(n):\n"` -> `"        # Base case: return n if n is 0 or 1\n        return n\n    # Recursive ca..."` (correct fibonacci logic on a RAW prompt).

### DECISIVE held-out result: HumanEval pass@1 = 0/30 = 0.0% (step 2000)
Mid-run, the in-distribution samples (correct reverse_string, is_prime, fibonacci) looked like real recovery and I briefly revised toward "usable coder reachable in days." The objective held-out test KILLS that read:
- `byteify_humaneval.py` on `step_2000.pt`, 30 real HumanEval problems, generated byte-by-byte, unit tests executed: **0/30 = 0.0%** (raw prompts).
- Templated-framing spot check (matching the model's chat-template training distribution) was WORSE: asked for `truncate_number`/`all_prefixes`/`strlen`, it emitted unrelated functions (`do_rectangles_intersect`, `generate_parser`, `generate_password`) with syntax errors. It ignores the prompt entirely.
- Held-out training samples had already hinted at this: `is_prime` sometimes right (common pattern), `sum_even` confused (bled prime logic into a sum). It memorized SURFACE patterns of the 10 MB distilled corpus, not coding.

CONCLUSION: the cheap byteification path (surgery + tiny distilled corpus + short train) produces a Python-SHAPED babbler at 0% HumanEval, not a coder. This CONFIRMS agent A, not the mid-run optimism. The surgery/warm-init is free and real; the recovery of actual coding ability is the expensive part and is NOT dodged by a small corpus. To have any chance it needs a LARGE, DIVERSE, REAL code corpus (codeparrot/the-stack, tens of GB) + long training = the weeks-to-months+ budget on this box, which loops back to the same compute wall. The in-distribution samples are a trap: they look like success and aren't. Always run held-out HumanEval before believing a byteified checkpoint.
