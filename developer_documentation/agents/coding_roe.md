# coding roe

Rules of engagement for writing code in Veritate. Read after `claude_preflight.md` and `agent_roe.md`. Every model writing or editing code here is required to read this. Rule numbers extend the preflight series so a single rule can be cited project-wide.

## the prime directives

100. **Lean code, simple code, no bloat.** If a line, function, file, or abstraction does not earn its place, delete it. The default for any addition is "no." Justify why it must exist before writing it.
101. **Write the simplest thing that could possibly work.** Then leave it alone. Premature generality is a tax paid forever; concrete code today is cheaper than a one-day-maybe abstraction.
102. **Measure before optimizing.** Replace "feels slow" with a number. Replace a hunch with a benchmark. Replace `try {} except: pass` with the actual error it catches.

## what lean means in practice

103. **Lines of code is a cost, not a deliverable.** A 50-line PR that solves the problem beats a 500-line PR that solves it and a future hypothetical problem. Smaller diffs are easier to review, easier to revert, easier to reason about.
104. **Functions do one thing.** If you can't name a function without "and" / "or", it's two functions. Cap soft length at ~30 lines; if you exceed that, ask whether it should split.
105. **No layered abstractions for one caller.** A class with one method called from one place is a function. A function called from one place can be inlined. Build abstractions on the second use, not the first.
106. **Delete code aggressively.** Dead code, commented-out code, "we might need this later," unreachable branches, unused params, defensive checks for impossible states — all go. The git history is the museum; the working tree is the office.
107. **Reuse before you re-write.** Before writing a helper, grep for an equivalent that already exists in `veritate_core/`, `veritate_mri/training/`, `veritate_mri/readers/`, or `trainers/common/`. Two callers of the same helper beats two implementations.

## naming + style

108. **Names carry the explanation.** A function named `_process_data` is a smell; `_decode_byte_stream` is a contract. Comments compensate for bad names; rename instead.
109. **Lowercase, snake_case, no abbreviations unless the abbreviation is the canonical term in the domain** (`lr`, `bf16`, `attn`, `mtp` are fine; `proc`, `mgr`, `cfg_hndlr` are not).
110. **One concept per name.** Don't use `model` for both the trainer state and the Python module. Don't use `step` for both training-step-counter and a function-named-step.
111. **Constants live at module top, never in function bodies.** Repeats preflight rule 11. If you find yourself writing a literal three times, name it once.

## what to NOT write

112. **No defensive code for cases that cannot happen.** Trust internal callers and framework guarantees. Validate at system boundaries only (user input, network, file I/O). Catching `Exception` to "be safe" hides real bugs.
113. **No backward-compatibility shims for code that has never shipped.** Inside Veritate, every caller is in the repo. Change them all in the same PR.
114. **No "TODO" / "FIXME" / "XXX" / commented-out blocks.** Delete or do. The repo is not a sticky-note board.
115. **No try/except as flow control.** `if x is None: ...` beats `try: x.foo() except AttributeError: ...`.
116. **No print-debugging left in.** Use `runtime.logs::logmod.{info,warn,error,ok}` for anything that should ship; delete anything that shouldn't.
117. **No docstrings that re-state the function signature.** A docstring earns its place by explaining *why* or *gotcha*, not *what*. Most one-liner functions need no docstring; their name is the docstring.

## performance hygiene

118. **Profile, then optimize.** Pull `time.perf_counter()`, `cProfile`, `torch.profiler`, `Instruments.app`. Optimizing code you didn't measure is theater.
119. **Optimize the inner loop, not the setup.** A 1ms speedup in a 1000-iter loop beats a 100ms speedup in a once-per-run import.
120. **Allocation is the silent cost.** In hot paths, prefer in-place ops (`x.add_(y)`), pre-allocated buffers, and `set_to_none=True` for zero-grad over default zero-fill. On MPS specifically, every `.to(device)` round-trip is a stall — keep tensors on device.
121. **Fuse what you can, then check it's still correct.** Combining N small ops into one big op typically wins on GPU even at +5% peak FLOPs cost because launch overhead dominates at small N.
122. **Batch over loop.** `[f(x) for x in xs]` in Python is N kernel launches. `f(torch.stack(xs))` is one. Same math, often 5-50x faster on GPU.

## correctness gates

123. **A change that touches numerical code ships with a tolerance test.** Compare against the pre-change implementation on a fixed seed; assert max-abs-diff under a tolerance you stated explicitly.
124. **A new file ships with a test under `tests/<area>/test_*.py`** (preflight rule 51). No new module is "tested by the existing test suite" — name the test.
125. **A change that adds a kernel ships with a scalar reference and a bitwise-identity check** (preflight rule 24).

## review your own diff before declaring done

126. **Re-read the diff in the editor.** Ask: would a reviewer who has never seen this branch understand why each line is there? If no, fix it.
127. **Run the file's own tests** before claiming the change works. "Should work" is not a status.
128. **Quantify what changed.** "+50 lines, -20 lines, 1 new public function, 0 deps added" is a sentence. "Refactored auth" is not.

## the lean code mantra

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away." — Saint-Exupéry

Print this on the inside of your forehead. Every PR should subtract complexity it cannot justify adding.

## escalation

If a rule here conflicts with a real constraint (a vendor SDK, a wire format, a hardware quirk), the rule loses and the constraint wins — but say so out loud in the PR / comment / chat, so the reader knows you noticed and chose the trade-off. Silent rule-breaking is the worst outcome.
