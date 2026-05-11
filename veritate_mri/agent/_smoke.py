# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - End-to-end smoke for the agent loop. Loads the trained 85M TinyStories
#   checkpoint via the existing Brain backend, builds a default toolbox, and
#   runs a handful of synthetic user queries.
# - The 85M model is NOT trained on tool use, so it WILL produce nonsense for
#   the "action" / "answer" content. The point of this smoke is to verify:
#     (1) the agent loop machinery functions end-to-end without crashing
#     (2) JSON validity is 100% by construction (build-7 constraint)
#     (3) the schema validator catches bad / missing fields
#     (4) tool execution + observation injection works
#     (5) the loop respects max_turns and terminates cleanly
# - A real evaluation requires a tool-use SFT'd model. That's W08/W09 work.
# veritate_mri/agent/_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

from backends.pytorch import Brain
from agent import AgentLoop, build_default_toolbox

CKPT = os.path.join(REPO, "experiments", "overnight", "ckpt_step_2000.pt")
CORPUS_PATH = None  # we won't index a corpus for the smoke
FS_ROOT = None      # no filesystem tool either (avoid leaking test machine paths)

PROMPTS = [
    "What is 2 plus 2?",
    "Compute sqrt(2) times pi.",
    "Hello, what is your name?",
    "Tell me a story about a dragon.",
]


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading Brain from {CKPT}")
    t0 = time.time()
    brain = Brain(CKPT, threads=2)
    print(f"  loaded in {time.time()-t0:.1f}s; {brain.n_params/1e6:.1f}M params")

    toolbox = build_default_toolbox(corpus_path=CORPUS_PATH, fs_root=FS_ROOT)
    print(f"  toolbox: {toolbox.names()}")

    loop = AgentLoop(
        backend=brain,
        toolbox=toolbox,
        max_turns=3,
        max_new_per_turn=128,    # tight on the untrained model — keep iterations cheap
        temperature=0.7,
        top_k_sample=20,
    )

    summary = []
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n[{time.strftime('%H:%M:%S')}] === prompt {pi+1}/{len(PROMPTS)}: {prompt!r} ===")
        t0 = time.time()
        result = loop.run(prompt)
        elapsed = time.time() - t0
        print(f"  total: {elapsed:.1f}s, turns: {len(result.turns)}, stop: {result.stop_reason}")
        for ti, t in enumerate(result.turns):
            print(f"  turn {ti+1} ({t.elapsed_s:.1f}s):")
            print(f"    raw_bytes ({len(t.raw_bytes)} bytes): {t.raw_bytes[:200]!r}...")
            if t.parse_error:
                print(f"    parse_error: {t.parse_error}")
            if t.schema_error:
                print(f"    schema_error: {t.schema_error}")
            if t.action:
                print(f"    action={t.action}  args={json.dumps(t.args)[:120]}")
                obs = t.observation or ""
                print(f"    observation: {obs[:200]}")
            if t.answer:
                print(f"    ANSWER: {t.answer[:200]}")
            if t.thought:
                print(f"    thought: {t.thought[:200]}")
        if result.final_answer:
            print(f"  >>> final answer: {result.final_answer[:240]}")
        else:
            print(f"  (no final answer)")
        # JSON-valid: bytes parsed as JSON without error (any type)
        # Schema-valid: top-level is a dict with action or answer
        json_valid = sum(1 for t in result.turns if t.parse_error is None and len(t.raw_bytes) > 0)
        schema_valid = sum(1 for t in result.turns if t.schema_error is None)
        summary.append({
            "prompt": prompt,
            "turns":  len(result.turns),
            "answered": result.final_answer is not None,
            "stop": result.stop_reason,
            "elapsed_s": elapsed,
            "json_valid_rate": json_valid / max(1, len(result.turns)),
            "schema_valid_rate": schema_valid / max(1, len(result.turns)),
        })

    print(f"\n=== SUMMARY ===")
    n_answered = sum(1 for s in summary if s["answered"])
    avg_json = sum(s["json_valid_rate"] for s in summary) / len(summary)
    avg_schema = sum(s["schema_valid_rate"] for s in summary) / len(summary)
    print(f"  prompts run: {len(summary)}")
    print(f"  answered (any): {n_answered}/{len(summary)}")
    print(f"  avg JSON-validity rate: {avg_json*100:.1f}% (should be 100% if constraint works)")
    print(f"  avg schema-validity rate: {avg_schema*100:.1f}%")
    print(f"  (85M is NOT tool-trained — schema-validity is expected to be LOW;")
    print(f"   the load-bearing test is JSON-validity 100%.)")
    print()
    if avg_json >= 0.95:
        print(f"  VERDICT: agent loop machinery is HEALTHY.")
        print(f"  Real eval awaits a tool-use SFT'd model (W08/W09 dispatch).")
    else:
        print(f"  VERDICT: JSON constraint isn't enforcing 100% — investigate.")


if __name__ == "__main__":
    main()
