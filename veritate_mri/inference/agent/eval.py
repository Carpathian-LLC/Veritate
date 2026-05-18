# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Eval harness for tool-using agents. A test case is a (prompt, oracle)
#   pair: prompt is what we give the agent, oracle decides if the final
#   answer is correct given the trajectory.
# - Oracle styles:
#     "exact":     final_answer must contain the expected substring
#     "numeric":   parse final_answer's last number; must equal expected
#                  (within rel-tolerance 1e-6)
#     "calls":     trajectory must include a specific tool call (action match)
#     "answers":   trajectory must terminate with {"answer": ...}
# - The harness measures:
#     (a) JSON-validity rate    , should be ~100% with build-7 constraint
#     (b) schema-validity rate  , fraction of turns with valid action/answer
#     (c) answered rate         , fraction that produced any final answer
#     (d) tool-call-correct rate, fraction with the expected tool used
#     (e) final-answer correct  , overall pass rate
# - The cases below are byte-level-tractable: short prompts, short answers,
#   no fancy reasoning. They're the floor, a useful 1B model should clear
#   them all.
# veritate_mri/agent/eval.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .loop import AgentLoop, AgentResult
from .tools import Toolbox, build_default_toolbox

# ------------------------------------------------------------------------------------
# Constants

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class EvalCase:
    name:           str
    prompt:         str
    oracle:         str            # "exact" / "numeric" / "calls" / "answers"
    expected:       Any
    expected_tool:  Optional[str] = None    # for "calls" oracle


@dataclass
class CaseResult:
    case:           EvalCase
    agent_result:   AgentResult
    passed:         bool = False
    why:            str = ""


def _check(case: EvalCase, res: AgentResult) -> "CaseResult":
    cr = CaseResult(case=case, agent_result=res)
    # Universal: any turn that produced a valid schema?
    if not any(t.schema_error is None for t in res.turns):
        cr.passed = False
        cr.why = "no turn produced a schema-valid JSON object"
        return cr

    if case.oracle == "answers":
        cr.passed = res.final_answer is not None
        cr.why = "answered" if cr.passed else "did not produce an {answer: ...} object"
        return cr

    if case.oracle == "calls":
        called = any(t.action == case.expected_tool for t in res.turns)
        cr.passed = called
        cr.why = f"called {case.expected_tool!r}" if called else \
                 f"never called {case.expected_tool!r}; called {[t.action for t in res.turns if t.action]}"
        return cr

    if res.final_answer is None:
        cr.passed = False
        cr.why = "no final answer produced"
        return cr

    if case.oracle == "exact":
        needle = str(case.expected).lower().strip()
        hay    = res.final_answer.lower().strip()
        cr.passed = needle in hay
        cr.why = f"{'found' if cr.passed else 'missing'} substring {needle!r} in answer"
        return cr

    if case.oracle == "numeric":
        nums = _NUM_RE.findall(res.final_answer)
        if not nums:
            cr.passed = False
            cr.why = "answer has no number"
            return cr
        try:
            got = float(nums[-1])
            exp = float(case.expected)
        except ValueError:
            cr.passed = False
            cr.why = f"non-numeric answer: {nums[-1]!r}"
            return cr
        rel = abs(got - exp) / max(1e-9, abs(exp))
        cr.passed = rel < 1e-6
        cr.why = f"got {got!r} vs expected {exp!r} (rel err {rel:.2e})"
        return cr

    cr.passed = False
    cr.why = f"unknown oracle: {case.oracle}"
    return cr


def default_cases() -> List[EvalCase]:
    """Byte-level-tractable test cases. The floor a useful tool-using 1B
    should clear. Edit / extend in-place, the harness is dispatch-friendly."""
    return [
        # Pure arithmetic via calculator
        EvalCase("calc_basic_add",     "What is 17 + 24?",
                 oracle="numeric", expected=41, expected_tool="calculator"),
        EvalCase("calc_basic_mul",     "Compute 13 times 7.",
                 oracle="numeric", expected=91, expected_tool="calculator"),
        EvalCase("calc_sqrt",          "What is the square root of 144?",
                 oracle="numeric", expected=12, expected_tool="calculator"),
        EvalCase("calc_compound",      "Calculate (45 + 18) * 2 - 11.",
                 oracle="numeric", expected=115, expected_tool="calculator"),
        EvalCase("calc_log",           "What is log base 10 of 1000?",
                 oracle="numeric", expected=3, expected_tool="calculator"),

        # Tool selection: should choose calculator, not just guess
        EvalCase("must_use_calculator", "What is 47 times 89?",
                 oracle="calls", expected=None, expected_tool="calculator"),

        # Termination behavior: must emit {answer: ...} eventually
        EvalCase("greeting_terminates", "Say hello.",
                 oracle="answers", expected=None),
        EvalCase("knowledge_terminates", "What is the capital of France?",
                 oracle="answers", expected=None),

        # String matching for known facts (without retrieval/tools)
        EvalCase("answer_2plus2",       "What is 2 plus 2? Answer with just the number.",
                 oracle="exact", expected="4"),
    ]


def run_eval(backend, toolbox: Optional[Toolbox] = None,
             cases: Optional[List[EvalCase]] = None,
             max_turns: int = 4,
             max_new_per_turn: int = 256,
             temperature: float = 0.5,
             top_k_sample: int = 20,
             best_of_n: int = 1,
             verbose: bool = True) -> Dict[str, Any]:
    if toolbox is None:
        toolbox = build_default_toolbox()
    if cases is None:
        cases = default_cases()

    loop = AgentLoop(
        backend=backend, toolbox=toolbox,
        max_turns=max_turns, max_new_per_turn=max_new_per_turn,
        temperature=temperature, top_k_sample=top_k_sample,
        best_of_n=best_of_n,
    )

    results: List[CaseResult] = []
    t_start = time.time()
    for i, case in enumerate(cases):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(cases)} {case.name}: {case.prompt!r}")
        res = loop.run(case.prompt)
        cr = _check(case, res)
        results.append(cr)
        if verbose:
            mark = "PASS" if cr.passed else "FAIL"
            print(f"  [{mark}] {cr.why}")
            if res.final_answer:
                print(f"  answer: {res.final_answer[:160]!r}")
            for t in res.turns:
                if t.action:
                    print(f"    -> action: {t.action}({json.dumps(t.args)[:80]})")

    elapsed = time.time() - t_start

    # Aggregate stats
    n_total = len(results)
    n_passed = sum(1 for r in results if r.passed)
    json_valid = sum(1 for r in results
                     for t in r.agent_result.turns
                     if t.parse_error is None and len(t.raw_bytes) > 0)
    total_turns = sum(len(r.agent_result.turns) for r in results)
    schema_valid = sum(1 for r in results
                       for t in r.agent_result.turns
                       if t.schema_error is None)
    answered = sum(1 for r in results if r.agent_result.final_answer is not None)
    tool_called_correctly = sum(1 for r in results if r.case.expected_tool
                                and any(t.action == r.case.expected_tool
                                        for t in r.agent_result.turns))
    tool_expectation_count = sum(1 for r in results if r.case.expected_tool)

    out = {
        "n_cases":               n_total,
        "pass_rate":             n_passed / max(1, n_total),
        "json_validity_rate":    json_valid / max(1, total_turns),
        "schema_validity_rate":  schema_valid / max(1, total_turns),
        "answered_rate":         answered / max(1, n_total),
        "tool_correct_rate":     tool_called_correctly / max(1, tool_expectation_count),
        "elapsed_s":             elapsed,
        "config": {
            "max_turns": max_turns,
            "max_new_per_turn": max_new_per_turn,
            "temperature": temperature,
            "top_k_sample": top_k_sample,
            "best_of_n": best_of_n,
        },
        "details": [
            {
                "name":   r.case.name,
                "passed": r.passed,
                "why":    r.why,
                "final_answer": r.agent_result.final_answer,
                "n_turns": len(r.agent_result.turns),
            } for r in results
        ],
    }

    if verbose:
        print(f"\n=== EVAL SUMMARY ===")
        print(f"  cases:               {n_total}")
        print(f"  passed:              {n_passed}/{n_total} ({100*out['pass_rate']:.1f}%)")
        print(f"  json validity:       {100*out['json_validity_rate']:.1f}%")
        print(f"  schema validity:     {100*out['schema_validity_rate']:.1f}%")
        print(f"  answered:            {100*out['answered_rate']:.1f}%")
        print(f"  tool correct:        {100*out['tool_correct_rate']:.1f}% "
              f"(of {tool_expectation_count} cases that expected a tool)")
        print(f"  elapsed:             {elapsed:.1f}s")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Agent eval harness smoke runner.")
    ap.add_argument("--ckpt",      required=True, help="Path to a Veritate .pt checkpoint.")
    ap.add_argument("--out",       default="",    help="Where to write the JSON report. Defaults to <ckpt-stem>_agent_eval.json next to the checkpoint.")
    ap.add_argument("--threads",   type=int, default=2)
    ap.add_argument("--max_turns", type=int, default=2)
    ap.add_argument("--max_new",   type=int, default=96)
    ap.add_argument("--best_of_n", type=int, default=2)
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
    from inference.backends.pytorch import Brain
    brain = Brain(args.ckpt, threads=args.threads)
    print(f"loaded {brain.n_params/1e6:.1f}M params from {args.ckpt}")
    out = run_eval(brain, max_turns=args.max_turns, max_new_per_turn=args.max_new,
                   best_of_n=args.best_of_n, verbose=True)
    out_path = args.out or os.path.splitext(args.ckpt)[0] + "_agent_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
