# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - AgentLoop: a ReAct-style multi-turn loop that drives any object exposing
#   `stream(prompt, ..., constraint=...)` (matches Brain.stream signature) and
#   wraps it in tool-call dispatch.
# - Each turn the model produces ONE JSON object under JSONConstraint. The
#   object is one of:
#     {"action": "<tool_name>", "args": {...}, "thought": "<optional>"}
#     {"answer": "<final user-facing response>", "thought": "<optional>"}
#   Schema is validated AFTER parse (constraint enforces JSON well-formedness;
#   the loop enforces the schema). Bad schema -> a synthetic error observation
#   is injected and the loop continues. The model gets to recover.
# - Termination: explicit {"answer": ...} OR max_turns reached OR
#   constraint-allowed-no-bytes (rare; treated as fatal).
# - History format injected into the next prompt is a compact transcript:
#     [thought ...]
#     [action <tool>(<args_json>)]
#     [observation <text>]
#     ...
# veritate_mri/agent/loop.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tools import Toolbox

# ------------------------------------------------------------------------------------
# Constants

_DEFAULT_MAX_TURNS  = 8
_DEFAULT_MAX_NEW    = 384
_DEFAULT_OBS_TRIM   = 1024     # observation truncation length

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class AgentTurn:
    """One iteration of the loop. Captures what the model emitted + what
    happened when (if anything) we ran the requested tool."""
    raw_bytes:    bytes = b""
    parsed:       Optional[Dict[str, Any]] = None
    parse_error:  Optional[str] = None
    action:       Optional[str] = None
    args:         Optional[Dict[str, Any]] = None
    thought:      Optional[str] = None
    observation:  Optional[str] = None
    answer:       Optional[str] = None
    schema_error: Optional[str] = None
    stop_reason:  Optional[str] = None
    elapsed_s:    float = 0.0


@dataclass
class AgentResult:
    """Loop output. `final_answer` is None if the loop stopped without one."""
    final_answer: Optional[str] = None
    turns:        List[AgentTurn] = field(default_factory=list)
    stop_reason:  str = ""
    total_elapsed_s: float = 0.0


def _build_system_prompt(toolbox: Toolbox) -> str:
    """The prompt header. Tool list + JSON schema instructions."""
    return (
        "You are a careful assistant that solves the user's task by reasoning and using tools.\n"
        + toolbox.prompt_block() + "\n\n"
        "Every response you produce MUST be a single JSON object on one line.\n"
        "To use a tool, emit: "
        '{"thought": "what you are reasoning", "action": "<tool_name>", "args": {<arg>: <value>}}\n'
        'To answer the user, emit: {"thought": "what you are reasoning", "answer": "<final answer>"}\n'
        "After you emit a tool action, you will see the observation. Use it to decide the next step.\n"
        "Always finish with an {\"answer\": ...} object."
    )


def _format_history(turns: List[AgentTurn]) -> str:
    """Compact transcript of prior turns for prompt injection."""
    if not turns:
        return ""
    lines = ["", "Transcript so far:"]
    for t in turns:
        if t.thought:
            lines.append(f"[thought] {t.thought}")
        if t.action is not None:
            args_json = json.dumps(t.args or {}, separators=(",", ":"))
            lines.append(f"[action] {t.action}({args_json})")
        if t.observation is not None:
            obs = t.observation
            if len(obs) > _DEFAULT_OBS_TRIM:
                obs = obs[:_DEFAULT_OBS_TRIM] + f"\n... [truncated, {len(t.observation) - _DEFAULT_OBS_TRIM} more bytes]"
            lines.append(f"[observation] {obs}")
        if t.schema_error:
            lines.append(f"[error] {t.schema_error}")
    return "\n".join(lines)


def _build_turn_prompt(system: str, user: str, turns: List[AgentTurn]) -> str:
    history = _format_history(turns)
    return f"{system}\n\nUser: {user}\n{history}\n\nAssistant: "


def _collect_bytes(stream_gen) -> tuple:
    """Drain a Brain.stream generator. Returns (bytes_out, stop_reason)."""
    out = bytearray()
    stop = None
    for ev in stream_gen:
        k = ev.get("kind")
        if k == "token":
            b = ev.get("byte")
            if isinstance(b, int):
                out.append(b & 0xff)
        elif k == "fast_byte":
            b = ev.get("byte")
            if isinstance(b, int):
                out.append(b & 0xff)
        elif k == "stop":
            stop = ev.get("reason") or "stop"
            break
        elif k == "error":
            stop = f"error: {ev.get('message') or 'stream error'}"
            break
    return bytes(out), stop


class AgentLoop:
    """Tool-using loop on top of a Brain-shaped backend.

    Required `backend` interface:
        backend.stream(prompt, temperature, top_k_sample, max_new,
                       addons_chain=None, constraint=None) -> generator

    Brain (the MRI PyTorch backend) and Brain.stream_fast both satisfy this.

    Set `best_of_n > 1` to run K independent samples per turn and pick the
    one with valid schema + best score (lowest mean NLL by default). This
    is the published +20-30% inference-time quality lever — works on any
    model without retraining.
    """

    def __init__(self, backend, toolbox: Toolbox,
                 max_turns: int = _DEFAULT_MAX_TURNS,
                 max_new_per_turn: int = _DEFAULT_MAX_NEW,
                 temperature: float = 0.7,
                 top_k_sample: int = 40,
                 best_of_n: int = 1,
                 seed_base: int = 0):
        self.backend = backend
        self.toolbox = toolbox
        self.max_turns = int(max_turns)
        self.max_new = int(max_new_per_turn)
        self.temperature = float(temperature)
        self.top_k = int(top_k_sample)
        self.best_of_n = max(1, int(best_of_n))
        self.seed_base = int(seed_base)
        self.system_prompt = _build_system_prompt(toolbox)

    def run(self, user_input: str) -> AgentResult:
        # Avoid cycle: local import. We need a JSONConstraint subclass that
        # skips Brain.stream's "prime with full prompt" step, because the
        # agent prompt is natural language up to the Assistant: prefix, not
        # a JSON document. Priming on the prompt corrupts the parser state.
        from decode import JSONConstraint

        class _AgentJSONConstraint(JSONConstraint):
            def prime(self, prefix: bytes) -> None:
                return  # no-op; the prompt is not JSON

        res = AgentResult()
        t_start = time.time()
        import torch
        for turn_i in range(self.max_turns):
            turn = AgentTurn()
            t0 = time.time()
            prompt = _build_turn_prompt(self.system_prompt, user_input, res.turns)

            # Sample K candidates. K=1 is the default cheap path.
            # Pick the first that produces a valid (parseable + schema-correct)
            # action-or-answer JSON. If none do, take the longest output as a
            # representative for the error log.
            candidates = []
            for k in range(self.best_of_n):
                torch.manual_seed(self.seed_base + turn_i * self.best_of_n + k)
                constraint = _AgentJSONConstraint()
                gen = self.backend.stream(prompt,
                                          temperature=self.temperature,
                                          top_k_sample=self.top_k,
                                          max_new=self.max_new,
                                          constraint=constraint)
                raw, stop = _collect_bytes(gen)
                candidates.append((raw, stop))

            # Try each candidate in order; the first one that parses cleanly +
            # has a valid schema wins. Otherwise fall back to the longest
            # output for diagnostics.
            chosen_raw, chosen_stop = candidates[0]
            chosen_obj = None
            chosen_parse_err = None
            for raw, stop in candidates:
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    if chosen_parse_err is None:
                        chosen_parse_err = f"{type(e).__name__}: {e}"
                    continue
                if isinstance(obj, dict) and ("action" in obj or "answer" in obj):
                    chosen_raw, chosen_stop, chosen_obj = raw, stop, obj
                    chosen_parse_err = None
                    break
            # If none of the candidates had a valid schema, pick the longest
            # output (most informative for the error log).
            if chosen_obj is None:
                longest = max(candidates, key=lambda c: len(c[0]))
                chosen_raw, chosen_stop = longest

            raw, stop = chosen_raw, chosen_stop
            turn.raw_bytes = raw
            turn.stop_reason = stop
            turn.elapsed_s = time.time() - t0

            # Parse the JSON. The constraint guarantees well-formedness — but
            # only IF the model emitted enough bytes before max_new. Truncated
            # output is still possible. Catch + recover.
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                turn.parse_error = f"{type(e).__name__}: {e}"
                turn.schema_error = "model emitted incomplete JSON; retry"
                res.turns.append(turn)
                continue

            if not isinstance(obj, dict):
                turn.schema_error = f"top-level JSON must be an object, got {type(obj).__name__}"
                res.turns.append(turn)
                continue
            turn.parsed = obj
            turn.thought = obj.get("thought") if isinstance(obj.get("thought"), str) else None

            # Branch on schema:
            if "answer" in obj:
                ans = obj.get("answer")
                if not isinstance(ans, str):
                    turn.schema_error = "'answer' must be a string"
                    res.turns.append(turn)
                    continue
                turn.answer = ans
                res.turns.append(turn)
                res.final_answer = ans
                res.stop_reason = "answer"
                break
            if "action" in obj:
                action = obj.get("action")
                args = obj.get("args") or {}
                if not isinstance(action, str):
                    turn.schema_error = "'action' must be a string"
                    res.turns.append(turn)
                    continue
                if not isinstance(args, dict):
                    turn.schema_error = "'args' must be an object"
                    res.turns.append(turn)
                    continue
                tool = self.toolbox.get(action)
                if tool is None:
                    turn.schema_error = f"unknown tool {action!r}; available: {self.toolbox.names()}"
                    res.turns.append(turn)
                    continue
                turn.action = action
                turn.args = args
                turn.observation = tool.call(args)
                res.turns.append(turn)
                continue
            # Neither answer nor action present.
            turn.schema_error = "JSON object must contain either 'answer' or 'action'"
            res.turns.append(turn)
            continue

        if res.final_answer is None:
            res.stop_reason = res.stop_reason or "max_turns"
        res.total_elapsed_s = time.time() - t_start
        return res
