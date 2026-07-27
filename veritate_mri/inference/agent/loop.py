# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - AgentLoop: ReAct-style multi-turn loop driving any backend exposing
#   `stream(prompt, ..., constraint=...)` (matches Brain.stream).
# - Wire format: ChatML + Hermes function-calling, per
#   developer_documentation/corpus/framing.md. Each model turn is an assistant block
#   bounded by <|im_start|>assistant\n...<|im_end|>. Tool calls are emitted
#   as <tool_call>{"name": str, "arguments": object}</tool_call> blocks
#   inside the assistant turn. Tool replies are injected as a tool-role
#   ChatML turn: <|im_start|>tool\n<tool_response>{"name", "result"}
#   </tool_response><|im_end|>.
# - Termination: assistant turn without a <tool_call> -> treated as the
#   final answer; OR max_turns reached; OR backend stop.
# - Constraint: StopOnConstraint(b"<|im_end|>") so each turn halts at the
#   end of the assistant block rather than running through max_new.
# veritate_mri/inference/agent/loop.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .tools import Toolbox

# ------------------------------------------------------------------------------------
# Constants

_DEFAULT_MAX_TURNS   = 8
_DEFAULT_MAX_NEW     = 384
_DEFAULT_OBS_TRIM    = 1024
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_K       = 40
_DEFAULT_BEST_OF_N   = 1
_DEFAULT_SEED_BASE   = 0
_RAW_HEAD_CAP        = 120

# ------------------------------------------------------------------------------------
# Model contract. Every string below is what the model actually sees or is
# expected to emit (ChatML + Hermes function calling, developer_documentation/corpus/
# framing.md). Editing one edits the prompt the model was framed against, so
# they live here rather than inline in the builders.

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
IM_END_B = IM_END.encode("ascii")

ROLE_SYSTEM    = "system"
ROLE_USER      = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL      = "tool"

TOOL_CALL_OPEN      = "<tool_call>"
TOOL_CALL_CLOSE     = "</tool_call>"
TOOL_RESPONSE_OPEN  = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

KEY_NAME      = "name"
KEY_ARGUMENTS = "arguments"
KEY_ARGS_ALT  = "args"
KEY_RESULT    = "result"
KEY_ERROR     = "error"
JSON_SEPARATORS = (",", ": ")

TOOL_CALL_EXAMPLE = f'{{"{KEY_NAME}": "<tool_name>", "{KEY_ARGUMENTS}": {{<arg>: <value>}}}}'

SYSTEM_PROMPT_HEAD = (
    "You are a careful assistant that solves the user's task by reasoning and using tools.\n"
)
SYSTEM_PROMPT_TAIL = (
    "\n\nTo call a tool, emit a single line:\n"
    f"{TOOL_CALL_OPEN}\n{TOOL_CALL_EXAMPLE}\n{TOOL_CALL_CLOSE}\n"
    "Wait for the tool response, then continue. To finish, reply normally without a "
    f"{TOOL_CALL_OPEN} block."
)

OBS_TRIM_FMT  = "\n... [truncated, {n} more bytes]"
OBS_TRIM_VIEW = " ... [trimmed]"

STOP_ANSWER    = "answer"
STOP_MAX_TURNS = "max_turns"
STOP_DEFAULT   = "stop"
EMPTY_TURN_ERROR = "assistant turn was empty (no tool_call and no answer)"

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class AgentTurn:
    raw_bytes:    bytes = b""
    parsed:       dict[str, Any] | None = None
    parse_error:  str | None = None
    action:       str | None = None
    args:         dict[str, Any] | None = None
    thought:      str | None = None
    observation:  str | None = None
    answer:       str | None = None
    schema_error: str | None = None
    stop_reason:  str | None = None
    elapsed_s:    float = 0.0


@dataclass
class AgentResult:
    final_answer: str | None = None
    turns:        list[AgentTurn] = field(default_factory=list)
    stop_reason:  str = ""
    total_elapsed_s: float = 0.0


def _chatml_turn(role, body):
    return f"{IM_START}{role}\n{body}{IM_END}\n"


def _build_system_prompt(toolbox: Toolbox) -> str:
    return SYSTEM_PROMPT_HEAD + toolbox.prompt_block() + SYSTEM_PROMPT_TAIL


def _render_history(turns: list[AgentTurn]) -> str:
    """Render prior turns as ChatML transcript: assistant turns echo the
    model's bytes; observations become tool-role turns."""
    parts = []
    for t in turns:
        if t.action is not None:
            call_obj = json.dumps({KEY_NAME: t.action, KEY_ARGUMENTS: t.args or {}},
                                  ensure_ascii=False, separators=JSON_SEPARATORS)
            body = f"{TOOL_CALL_OPEN}\n{call_obj}\n{TOOL_CALL_CLOSE}"
            if t.thought:
                body = f"{t.thought}\n{body}"
            parts.append(_chatml_turn(ROLE_ASSISTANT, body))
            obs = t.observation if t.observation is not None else ""
            if len(obs) > _DEFAULT_OBS_TRIM:
                obs = obs[:_DEFAULT_OBS_TRIM] + OBS_TRIM_FMT.format(n=len(t.observation) - _DEFAULT_OBS_TRIM)
            resp_obj = json.dumps({KEY_NAME: t.action, KEY_RESULT: obs},
                                  ensure_ascii=False, separators=JSON_SEPARATORS)
            parts.append(_chatml_turn(ROLE_TOOL,
                                      f"{TOOL_RESPONSE_OPEN}\n{resp_obj}\n{TOOL_RESPONSE_CLOSE}"))
        elif t.schema_error:
            err_obj = json.dumps({KEY_ERROR: t.schema_error}, ensure_ascii=False,
                                 separators=JSON_SEPARATORS)
            parts.append(_chatml_turn(ROLE_TOOL,
                                      f"{TOOL_RESPONSE_OPEN}\n{err_obj}\n{TOOL_RESPONSE_CLOSE}"))
    return "".join(parts)


def _build_turn_prompt(system: str, user: str, turns: list[AgentTurn]) -> str:
    head = _chatml_turn(ROLE_SYSTEM, system) + _chatml_turn(ROLE_USER, user)
    return head + _render_history(turns) + f"{IM_START}{ROLE_ASSISTANT}\n"


def _collect_bytes(stream_gen):
    out = bytearray()
    stop = None
    for ev in stream_gen:
        k = ev.get("kind")
        if k == "token" or k == "fast_byte":
            b = ev.get("byte")
            if isinstance(b, int):
                out.append(b & 0xff)
        elif k == "stop":
            stop = ev.get("reason") or STOP_DEFAULT
            break
        elif k == "error":
            stop = f"error: {ev.get('message') or 'stream error'}"
            break
    return bytes(out), stop


def _strip_imend(s: str) -> str:
    i = s.find(IM_END)
    if i >= 0:
        s = s[:i]
    return s.rstrip()


def _parse_assistant_turn(raw_bytes: bytes):
    """Parse a raw assistant-turn byte stream into (thought, tool_call, answer)
    where tool_call is a dict {name, arguments} or None.

    Hermes spec: a <tool_call>...</tool_call> block means the model is
    invoking a tool; everything before it is treated as thought. Without a
    tool_call block, the entire turn (sans <|im_end|>) is the answer."""
    text = raw_bytes.decode("utf-8", errors="replace")
    text = _strip_imend(text)
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return (None, None, text or None, None)
    inner = m.group(1).strip()
    try:
        obj = json.loads(inner)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return (None, None, None, f"tool_call JSON parse error: {type(e).__name__}: {e}")
    if not isinstance(obj, dict) or KEY_NAME not in obj:
        return (None, None, None, f"tool_call missing {KEY_NAME!r} field")
    args = obj.get(KEY_ARGUMENTS)
    if args is None:
        args = obj.get(KEY_ARGS_ALT) or {}
    if not isinstance(args, dict):
        return (None, None, None, f"tool_call {KEY_ARGUMENTS!r} must be an object")
    pre = text[:m.start()].strip()
    thought = pre or None
    return (thought, {KEY_NAME: obj[KEY_NAME], KEY_ARGUMENTS: args}, None, None)


class AgentLoop:
    """Tool-using loop on top of a Brain-shaped backend.

    Required `backend` interface:
        backend.stream(prompt, temperature, top_k_sample, max_new,
                       addons_chain=None, constraint=None) -> generator
    """

    def __init__(self, backend, toolbox: Toolbox,
                 max_turns: int = _DEFAULT_MAX_TURNS,
                 max_new_per_turn: int = _DEFAULT_MAX_NEW,
                 temperature: float = _DEFAULT_TEMPERATURE,
                 top_k_sample: int = _DEFAULT_TOP_K,
                 best_of_n: int = _DEFAULT_BEST_OF_N,
                 seed_base: int = _DEFAULT_SEED_BASE):
        self.backend = backend
        self.toolbox = toolbox
        self.max_turns = int(max_turns)
        self.max_new = int(max_new_per_turn)
        self.temperature = float(temperature)
        self.top_k = int(top_k_sample)
        self.best_of_n = max(1, int(best_of_n))
        self.seed_base = int(seed_base)
        self.system_prompt = _build_system_prompt(toolbox)

    def _sample_turn(self, prompt, turn_i):
        import torch
        from inference.decode import StopOnConstraint
        candidates = []
        for k in range(self.best_of_n):
            torch.manual_seed(self.seed_base + turn_i * self.best_of_n + k)
            constraint = StopOnConstraint(IM_END_B)
            gen = self.backend.stream(prompt,
                                      temperature=self.temperature,
                                      top_k_sample=self.top_k,
                                      max_new=self.max_new,
                                      constraint=constraint)
            raw, stop = _collect_bytes(gen)
            candidates.append((raw, stop))
        # Pick first candidate that parses cleanly (has tool_call or non-empty answer).
        for raw, stop in candidates:
            thought, tc, ans, err = _parse_assistant_turn(raw)
            if err is None and (tc is not None or (ans and ans.strip())):
                return raw, stop, thought, tc, ans, None
        # Fall back to the longest candidate's parse for diagnostics.
        longest = max(candidates, key=lambda c: len(c[0]))
        thought, tc, ans, err = _parse_assistant_turn(longest[0])
        return longest[0], longest[1], thought, tc, ans, err

    def run_streaming(self, user_input: str):
        result = AgentResult()
        t_start = time.time()
        for turn_i in range(self.max_turns):
            yield {"kind": "turn_start", "turn": turn_i}
            turn = AgentTurn()
            t0 = time.time()
            prompt = _build_turn_prompt(self.system_prompt, user_input, result.turns)

            raw, stop, thought, tool_call, answer, err = self._sample_turn(prompt, turn_i)
            turn.raw_bytes = raw
            turn.stop_reason = stop
            turn.elapsed_s = time.time() - t0
            turn.thought = thought
            if thought:
                yield {"kind": "thought", "turn": turn_i, "text": thought}

            if err is not None:
                turn.schema_error = err
                yield {"kind": "schema_err", "turn": turn_i, "error": err,
                       "raw_head": raw[:_RAW_HEAD_CAP].decode("utf-8", errors="replace")}
                result.turns.append(turn)
                continue

            if tool_call is not None:
                action = tool_call[KEY_NAME]
                args = tool_call[KEY_ARGUMENTS]
                tool = self.toolbox.get(action)
                if tool is None:
                    turn.schema_error = f"unknown tool {action!r}; available: {self.toolbox.names()}"
                    yield {"kind": "schema_err", "turn": turn_i, "error": turn.schema_error}
                    result.turns.append(turn)
                    continue
                turn.action = action
                turn.args = args
                yield {"kind": "action", "turn": turn_i, "tool": action, "args": args}
                tool_t0 = time.time()
                turn.observation = tool.call(args)
                tool_dt = time.time() - tool_t0
                obs_view = turn.observation
                if len(obs_view) > _DEFAULT_OBS_TRIM:
                    obs_view = obs_view[:_DEFAULT_OBS_TRIM] + OBS_TRIM_VIEW
                yield {"kind": "observation", "turn": turn_i, "text": obs_view,
                       "elapsed_s": tool_dt}
                result.turns.append(turn)
                continue

            if answer is not None and answer.strip():
                turn.answer = answer
                yield {"kind": "answer", "turn": turn_i, "text": answer}
                result.turns.append(turn)
                result.final_answer = answer
                result.stop_reason = STOP_ANSWER
                break

            turn.schema_error = EMPTY_TURN_ERROR
            yield {"kind": "schema_err", "turn": turn_i, "error": turn.schema_error}
            result.turns.append(turn)

        if result.final_answer is None:
            result.stop_reason = result.stop_reason or STOP_MAX_TURNS
        result.total_elapsed_s = time.time() - t_start
        yield {"kind": "stop", "reason": result.stop_reason,
               "total_elapsed_s": result.total_elapsed_s,
               "turns": len(result.turns)}

    def run(self, user_input: str) -> AgentResult:
        res = AgentResult()
        t_start = time.time()
        for turn_i in range(self.max_turns):
            turn = AgentTurn()
            t0 = time.time()
            prompt = _build_turn_prompt(self.system_prompt, user_input, res.turns)

            raw, stop, thought, tool_call, answer, err = self._sample_turn(prompt, turn_i)
            turn.raw_bytes = raw
            turn.stop_reason = stop
            turn.elapsed_s = time.time() - t0
            turn.thought = thought

            if err is not None:
                turn.schema_error = err
                res.turns.append(turn)
                continue

            if tool_call is not None:
                action = tool_call[KEY_NAME]
                args = tool_call[KEY_ARGUMENTS]
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

            if answer is not None and answer.strip():
                turn.answer = answer
                res.turns.append(turn)
                res.final_answer = answer
                res.stop_reason = STOP_ANSWER
                break

            turn.schema_error = EMPTY_TURN_ERROR
            res.turns.append(turn)

        if res.final_answer is None:
            res.stop_reason = res.stop_reason or STOP_MAX_TURNS
        res.total_elapsed_s = time.time() - t_start
        return res
