# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Toolbox: the registry of agentic tools. Each tool is a small dataclass +
#   an execute function that takes a dict of args and returns a string
#   observation. Tools must NOT raise on bad args; they should return a
#   string starting with "error: ..." so the model can read it and recover.
# - The default toolbox bundles calculator + filesystem-read + web-fetch +
#   BM25 retriever. Other tools can be registered at runtime via
#   toolbox.register(Tool(...)).
# veritate_mri/agent/tools/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------------------------
# Constants

# Prompt-block templates. This is what the model reads when deciding which tool
# to call, so the wording is part of the agent contract (see agent/loop.py).
PROMPT_BLOCK_HEADER = "Available tools:"
PROMPT_TOOL_FMT     = "- {name}: {description}"
PROMPT_ARG_FMT      = "    {name} ({type}{required}): {doc}"
PROMPT_REQUIRED_TAG = " (required)"
PROMPT_DEFAULT_TYPE = "string"

# Tools return an observation string; failures are reported in-band with this
# prefix so the model can read the error and recover instead of crashing.
ERROR_PREFIX = "error: "

# ------------------------------------------------------------------------------------
# Functions


@dataclass
class Tool:
    """A single tool the agent can invoke.

    - name: short identifier the model emits in the "action" field.
    - description: one-line prompt-facing description.
    - args_schema: dict mapping arg_name -> {type: str, required: bool, doc: str}.
                   Used for prompt construction and post-parse validation.
    - execute: callable taking parsed args dict; returns observation string.
               MUST handle bad args gracefully (return "error: ..." rather than raise).
    """
    name:         str
    description:  str
    args_schema:  dict[str, dict[str, Any]] = field(default_factory=dict)
    execute:      Callable[[dict[str, Any]], str] | None = None

    def call(self, args: dict[str, Any]) -> str:
        if self.execute is None:
            return f"{ERROR_PREFIX}tool '{self.name}' has no executor"
        try:
            out = self.execute(args)
            if not isinstance(out, str):
                out = str(out)
            return out
        except Exception as e:
            return f"{ERROR_PREFIX}{type(e).__name__}: {e}"


class Toolbox:
    """Registry of tools. Lookup by name. Prompt-block generator."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool name collision: {tool.name!r}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def prompt_block(self) -> str:
        """Human-readable list of tools for prompt injection."""
        lines = [PROMPT_BLOCK_HEADER]
        for name in self.names():
            t = self._tools[name]
            lines.append(PROMPT_TOOL_FMT.format(name=name, description=t.description))
            for arg_name, meta in (t.args_schema or {}).items():
                lines.append(PROMPT_ARG_FMT.format(
                    name=arg_name,
                    type=meta.get("type", PROMPT_DEFAULT_TYPE),
                    required=PROMPT_REQUIRED_TAG if meta.get("required") else "",
                    doc=meta.get("doc", "")))
        return "\n".join(lines)


def build_default_toolbox(corpus_path: str | None = None,
                          fs_root: str | None = None) -> Toolbox:
    """Bundle calculator + fs_read + fetch + (optional) retriever."""
    from .calculator import TOOL as CALC
    from .fetch import TOOL as FETCH
    from .filesystem import make_tool as fs_make
    from .retriever import make_tool as retr_make

    tb = Toolbox([CALC, FETCH])
    if fs_root:
        tb.register(fs_make(fs_root))
    if corpus_path:
        tb.register(retr_make(corpus_path))
    return tb


__all__ = ["Tool", "Toolbox", "build_default_toolbox"]
