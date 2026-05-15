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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ------------------------------------------------------------------------------------
# Constants

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
    args_schema:  Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execute:      Optional[Callable[[Dict[str, Any]], str]] = None

    def call(self, args: Dict[str, Any]) -> str:
        if self.execute is None:
            return f"error: tool '{self.name}' has no executor"
        try:
            out = self.execute(args)
            if not isinstance(out, str):
                out = str(out)
            return out
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


class Toolbox:
    """Registry of tools. Lookup by name. Prompt-block generator."""

    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool name collision: {tool.name!r}")
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def prompt_block(self) -> str:
        """Human-readable list of tools for prompt injection."""
        lines = ["Available tools:"]
        for name in self.names():
            t = self._tools[name]
            lines.append(f"- {name}: {t.description}")
            for arg_name, meta in (t.args_schema or {}).items():
                req = " (required)" if meta.get("required") else ""
                lines.append(f"    {arg_name} ({meta.get('type', 'string')}{req}): {meta.get('doc', '')}")
        return "\n".join(lines)


def build_default_toolbox(corpus_path: Optional[str] = None,
                          fs_root: Optional[str] = None) -> Toolbox:
    """Bundle calculator + fs_read + fetch + (optional) retriever."""
    from .calculator import TOOL as CALC
    from .filesystem import make_tool as fs_make
    from .fetch import TOOL as FETCH
    from .retriever import make_tool as retr_make

    tb = Toolbox([CALC, FETCH])
    if fs_root:
        tb.register(fs_make(fs_root))
    if corpus_path:
        tb.register(retr_make(corpus_path))
    return tb


__all__ = ["Tool", "Toolbox", "build_default_toolbox"]
