# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Agentic tool loop. ReAct-style decoding under JSON-constrained output.
#   The model emits one of two JSON shapes per turn:
#     {"action": "<tool>", "args": {...}}     # invokes a tool, observation appended
#     {"answer": "<final answer>"}            # terminates the loop
#   The decode is JSON-constrained (build-7) so every byte is grammar-valid by
#   construction. Schema validity (the action field must name a real tool,
#   args must match the tool's schema) is enforced by the loop after parse.
# - Tools live in veritate_mri/inference/agent/tools/ as self-contained
#   modules. Each tool exports: name, description, args_schema,
#   execute(args) -> str.
# - The loop is model-agnostic: it consumes any object that implements the
#   Brain.stream(prompt, constraint=...) interface.
# veritate_mri/inference/agent/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from .loop import AgentLoop, AgentTurn, AgentResult
from .tools import build_default_toolbox, Tool, Toolbox

# ------------------------------------------------------------------------------------
# Constants

__all__ = [
    "AgentLoop", "AgentTurn", "AgentResult",
    "Tool", "Toolbox", "build_default_toolbox",
]
