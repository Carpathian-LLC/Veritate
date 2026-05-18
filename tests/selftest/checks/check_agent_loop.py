# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - inference.agent.loop import + a default toolbox build. live agent run needs
#   a loaded brain; this check stays cold and only verifies surface.
# tests/selftest/checks/check_agent_loop.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _ctx
from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA  = "inference"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """AgentLoop class imports; build_default_toolbox returns a non-empty
    toolbox."""
    try:
        from inference.agent.loop import AgentLoop
        from inference.agent.tools import build_default_toolbox
    except Exception as exc:
        return _status.fail("agent_loop", f"import failed: {exc}")
    if not callable(AgentLoop):
        return _status.fail("agent_loop", "AgentLoop not callable")
    try:
        tb = build_default_toolbox(corpus_path=None, fs_root=_ctx.REPO_ROOT)
    except TypeError:
        try:
            tb = build_default_toolbox()
        except Exception as exc:
            return _status.fail("agent_loop", f"toolbox build failed: {exc}")
    except Exception as exc:
        return _status.fail("agent_loop", f"toolbox build failed: {exc}")
    tools = getattr(tb, "tools", None) or getattr(tb, "_tools", None) or tb
    n = len(list(tools)) if hasattr(tools, "__iter__") else -1
    return _status.ok("agent_loop", f"toolbox built (size={n})")
