# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Trainer-agnostic OOM recovery. Detects memory-exhaustion exceptions across
#   CUDA/MPS/CPU backends, bounds fallback depth so a misconfigured run does
#   not loop, and re-execs the trainer with caller-specified safe flags forced
#   on. The list of safe flags is plugin-specific and provided by the caller
#   (typically from the plugin's manifest.json), never assumed here.
# - Use as: catch the exception, call `is_oom_error` to filter, do emergency
#   ckpt, then call `reexec_with_flags(safe_flags)` which never returns.
# veritate_core/plugin/oom_recovery.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

# ------------------------------------------------------------------------------------
# Constants

# Substrings that identify a memory-exhaustion error across PyTorch backends.
# Matched case-insensitively against str(exc).
OOM_MARKERS = (
    "out of memory",
    "mps backend out of memory",
    "could not allocate",
    "cuda error: out of memory",
    "alloc failed",
)

# Env var carrying the running fallback count across re-execs. Bounded so a
# misconfigured run (e.g. fallback flag was already on) cannot loop forever.
COUNTER_ENV   = "VERITATE_OOM_FALLBACKS"
MAX_FALLBACKS = 3

# ------------------------------------------------------------------------------------
# Functions

def is_oom_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in OOM_MARKERS)


def fallback_count():
    raw = os.environ.get(COUNTER_ENV)
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def should_recover():
    return fallback_count() < MAX_FALLBACKS


def reexec_with_flags(safe_flags):
    """Replace the current process with a fresh interpreter running the same
    trainer, forcing each name in `safe_flags` ON in argv. Strips any
    `--no-<flag>` counterpart so the override sticks past apply_resume_overrides.
    Increments COUNTER_ENV so the new process knows the depth.

    `safe_flags`: iterable of flag names without leading `--` (plugin-specific;
    typically read from the plugin's manifest oom_safe_flags entry).

    Never returns."""
    argv = list(sys.argv[1:])
    for flag in safe_flags:
        no_form  = "--no-" + flag
        yes_form = "--" + flag
        argv = [a for a in argv
                if a != no_form and not a.startswith(no_form + "=")]
        if yes_form not in argv:
            argv.append(yes_form)
    env = os.environ.copy()
    env[COUNTER_ENV] = str(fallback_count() + 1)
    os.execve(sys.executable, [sys.executable, sys.argv[0]] + argv, env)
