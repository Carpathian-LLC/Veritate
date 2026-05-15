# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - one-shot connection probe. sends a tiny ping and returns ok/latency/error.
#   used by the Settings page Test button.
# veritate_mri/teacher/test_connection.py
# ------------------------------------------------------------------------------------
# Imports:

import time

from .client import (
    Client,
    TeacherAuthError,
    TeacherError,
    TeacherRateLimitError,
    TeacherUnavailableError,
)

# ------------------------------------------------------------------------------------
# Constants

_PING_MESSAGES = [{"role": "user", "content": "ping"}]
_PING_MAX_TOKENS = 4

# ------------------------------------------------------------------------------------
# Functions

def test(provider_id, model=None, base_url=None, api_key=None):
    c = Client(provider_id, model=model, base_url=base_url, api_key=api_key)
    out = {"ok": False, "latency_ms": 0, "error": None, "model": c.model or ""}
    t0 = time.time()
    try:
        c.complete(_PING_MESSAGES, max_tokens=_PING_MAX_TOKENS)
        out["ok"] = True
    except TeacherAuthError as e:
        out["error"] = f"auth: {e}"
    except TeacherRateLimitError as e:
        out["error"] = f"rate_limit: {e}"
    except TeacherUnavailableError as e:
        out["error"] = f"unavailable: {e}"
    except TeacherError as e:
        out["error"] = f"error: {e}"
    except Exception as e:
        out["error"] = f"unexpected: {e}"
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out
