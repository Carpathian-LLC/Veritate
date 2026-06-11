# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - two-tier connection probe used by the Settings Test button, mirroring the
#   gateway health-check standard: stage 1 lists models (reachability + auth +
#   served-model check, zero inference; skipped when no listing endpoint), stage
#   2 sends a minimal completion judged by response shape, not HTTP status.
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

_PING_MESSAGES = [{"role": "user", "content": "Reply with the single word: pong"}]
# Cap output so the probe returns fast instead of generating a full reply, which
# against a large local model can outlast the timeout. No retries; the timeout is
# a hard ceiling generous enough for a cold model load.
_PING_MAX_TOKENS = 16
_PROBE_TIMEOUT_S = 30
_PROBE_MAX_RETRIES = 0
_AVAILABLE_SHOWN = 8

# ------------------------------------------------------------------------------------
# Functions

def _ms(t0):
    return int((time.time() - t0) * 1000)


def _served_match(model, served):
    """Lenient membership: exact id, or basename match to tolerate prefixed ids
    (e.g. 'models/gemini-2.0-flash' vs 'gemini-2.0-flash')."""
    if not model or not served:
        return True
    base = model.split("/")[-1]
    return any(s == model or s.split("/")[-1] == base for s in served)


def list_models(provider_id, base_url=None, api_key=None):
    c = Client(provider_id, base_url=base_url, api_key=api_key,
               timeout_s=_PROBE_TIMEOUT_S, max_retries=_PROBE_MAX_RETRIES)
    try:
        return c.list_models() or []
    except TeacherError:
        return []


def test(provider_id, model=None, base_url=None, api_key=None):
    c = Client(provider_id, model=model, base_url=base_url, api_key=api_key,
               timeout_s=_PROBE_TIMEOUT_S, max_retries=_PROBE_MAX_RETRIES)
    out = {"ok": False, "latency_ms": 0, "error": None, "model": c.model or ""}
    t0 = time.time()
    # Stage 1: listing call. Reachability + auth without inference. A missing
    # listing endpoint (None) is skipped; a bad key or dead host fails here.
    try:
        served = c.list_models()
    except TeacherAuthError as e:
        out["error"] = f"auth: {e}"; out["latency_ms"] = _ms(t0); return out
    except TeacherUnavailableError as e:
        out["error"] = f"unavailable: {e}"; out["latency_ms"] = _ms(t0); return out
    except TeacherError:
        served = None
    if served:
        if len(served) == 1:
            out["model"] = served[0]
        if c.provider.get("model_selectable", True) and not _served_match(c.model, served):
            shown = ", ".join(served[:_AVAILABLE_SHOWN])
            out["error"] = f"model '{c.model}' not served; available: {shown}"
            out["latency_ms"] = _ms(t0)
            return out
    # Stage 2: minimal completion, judged by parsed response shape.
    try:
        text = c.complete(_PING_MESSAGES, temperature=None, max_tokens=_PING_MAX_TOKENS)
        if isinstance(text, str):
            out["ok"] = True
        else:
            out["error"] = "no content in response"
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
    out["latency_ms"] = _ms(t0)
    return out
