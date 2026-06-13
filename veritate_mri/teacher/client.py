# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sync HTTP client over urllib.request. uniform .complete() across all
#   providers; differences encoded as data in the registry (auth header,
#   system style, response path). retries on transient status with exponential
#   backoff + jitter; respects Retry-After int seconds.
# veritate_mri/teacher/client.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import random
import ssl
import time
import urllib.error
import urllib.request

from .providers import (
    DEFAULT_BACKOFF_BASE_S,
    DEFAULT_BACKOFF_MAX_S,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    RETRY_STATUS,
    default_model_for,
    get_provider,
)

# ------------------------------------------------------------------------------------
# Constants

# certifi-backed context so HTTPS verification works on Python builds that don't
# resolve a system CA bundle (macOS). Mirrors runtime.heartbeat / runtime.ai_assist.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

_JSON_CONTENT_TYPE = "application/json"
_AUTH_STATUS = (401, 403)
_ERR_BODY_MAX = 300

# ------------------------------------------------------------------------------------
# Functions

class TeacherError(Exception):
    pass


class TeacherAuthError(TeacherError):
    pass


class TeacherRateLimitError(TeacherError):
    pass


class TeacherUnavailableError(TeacherError):
    pass


class TeacherCancelled(TeacherError):
    pass


def _extract(data, path):
    cur = data
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                raise TeacherError(f"response path missing at index {key}")
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                raise TeacherError(f"response path missing key {key}")
            cur = cur[key]
    if not isinstance(cur, str):
        raise TeacherError("response text not a string")
    return cur


def _split_system(messages, style):
    if style != "field":
        return None, list(messages)
    sys_text = None
    rest = []
    for m in messages:
        if m.get("role") == "system" and sys_text is None:
            sys_text = m.get("content", "")
        else:
            rest.append(m)
    return sys_text, rest


def _build_payload(provider, model, messages, temperature, max_tokens, system):
    msgs = list(messages)
    if system is not None:
        msgs = [{"role": "system", "content": system}] + msgs
    sys_field, msgs = _split_system(msgs, provider["system_message_style"])
    body = {
        "model": model,
        provider["messages_key"]: msgs,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if sys_field is not None:
        body["system"] = sys_field
    return body


def _build_headers(provider, api_key):
    headers = {"Content-Type": _JSON_CONTENT_TYPE}
    for k, v in provider["extra_headers"].items():
        headers[k] = v
    if provider["auth_header"] and api_key:
        headers[provider["auth_header"]] = provider["auth_prefix"] + api_key
    return headers


def _parse_retry_after(value):
    if not value:
        return None
    try:
        return float(int(value))
    except (TypeError, ValueError):
        return None


def _backoff(attempt):
    delay = min(DEFAULT_BACKOFF_BASE_S * (2 ** attempt), DEFAULT_BACKOFF_MAX_S)
    return delay + random.uniform(0, delay * 0.25)


def _err_body(e):
    try:
        return e.read().decode("utf-8", "replace")[:_ERR_BODY_MAX].strip()
    except (OSError, ValueError):
        return ""


class Client:
    def __init__(self, provider_id, model=None, base_url=None, api_key=None,
                 timeout_s=DEFAULT_TIMEOUT_S, max_retries=DEFAULT_MAX_RETRIES):
        self.provider = get_provider(provider_id)
        self.model = model or default_model_for(provider_id)
        self.base_url = (base_url or self.provider["base_url"]).rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def complete(self, messages, temperature=DEFAULT_TEMPERATURE,
                 max_tokens=DEFAULT_MAX_TOKENS, system=None, cancel_check=None):
        if not self.model:
            raise TeacherError("no model set")
        url = self.base_url + self.provider["chat_path"]
        payload = _build_payload(self.provider, self.model, messages,
                                 temperature, max_tokens, system)
        headers = _build_headers(self.provider, self.api_key)
        # Cancellable path: stream so the caller can abort between tokens; closing
        # the connection makes the server stop generating (frees the GPU). Only
        # OpenAI-style chat endpoints (every local provider) emit the delta format.
        if cancel_check is not None and self.provider["chat_path"].endswith("chat/completions"):
            return self._complete_stream(url, payload, headers, cancel_check)
        data = json.dumps(payload).encode("utf-8")
        last_status = None
        last_err = None
        last_detail = ""
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX) as resp:
                    status = getattr(resp, "status", 200)
                    body = resp.read()
                    if status < 300:
                        parsed = json.loads(body.decode("utf-8"))
                        return _extract(parsed, self.provider["response_text_path"])
                    last_status = status
            except urllib.error.HTTPError as e:
                last_status = e.code
                last_detail = _err_body(e)
                if e.code in _AUTH_STATUS:
                    raise TeacherAuthError(f"auth failed: {e.code} {last_detail}".rstrip())
                if e.code not in RETRY_STATUS:
                    raise TeacherError(f"http error: {e.code} {last_detail}".rstrip())
                last_err = e
            except urllib.error.URLError as e:
                last_err = e
            except (TimeoutError, ConnectionError) as e:
                last_err = e
            if attempt >= self.max_retries:
                break
            wait = None
            if last_err is not None and isinstance(last_err, urllib.error.HTTPError):
                wait = _parse_retry_after(last_err.headers.get("Retry-After") if last_err.headers else None)
            if wait is None:
                wait = _backoff(attempt)
            time.sleep(wait)
        if last_status == 429:
            raise TeacherRateLimitError(f"rate limit exhausted {last_detail}".rstrip())
        if last_status is not None and 500 <= last_status < 600:
            raise TeacherUnavailableError(f"upstream unavailable: {last_status} {last_detail}".rstrip())
        raise TeacherError(f"request failed: status={last_status} err={last_err} {last_detail}".rstrip())

    def _complete_stream(self, url, payload, headers, cancel_check):
        # Retry transients (cold-load empty stream, connection blips) the same way
        # the non-streaming path does. A set cancel flag aborts without retrying.
        body = json.dumps({**payload, "stream": True}).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries + 1):
            if cancel_check():
                raise TeacherCancelled("cancelled")
            try:
                return self._stream_once(url, body, headers, cancel_check)
            except TeacherCancelled:
                raise
            except (TeacherError, urllib.error.URLError, OSError) as e:
                last_err = e
            if attempt >= self.max_retries or cancel_check():
                break
            time.sleep(_backoff(attempt))
        raise last_err if last_err is not None else TeacherError("empty stream response")

    def _stream_once(self, url, body, headers, cancel_check):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        parts = []
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX) as resp:
            for raw in resp:
                if cancel_check():
                    raise TeacherCancelled("cancelled")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    parts.append(delta)
        if not parts:
            raise TeacherError("empty stream response")
        return "".join(parts)

    def unload(self, model=None):
        """Drop a model from the server's memory (Ollama native keep_alive=0).
        Best-effort: no-op for providers without the native unload endpoint."""
        m = model or self.model
        if not m:
            return False
        body = json.dumps({"model": m, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/api/generate", data=body,
                                     headers={"Content-Type": _JSON_CONTENT_TYPE}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self):
        """GET the provider's model-listing endpoint. Returns a list of served
        model-id strings, None when the provider has no listing endpoint or it
        404s. Raises TeacherAuthError / TeacherUnavailableError so a probe can
        tell a bad key or unreachable host from a missing listing API."""
        path = self.provider.get("models_path") or ""
        if not path:
            return None
        headers = _build_headers(self.provider, self.api_key)
        headers.pop("Content-Type", None)
        req = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in _AUTH_STATUS:
                raise TeacherAuthError(f"auth failed: {e.code}")
            if e.code == 404:
                return None
            raise TeacherError(f"models list http {e.code}")
        except (urllib.error.URLError, OSError) as e:
            raise TeacherUnavailableError(f"unreachable: {e}")
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        arr = data.get(self.provider.get("models_array", "data")) or []
        idk = self.provider.get("models_id", "id")
        return [str(m[idk]) for m in arr if isinstance(m, dict) and m.get(idk)]


def complete(provider_id, model, messages, **opts):
    timeout_s = opts.pop("timeout_s", DEFAULT_TIMEOUT_S)
    max_retries = opts.pop("max_retries", DEFAULT_MAX_RETRIES)
    base_url = opts.pop("base_url", None)
    api_key = opts.pop("api_key", None)
    c = Client(provider_id, model=model, base_url=base_url, api_key=api_key,
               timeout_s=timeout_s, max_retries=max_retries)
    return c.complete(messages, **opts)
