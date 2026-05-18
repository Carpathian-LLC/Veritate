# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests for the HTTP client. mocks urllib.request.urlopen so no live calls.
# tests/teacher/test_client.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import json
from unittest.mock import patch

import pytest
import urllib.error

from veritate_mri.teacher.client import (
    Client,
    TeacherAuthError,
    TeacherRateLimitError,
    TeacherUnavailableError,
)

# ------------------------------------------------------------------------------------
# Constants

_OPENAI_BODY = {"choices": [{"message": {"content": "ok"}}]}
_ANTHROPIC_BODY = {"content": [{"text": "ok"}]}

# ------------------------------------------------------------------------------------
# Functions

class _MockResp:
    def __init__(self, status, body):
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_open_returning(*resps):
    calls = {"count": 0, "requests": []}

    def fake(req, timeout=None):
        calls["requests"].append(req)
        i = calls["count"]
        calls["count"] += 1
        r = resps[min(i, len(resps) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    return fake, calls


def _http_err(code, msg="err"):
    return urllib.error.HTTPError(url="http://x", code=code, msg=msg, hdrs={}, fp=io.BytesIO(b""))


def test_openai_shape_request():
    """openai client sends Authorization bearer header and messages key with inline system."""
    fake, calls = _mock_open_returning(_MockResp(200, _OPENAI_BODY))
    c = Client("openai", model="gpt-4o", api_key="KEY", max_retries=0)
    with patch("urllib.request.urlopen", side_effect=fake):
        out = c.complete([{"role": "user", "content": "hi"}], system="be brief")
    assert out == "ok"
    req = calls["requests"][0]
    assert req.headers.get("Authorization") == "Bearer KEY"
    body = json.loads(req.data.decode("utf-8"))
    assert "messages" in body
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "be brief"
    assert "system" not in body


def test_anthropic_shape_request():
    """anthropic client sends x-api-key header and extracts system to top-level field."""
    fake, calls = _mock_open_returning(_MockResp(200, _ANTHROPIC_BODY))
    c = Client("anthropic", model="claude-opus-4-7", api_key="KEY", max_retries=0)
    with patch("urllib.request.urlopen", side_effect=fake):
        out = c.complete([{"role": "user", "content": "hi"}], system="be brief")
    assert out == "ok"
    req = calls["requests"][0]
    assert req.headers.get("X-api-key") == "KEY"
    body = json.loads(req.data.decode("utf-8"))
    assert body["system"] == "be brief"
    for m in body["messages"]:
        assert m["role"] != "system"


def test_auth_error_on_401():
    """401 response raises TeacherAuthError."""
    fake, _ = _mock_open_returning(_http_err(401))
    c = Client("openai", model="gpt-4o", api_key="bad", max_retries=2)
    with patch("urllib.request.urlopen", side_effect=fake):
        with pytest.raises(TeacherAuthError):
            c.complete([{"role": "user", "content": "hi"}])


def test_retry_after_429_then_success():
    """429 then 200 succeeds after retry."""
    fake, calls = _mock_open_returning(_http_err(429), _MockResp(200, _OPENAI_BODY))
    c = Client("openai", model="gpt-4o", api_key="KEY", max_retries=3)
    with patch("urllib.request.urlopen", side_effect=fake):
        with patch("time.sleep", return_value=None):
            out = c.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert calls["count"] == 2


def test_rate_limit_exhausted():
    """repeated 429 raises TeacherRateLimitError after retries exhausted."""
    fake, _ = _mock_open_returning(_http_err(429), _http_err(429), _http_err(429))
    c = Client("openai", model="gpt-4o", api_key="KEY", max_retries=2)
    with patch("urllib.request.urlopen", side_effect=fake):
        with patch("time.sleep", return_value=None):
            with pytest.raises(TeacherRateLimitError):
                c.complete([{"role": "user", "content": "hi"}])


def test_retry_after_500_then_success():
    """500 then 200 succeeds after retry."""
    fake, calls = _mock_open_returning(_http_err(500), _MockResp(200, _OPENAI_BODY))
    c = Client("openai", model="gpt-4o", api_key="KEY", max_retries=3)
    with patch("urllib.request.urlopen", side_effect=fake):
        with patch("time.sleep", return_value=None):
            out = c.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert calls["count"] == 2


def test_unavailable_after_retries():
    """repeated 500 raises TeacherUnavailableError after retries exhausted."""
    fake, _ = _mock_open_returning(_http_err(500), _http_err(500), _http_err(500))
    c = Client("openai", model="gpt-4o", api_key="KEY", max_retries=2)
    with patch("urllib.request.urlopen", side_effect=fake):
        with patch("time.sleep", return_value=None):
            with pytest.raises(TeacherUnavailableError):
                c.complete([{"role": "user", "content": "hi"}])
