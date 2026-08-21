# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The streaming request path is the one every Distillation interview call takes:
#   interview.ask always passes cancel_check, and every provider but Anthropic
#   declares supports_stream. Its HTTP status handling has to match the plain path
#   or a teacher-side refusal reads as silence.
# - urllib.error.HTTPError subclasses URLError and OSError, so an except clause
#   ordered after those swallows it. That is what these tests pin.
# - No network: urlopen is stubbed, so the retry sleeps are the only wall clock
#   and every case here fails before the first one.
# tests/mri/test_teacher_client_errors.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import urllib.error

import pytest
from teacher import client as client_mod

# ------------------------------------------------------------------------------------
# Constants

STREAM_URL = "https://teacher.invalid/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "hello"}]

# ------------------------------------------------------------------------------------
# Functions

def _http_error(code, body, headers=None):
    return urllib.error.HTTPError(STREAM_URL, code, "err", headers or {},
                                  io.BytesIO(body.encode("utf-8")))


@pytest.fixture
def raising_urlopen(monkeypatch):
    """Stub urlopen that raises a queued error and counts the attempts."""
    calls = []

    def install(err_factory):
        def fake(req, timeout=None, context=None):
            calls.append(req)
            raise err_factory()
        monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake)
        monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
        return calls

    return install


def _client(max_retries=5):
    return client_mod.Client("openai", model="gpt-4o", base_url="https://teacher.invalid",
                             api_key="k", max_retries=max_retries)


def test_stream_auth_failure_raises_teacher_auth_error_without_retrying(raising_urlopen):
    """A 401 on the streaming path is fatal at once, not retried to exhaustion."""
    calls = raising_urlopen(lambda: _http_error(401, '{"error":"invalid api key"}'))
    with pytest.raises(client_mod.TeacherAuthError) as caught:
        _client().complete(MESSAGES, cancel_check=lambda: False)
    assert len(calls) == 1
    assert "invalid api key" in str(caught.value)


def test_stream_bad_request_reports_the_server_body(raising_urlopen):
    """A non-retryable 400 surfaces the teacher's own error text, once."""
    calls = raising_urlopen(lambda: _http_error(400, '{"error":"max_tokens too large"}'))
    with pytest.raises(client_mod.TeacherError) as caught:
        _client().complete(MESSAGES, cancel_check=lambda: False)
    assert len(calls) == 1
    assert "max_tokens too large" in str(caught.value)


def test_stream_server_error_exhausts_retries_as_unavailable(raising_urlopen):
    """A repeated 500 is retried, then classified rather than leaking urllib's error."""
    calls = raising_urlopen(lambda: _http_error(500, "upstream boom"))
    with pytest.raises(client_mod.TeacherUnavailableError) as caught:
        _client(max_retries=2).complete(MESSAGES, cancel_check=lambda: False)
    assert len(calls) == 3
    assert "upstream boom" in str(caught.value)


def test_stream_rate_limit_honours_retry_after(raising_urlopen, monkeypatch):
    """429 waits the header's seconds instead of the backoff curve, then reports the limit."""
    raising_urlopen(lambda: _http_error(429, "slow down", {"Retry-After": "7"}))
    waited = []
    monkeypatch.setattr(client_mod.time, "sleep", waited.append)
    with pytest.raises(client_mod.TeacherRateLimitError):
        _client(max_retries=1).complete(MESSAGES, cancel_check=lambda: False)
    assert waited == [7.0]
