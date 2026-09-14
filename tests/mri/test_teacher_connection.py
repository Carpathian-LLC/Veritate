# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the Settings tab's teacher Test button: stage 1 lists models (auth and reachability
#   fail here, a model the endpoint does not serve fails here, a lone served model is
#   adopted, a provider without a listing endpoint skips ahead), stage 2 is one tiny
#   completion judged by its shape. The client is a stub: no network.
# tests/mri/test_teacher_connection.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from teacher import test_connection as tc
from teacher.client import TeacherAuthError, TeacherError, TeacherRateLimitError, TeacherUnavailableError

# ------------------------------------------------------------------------------------
# Functions


def _client_class(served=None, listing_error=None, reply="pong", complete_error=None, selectable=True):
    class FakeClient:
        def __init__(self, provider_id, model=None, base_url=None, api_key=None, timeout_s=None, max_retries=None):
            self.model = model
            self.provider = {"model_selectable": selectable}

        def list_models(self):
            if listing_error is not None:
                raise listing_error
            return served

        def complete(self, messages, temperature=None, max_tokens=None):
            if complete_error is not None:
                raise complete_error
            return reply
    return FakeClient


def test_a_served_model_that_answers_is_ok(monkeypatch):
    monkeypatch.setattr(tc, "Client", _client_class(served=["gpt-x", "models/gemma"]))
    out = tc.test("prov", model="gemma")
    assert out["ok"] and out["error"] is None and out["model"] == "gemma"


@pytest.mark.parametrize("err, prefix", [(TeacherAuthError("bad key"), "auth:"),
                                         (TeacherUnavailableError("down"), "unavailable:")])
def test_stage_one_failures_stop_before_any_inference(monkeypatch, err, prefix):
    monkeypatch.setattr(tc, "Client", _client_class(listing_error=err, complete_error=RuntimeError("must not run")))
    out = tc.test("prov", model="m")
    assert not out["ok"] and out["error"].startswith(prefix)


def test_a_model_the_endpoint_does_not_serve_is_named_with_what_is(monkeypatch):
    monkeypatch.setattr(tc, "Client", _client_class(served=["a", "b"]))
    out = tc.test("prov", model="zzz")
    assert not out["ok"] and "not served" in out["error"] and "a, b" in out["error"]


def test_a_lone_served_model_is_adopted_and_a_missing_listing_is_skipped(monkeypatch):
    monkeypatch.setattr(tc, "Client", _client_class(served=["only-one"]))
    assert tc.test("prov", model="whatever")["model"] == "only-one"
    monkeypatch.setattr(tc, "Client", _client_class(listing_error=TeacherError("no listing endpoint")))
    assert tc.test("prov", model="m")["ok"]


@pytest.mark.parametrize("err, prefix", [(TeacherRateLimitError("slow"), "rate_limit:"),
                                         (TeacherError("boom"), "error:"), (ValueError("odd"), "unexpected:")])
def test_stage_two_failures_are_classified(monkeypatch, err, prefix):
    monkeypatch.setattr(tc, "Client", _client_class(served=None, complete_error=err))
    out = tc.test("prov", model="m")
    assert not out["ok"] and out["error"].startswith(prefix)


def test_a_reply_without_text_is_not_ok(monkeypatch):
    monkeypatch.setattr(tc, "Client", _client_class(served=None, reply=None))
    assert tc.test("prov", model="m")["error"] == "no content in response"


def test_list_models_swallows_teacher_errors_into_an_empty_list(monkeypatch):
    monkeypatch.setattr(tc, "Client", _client_class(listing_error=TeacherError("x")))
    assert tc.list_models("prov") == []
