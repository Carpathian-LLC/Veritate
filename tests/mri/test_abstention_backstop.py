# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression test for the inference-side abstention backstop wired into
#   /v1/chat/completions via _generate_local. Env-gated (VERITATE_ABSTENTION_GATE),
#   off by default so no arch is destabilised. Arch-agnostic: pure Python string
#   heuristic, identical behavior on win32/darwin/linux.
# tests/mri/test_abstention_backstop.py
# ------------------------------------------------------------------------------------
# Imports


import pytest
from inference.decode import abstention

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.delenv(abstention.ENV_FLAG_NAME, raising=False)
    yield


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv(abstention.ENV_FLAG_NAME, "on")
    yield


def test_gate_off_by_default_passes_through(gate_off):
    """When the env flag is unset, the backstop is a no-op: no arch loses default behavior."""
    assert abstention.wrap_response_text("aaaaaaaaaa") == "aaaaaaaaaa"
    assert abstention.wrap_response_text("Hello world!") == "Hello world!"
    assert abstention.wrap_response_text("") == ""


def test_gate_on_preserves_short_valid_answers(gate_on):
    """Short answers like math replies must NOT be gated: the model's zero-facts stance is fine."""
    assert abstention.wrap_response_text("8.") == "8."
    assert abstention.wrap_response_text("Yes.") == "Yes."
    assert abstention.wrap_response_text("Hi.") == "Hi."


def test_gate_on_preserves_normal_responses(gate_on):
    """Normal fluent responses pass through unchanged."""
    r = "Hello. How can I help you today?"
    assert abstention.wrap_response_text(r) == r


def test_gate_on_replaces_repetition_loop(gate_on):
    """A repetition-loop response is replaced with the abstention template."""
    loop = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    out = abstention.wrap_response_text(loop)
    assert out == abstention.ABSTAIN_TEMPLATE.decode("utf-8")


def test_gate_on_preserves_native_idk(gate_on):
    """When the model already said IDK naturally, we keep it (not double-replace)."""
    native = "I don't know that answer."
    assert abstention.wrap_response_text(native) == native


def test_env_flag_variants(monkeypatch):
    """Env flag accepts several truthy spellings."""
    r = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    for truthy in ("1", "true", "on", "yes", "TRUE", "On"):
        monkeypatch.setenv(abstention.ENV_FLAG_NAME, truthy)
        assert abstention.wrap_response_text(r) == abstention.ABSTAIN_TEMPLATE.decode("utf-8")
    for falsy in ("", "0", "false", "off", "no"):
        monkeypatch.setenv(abstention.ENV_FLAG_NAME, falsy)
        assert abstention.wrap_response_text(r) == r


def test_gate_on_handles_none(gate_on):
    """A None response is returned unchanged (no crash)."""
    assert abstention.wrap_response_text(None) is None


def test_streaming_gate_still_works(gate_on):
    """The AbstentionGate streaming class is independent of the env flag; unaffected."""
    g = abstention.make_gate(threshold=0.5, warmup=4, enabled=True)
    for _ in range(4):
        g.observe(65, 0.1)  # low top-1 → abstain
    assert g.decision() == abstention.DECISION_ABSTAIN

    g2 = abstention.make_gate(threshold=0.5, warmup=4, enabled=True)
    for _ in range(4):
        g2.observe(65, 0.9)  # high top-1 → proceed
    assert g2.decision() == abstention.DECISION_PROCEED
