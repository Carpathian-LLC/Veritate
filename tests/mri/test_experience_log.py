# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers inference/experience.py, the serving-exchange log that feeds sleep
#   consolidation (IDEA 20 T3). Pins: byte-lossless round-trip through base64,
#   the event-stream wrapper accumulating token/fast_byte bytes and writing one
#   record per exchange (including on a client disconnect mid-stream), empty
#   outputs skipped, the kill switch env, and the never-raise contract (append
#   returns False instead of raising when the root is unwritable).
# tests/mri/test_experience_log.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json

import pytest
from inference import experience

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture()
def exp_root(tmp_path, monkeypatch):
    root = tmp_path / "experience"
    monkeypatch.setattr(experience, "EXPERIENCE_ROOT", str(root))
    monkeypatch.delenv(experience.ENABLED_ENV, raising=False)
    return root


def _records(root):
    out = []
    for p in sorted(root.glob("*.jsonl")):
        out += [json.loads(ln) for ln in p.read_text().splitlines()]
    return out


def test_append_round_trips_raw_bytes(exp_root):
    prompt = bytes(range(256))
    output = b"\x00\xff hello \n"
    assert experience.append_exchange("wren1_3", prompt, output, meta={"backend": "pytorch"})
    (rec,) = _records(exp_root)
    assert base64.b64decode(rec["prompt_b64"]) == prompt
    assert base64.b64decode(rec["output_b64"]) == output
    assert rec["model"] == "wren1_3" and rec["meta"] == {"backend": "pytorch"}


def test_empty_output_is_skipped(exp_root):
    assert not experience.append_exchange("m", b"prompt", b"")
    assert not _records(exp_root)


def test_record_events_accumulates_and_writes_once(exp_root):
    evs = [{"kind": "meta"},
           {"kind": "token", "byte": 104},
           {"kind": "fast_byte", "byte": 105},
           {"kind": "stop", "reason": "x"}]
    seen = list(experience.record_events(iter(evs), model="m", prompt="p"))
    assert seen == evs                       # pass-through untouched
    (rec,) = _records(exp_root)
    assert base64.b64decode(rec["output_b64"]) == b"hi"
    assert base64.b64decode(rec["prompt_b64"]) == b"p"


def test_record_events_writes_on_disconnect(exp_root):
    """A client that stops reading mid-stream still leaves the partial reply
    in the log: partial experience is still experience."""
    evs = ({"kind": "token", "byte": b} for b in b"abcdef")
    gen = experience.record_events(evs, model="m", prompt="p")
    next(gen), next(gen)
    gen.close()                              # GeneratorExit path
    (rec,) = _records(exp_root)
    assert base64.b64decode(rec["output_b64"]) == b"ab"


def test_kill_switch_env(exp_root, monkeypatch):
    monkeypatch.setenv(experience.ENABLED_ENV, "0")
    assert not experience.append_exchange("m", b"p", b"out")
    assert not _records(exp_root)


def test_append_never_raises_on_unwritable_root(exp_root, monkeypatch):
    monkeypatch.setattr(experience, "EXPERIENCE_ROOT",
                        str(exp_root / "\x00bad"))  # NUL: makedirs must fail
    assert experience.append_exchange("m", b"p", b"out") is False


def test_v1_local_path_records_the_exchange(exp_root):
    """An OpenAI-route local completion lands in the log under the model dir name."""
    from routes import hybrid_routes as H

    class _Brain:
        def stream_fast(self, prompt, **kw):
            for b in b"hi":
                yield {"kind": "fast_byte", "byte": b}

    cfg = {"BRAIN": _Brain(), "BRAIN_MODEL": "wren1_3"}
    prompt = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
    list(H._local_events(cfg, "pytorch", prompt))
    (rec,) = _records(exp_root)
    assert rec["model"] == "wren1_3"
    assert rec["meta"] == {"backend": "pytorch", "route": "v1"}
    assert base64.b64decode(rec["output_b64"]) == b"hi"
