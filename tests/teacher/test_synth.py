# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests for the synth job runner. mocks Client.complete to avoid network.
# tests/teacher/test_synth.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
from unittest.mock import MagicMock

from veritate_mri.teacher.synth import SynthJob

# ------------------------------------------------------------------------------------
# Constants

_FIVE_PROMPTS = [
    {"id": f"p{i}", "messages": [{"role": "user", "content": f"prompt {i}"}], "system": None}
    for i in range(5)
]

# ------------------------------------------------------------------------------------
# Functions

def _mock_client_factory(responses):
    idx = {"i": 0}

    def factory():
        c = MagicMock()

        def complete(messages, temperature=0.7, max_tokens=2048, system=None):
            i = idx["i"]
            idx["i"] += 1
            return responses[i % len(responses)]

        c.complete = MagicMock(side_effect=complete)
        c._counter = idx
        return c

    return factory


def test_run_writes_samples(tmp_path):
    """5-prompt job writes 5 samples.jsonl lines and a state.json."""
    distinct = [
        "alpha apples ascend across azure arches always afternoon",
        "bravo berries balance between bright blue boats brilliantly",
        "charlie clouds carry calm contemplative cosmic chants carefully",
        "delta dancers drift down distant dunes during dawn diligently",
        "echo eagles examine elegant evergreen estates every evening",
    ]
    factory = _mock_client_factory(distinct)
    job = SynthJob("j1", "openai", "gpt-4o", _FIVE_PROMPTS, str(tmp_path), client_factory=factory)
    result = job.run()
    assert result["completed"] == 5
    assert result["failed"] == 0
    samples_path = os.path.join(str(tmp_path), "samples.jsonl")
    with open(samples_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 5
    state_path = os.path.join(str(tmp_path), "state.json")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    assert state["remaining"] == []


def test_rerun_skips_done(tmp_path):
    """re-running the job skips ids already in samples.jsonl."""
    distinct = [
        "alpha apples ascend across azure arches always afternoon",
        "bravo berries balance between bright blue boats brilliantly",
        "charlie clouds carry calm contemplative cosmic chants carefully",
        "delta dancers drift down distant dunes during dawn diligently",
        "echo eagles examine elegant evergreen estates every evening",
    ]
    factory1 = _mock_client_factory(distinct)
    job1 = SynthJob("j1", "openai", "gpt-4o", _FIVE_PROMPTS, str(tmp_path), client_factory=factory1)
    job1.run()
    calls_seen = {"n": 0}

    def factory2():
        c = MagicMock()

        def complete(messages, temperature=0.7, max_tokens=2048, system=None):
            calls_seen["n"] += 1
            return "should not be called"

        c.complete = MagicMock(side_effect=complete)
        return c

    job2 = SynthJob("j1", "openai", "gpt-4o", _FIVE_PROMPTS, str(tmp_path), client_factory=factory2)
    result = job2.run()
    assert result["completed"] == 0
    assert calls_seen["n"] == 0


def test_length_filter_rejects(tmp_path):
    """too-short response counts in failed and is not written."""
    factory = _mock_client_factory(["x"])
    prompt = [{"id": "p0", "messages": [{"role": "user", "content": "hi"}], "system": None}]
    job = SynthJob("j1", "openai", "gpt-4o", prompt, str(tmp_path), client_factory=factory, min_chars=50)
    result = job.run()
    assert result["failed"] == 1
    assert result["completed"] == 0


def test_dup_rejection(tmp_path):
    """exact duplicate response on a different prompt is skipped as dup."""
    same_text = "this is a long enough response that is the same across both prompts here"
    factory = _mock_client_factory([same_text, same_text])
    prompts = [
        {"id": "p0", "messages": [{"role": "user", "content": "a"}], "system": None},
        {"id": "p1", "messages": [{"role": "user", "content": "b"}], "system": None},
    ]
    job = SynthJob("j1", "openai", "gpt-4o", prompts, str(tmp_path), client_factory=factory, max_concurrency=1)
    result = job.run()
    assert result["completed"] == 1
    assert result["skipped_dup"] == 1


def test_cache_hit_avoids_call(tmp_path):
    """rerun hits sqlite cache so client.complete is not invoked."""
    factory1 = _mock_client_factory(["this is a long enough response for caching"])
    prompt = [{"id": "p0", "messages": [{"role": "user", "content": "hi"}], "system": None}]
    job1 = SynthJob("j1", "openai", "gpt-4o", prompt, str(tmp_path), client_factory=factory1)
    job1.run()
    os.remove(os.path.join(str(tmp_path), "samples.jsonl"))
    call_count = {"n": 0}

    def factory2():
        c = MagicMock()

        def complete(messages, temperature=0.7, max_tokens=2048, system=None):
            call_count["n"] += 1
            return "should not be called from network"

        c.complete = MagicMock(side_effect=complete)
        return c

    job2 = SynthJob("j1", "openai", "gpt-4o", prompt, str(tmp_path), client_factory=factory2)
    result = job2.run()
    assert call_count["n"] == 0
    assert result["completed"] == 1
