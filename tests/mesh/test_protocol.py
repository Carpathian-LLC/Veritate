# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mesh wire contract: the codec is dataclasses.asdict on the way out and
#   <Type>.from_dict on the way in, carried as json by veritate_mesh/client.py.
#   round-trip cases are parametrized off the module's own VALID_JOB_KINDS and
#   VALID_JOB_STATUS so a new kind is covered without editing this file.
# - transport is mocked at urlopen; no socket is ever opened.
# tests/mesh/test_protocol.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import random
import urllib.request
from dataclasses import asdict

import pytest

from veritate_mesh import protocol
from veritate_mesh.client import HubClient, HubClientError

# ------------------------------------------------------------------------------------
# Constants

SEED = 1743

HUB_ADDRESS = "http://hub.invalid:9000"
AUTH_TOKEN  = "token-abc"

CAPS_FIELDS = {
    "node_id":        "node-a",
    "hostname":       "box-a",
    "os_name":        "darwin",
    "arch":           "arm64",
    "cpu_cores":      8,
    "ram_gb":         16.0,
    "vram_gb":        0.0,
    "gpu_name":       "apple-gpu",
    "gpu_backend":    "mps",
    "veritate_build": 42,
}

UNKNOWN_KIND    = "not_a_real_job_kind"
TRUNCATED_JSON  = b'{"job_id": "j1", "kind": "data_gen"'
NON_OBJECT_JSON = b"[1, 2, 3]"

HTTP_OK = 200

# ------------------------------------------------------------------------------------
# Functions

class _FakeResponse:
    """stand-in for the urlopen context manager. no socket involved."""

    def __init__(self, status, raw):
        self._status = status
        self._raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getcode(self):
        return self._status

    def read(self):
        return self._raw


def _caps():
    return protocol.Capabilities(**CAPS_FIELDS)


def _job(kind, status=protocol.JOB_STATUS_PENDING):
    rng = random.Random(SEED)
    return protocol.Job(
        job_id       = f"job-{rng.randint(0, 10_000)}",
        kind         = kind,
        payload      = {"step": rng.randint(0, 10_000), "shard": rng.randint(0, 8)},
        requirements = protocol.JobRequirements(min_ram_gb=8.0, min_cpu_cores=4, arch_in=["arm64"]),
        status       = status,
        assigned_to  = CAPS_FIELDS["node_id"],
        created_at   = 1000.5,
        progress     = {"pct": 10},
        result       = {},
    )


def _client_reading(monkeypatch, raw, status=HTTP_OK):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(status, raw))
    return HubClient(HUB_ADDRESS, AUTH_TOKEN)


def _wire(job):
    return json.dumps(asdict(job)).encode("utf-8")


@pytest.mark.parametrize("kind", protocol.VALID_JOB_KINDS)
def test_job_roundtrip_per_kind(monkeypatch, kind):
    """Every kind in VALID_JOB_KINDS survives encode to json and decode by poll_job unchanged."""
    job = _job(kind)
    client = _client_reading(monkeypatch, _wire(job))
    assert client.poll_job(_caps(), long_poll_secs=0.0) == job


@pytest.mark.parametrize("status", protocol.VALID_JOB_STATUS)
def test_job_roundtrip_per_status(monkeypatch, status):
    """Every status in VALID_JOB_STATUS survives the job wire round-trip unchanged."""
    job = _job(protocol.JOB_KIND_DATA_GEN, status=status)
    client = _client_reading(monkeypatch, _wire(job))
    assert client.poll_job(_caps(), long_poll_secs=0.0) == job


def test_job_requirements_roundtrip(monkeypatch):
    """Nested JobRequirements survive the job wire round-trip unchanged."""
    job = _job(protocol.JOB_KIND_DATA_GEN)
    client = _client_reading(monkeypatch, _wire(job))
    assert client.poll_job(_caps(), long_poll_secs=0.0).requirements == job.requirements


def test_capabilities_roundtrip():
    """A Capabilities record survives asdict -> json -> Capabilities.from_dict unchanged."""
    caps = _caps()
    assert protocol.Capabilities.from_dict(json.loads(json.dumps(asdict(caps)))) == caps


def test_job_missing_job_id_raises():
    """Job.from_dict on a frame with no job_id raises KeyError."""
    with pytest.raises(KeyError):
        protocol.Job.from_dict({"kind": protocol.JOB_KIND_DATA_GEN})


def test_job_missing_kind_raises():
    """Job.from_dict on a frame with no kind raises KeyError."""
    with pytest.raises(KeyError):
        protocol.Job.from_dict({"job_id": "j1"})


def test_truncated_frame_raises(monkeypatch):
    """A truncated json frame raises HubClientError instead of decoding partially."""
    client = _client_reading(monkeypatch, TRUNCATED_JSON)
    with pytest.raises(HubClientError):
        client.poll_job(_caps(), long_poll_secs=0.0)


def test_non_object_frame_raises(monkeypatch):
    """A well-formed json frame that is not an object raises HubClientError."""
    client = _client_reading(monkeypatch, NON_OBJECT_JSON)
    with pytest.raises(HubClientError):
        client.poll_job(_caps(), long_poll_secs=0.0)


@pytest.mark.xfail(reason="protocol has no kind validation; only hub /mesh/hub/submit rejects unknown kinds")
def test_unknown_job_kind_raises():
    """Job.from_dict on a kind outside VALID_JOB_KINDS raises ValueError."""
    with pytest.raises(ValueError):
        protocol.Job.from_dict({"job_id": "j1", "kind": UNKNOWN_KIND})


@pytest.mark.xfail(reason="Capabilities.from_dict fills every absent field with None instead of raising")
def test_short_capabilities_frame_raises():
    """Capabilities.from_dict on a frame missing required fields raises rather than filling None."""
    with pytest.raises((KeyError, TypeError, ValueError)):
        protocol.Capabilities.from_dict({"node_id": CAPS_FIELDS["node_id"]})
