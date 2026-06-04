# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - wire contract between node and hub. dataclasses are the only types that cross
#   the network. every field documented inline because this is the api boundary.
# - kinds are a closed set. adding a new job kind requires touching both sides.
# - capabilities are box-static facts (cores, ram, vram, arch). load reports live
#   in heartbeat, not capabilities.
# veritate_mesh/protocol.py
# ------------------------------------------------------------------------------------
# Imports:

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------------------------
# Constants

PROTOCOL_VERSION = 1

ROLE_OFF  = "off"
ROLE_NODE = "node"
ROLE_HUB  = "hub"
ROLE_BOTH = "both"
VALID_ROLES = (ROLE_OFF, ROLE_NODE, ROLE_HUB, ROLE_BOTH)

JOB_STATUS_PENDING  = "pending"
JOB_STATUS_ASSIGNED = "assigned"
JOB_STATUS_RUNNING  = "running"
JOB_STATUS_DONE     = "done"
JOB_STATUS_FAILED   = "failed"
VALID_JOB_STATUS = (
    JOB_STATUS_PENDING, JOB_STATUS_ASSIGNED, JOB_STATUS_RUNNING,
    JOB_STATUS_DONE, JOB_STATUS_FAILED,
)

JOB_KIND_STAGE_A           = "stage_a_pretrain"
JOB_KIND_STAGE_B_SPECIALTY = "stage_b_specialty"
JOB_KIND_AFFECT_TRAIN      = "affect_probe_train"
JOB_KIND_SLEEP_ADAPTER     = "sleep_adapter_update"
JOB_KIND_DATA_GEN          = "data_gen"
JOB_KIND_EVAL_REGION       = "eval_region"
VALID_JOB_KINDS = (
    JOB_KIND_STAGE_A, JOB_KIND_STAGE_B_SPECIALTY, JOB_KIND_AFFECT_TRAIN,
    JOB_KIND_SLEEP_ADAPTER, JOB_KIND_DATA_GEN, JOB_KIND_EVAL_REGION,
)

# ------------------------------------------------------------------------------------
# Capabilities

@dataclass
class Capabilities:
    """box-static facts. immutable per process lifetime."""
    node_id:          str
    hostname:         str
    os_name:          str   # "windows" / "darwin" / "linux"
    arch:             str   # "x86_64" / "arm64"
    cpu_cores:        int
    ram_gb:           float
    vram_gb:          float
    gpu_name:         str
    gpu_backend:      str   # "cuda" / "mps" / "vulkan" / "none"
    veritate_build:   int
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Capabilities":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

# ------------------------------------------------------------------------------------
# Job spec

@dataclass
class JobRequirements:
    """min hardware needed to accept a job. hub matches against capabilities."""
    min_ram_gb:    float = 0.0
    min_vram_gb:   float = 0.0
    min_cpu_cores: int   = 1
    arch_in:       Optional[List[str]] = None  # None = any
    os_in:         Optional[List[str]] = None  # None = any
    gpu_required:  bool  = False

@dataclass
class Job:
    job_id:       str
    kind:         str                       # one of VALID_JOB_KINDS
    payload:      Dict[str, Any]            # kind-specific, opaque to the mesh layer
    requirements: JobRequirements
    status:       str   = JOB_STATUS_PENDING
    assigned_to:  Optional[str] = None      # node_id
    created_at:   float = field(default_factory=time.time)
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    progress:     Dict[str, Any] = field(default_factory=dict)
    result:       Dict[str, Any] = field(default_factory=dict)
    error:        str   = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        req_raw = d.get("requirements") or {}
        req = JobRequirements(**{
            k: req_raw.get(k, getattr(JobRequirements, k, None))
            for k in JobRequirements.__dataclass_fields__
            if k in req_raw or hasattr(JobRequirements, k)
        }) if isinstance(req_raw, dict) else JobRequirements()
        return cls(
            job_id       = d["job_id"],
            kind         = d["kind"],
            payload      = d.get("payload") or {},
            requirements = req,
            status       = d.get("status", JOB_STATUS_PENDING),
            assigned_to  = d.get("assigned_to"),
            created_at   = float(d.get("created_at") or time.time()),
            started_at   = d.get("started_at"),
            finished_at  = d.get("finished_at"),
            progress     = d.get("progress") or {},
            result       = d.get("result") or {},
            error        = d.get("error") or "",
        )

# ------------------------------------------------------------------------------------
# Matching

def capabilities_satisfy(req: JobRequirements, caps: Capabilities) -> bool:
    """true iff caps meet every minimum in req. used by hub when picking a node."""
    if caps.ram_gb    < req.min_ram_gb:    return False
    if caps.vram_gb   < req.min_vram_gb:   return False
    if caps.cpu_cores < req.min_cpu_cores: return False
    if req.gpu_required and caps.gpu_backend == "none": return False
    if req.arch_in and caps.arch not in req.arch_in:    return False
    if req.os_in   and caps.os_name not in req.os_in:   return False
    return True
