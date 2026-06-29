# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - node-side http client to the mesh hub. stdlib only. blocking.
# - one instance per node. caller owns retry policy.
# veritate_mesh/client.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import urllib.error
import urllib.parse
import urllib.request

from dataclasses import asdict
from typing import Optional

from .protocol import Capabilities, Job, PROTOCOL_VERSION

# ------------------------------------------------------------------------------------
# Constants

PATH_REGISTER  = "/mesh/register"
PATH_HEARTBEAT = "/mesh/heartbeat"
PATH_JOB_NEXT  = "/mesh/job/next"
PATH_JOB_PROGRESS_FMT = "/mesh/job/{job_id}/progress"
PATH_JOB_RESULT_FMT   = "/mesh/job/{job_id}/result"

HEADER_AUTH        = "Authorization"
HEADER_CONTENT     = "Content-Type"
HEADER_PROTOCOL    = "X-Veritate-Protocol"
CONTENT_JSON       = "application/json"
USER_AGENT         = "veritate-mesh-client/1"

LONG_POLL_BUFFER_SECS = 10.0
HTTP_NO_CONTENT       = 204

# ------------------------------------------------------------------------------------
# Errors

class HubClientError(Exception):
    """raised on http, network, auth, or protocol errors."""

# ------------------------------------------------------------------------------------
# Functions

def _decode_body(raw: bytes) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HubClientError(f"invalid json from hub: {e}")
    if not isinstance(data, dict):
        raise HubClientError(f"expected json object, got {type(data).__name__}")
    return data


def _raise_http(status: int, raw: bytes) -> None:
    try:
        body = raw.decode("utf-8", errors="replace")
    except Exception:
        body = repr(raw)
    raise HubClientError(f"{status}: {body[:200]}")


# ------------------------------------------------------------------------------------
# Client

class HubClient:
    """node-side http client. all calls are blocking. one instance per node."""

    def __init__(self, hub_address: str, auth_token: str, timeout_secs: float = 30.0):
        self.hub_address  = (hub_address or "").rstrip("/")
        self.auth_token   = auth_token
        self.timeout_secs = float(timeout_secs)

    # --------------------------------------------------------------------------------

    def _url(self, path: str, query: Optional[dict] = None) -> str:
        url = self.hub_address + path
        if query:
            url = url + "?" + urllib.parse.urlencode(query)
        return url

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 query: Optional[dict] = None, timeout: Optional[float] = None):
        url = self._url(path, query)
        data = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header(HEADER_AUTH, f"Bearer {self.auth_token}")
        req.add_header("User-Agent", USER_AGENT)
        req.add_header(HEADER_PROTOCOL, str(PROTOCOL_VERSION))
        if data is not None:
            req.add_header(HEADER_CONTENT, CONTENT_JSON)
        t = self.timeout_secs if timeout is None else timeout
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:
                status = resp.getcode()
                raw = resp.read()
                return status, raw
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
            except Exception:
                raw = b""
            _raise_http(e.code, raw)
        except urllib.error.URLError as e:
            raise HubClientError(f"network error: {e.reason}")
        except (TimeoutError, OSError) as e:
            raise HubClientError(f"socket error: {e}")

    # --------------------------------------------------------------------------------

    def register(self, caps: Capabilities) -> dict:
        """POST /mesh/register with caps json. returns hub response dict.
        raises HubClientError on http / network / auth failure."""
        status, raw = self._request("POST", PATH_REGISTER, body=asdict(caps))
        if status >= 400:
            _raise_http(status, raw)
        return _decode_body(raw)

    def heartbeat(self, node_id: str, load: dict) -> dict:
        """POST /mesh/heartbeat. returns hub response dict (may include directives)."""
        payload = {"node_id": node_id, "load": load or {}}
        status, raw = self._request("POST", PATH_HEARTBEAT, body=payload)
        if status >= 400:
            _raise_http(status, raw)
        return _decode_body(raw)

    def poll_job(self, caps: Capabilities, long_poll_secs: float = 25.0) -> Optional[Job]:
        """GET /mesh/job/next?node_id=...&long_poll=N. returns Job or None on 204.
        long_poll_secs MUST be honored by the hub; this client just sets a slightly
        higher http timeout to cover the wait."""
        query = {"node_id": caps.node_id, "long_poll": long_poll_secs}
        timeout = float(long_poll_secs) + LONG_POLL_BUFFER_SECS
        status, raw = self._request("GET", PATH_JOB_NEXT, query=query, timeout=timeout)
        if status == HTTP_NO_CONTENT:
            return None
        if status >= 400:
            _raise_http(status, raw)
        if not raw:
            return None
        data = _decode_body(raw)
        if not data:
            return None
        try:
            return Job.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise HubClientError(f"malformed job payload: {e}")

    def report_progress(self, job_id: str, progress: dict) -> bool:
        """POST /mesh/job/{job_id}/progress."""
        path = PATH_JOB_PROGRESS_FMT.format(job_id=urllib.parse.quote(job_id, safe=""))
        payload = {"progress": progress or {}}
        status, raw = self._request("POST", path, body=payload)
        if status >= 400:
            _raise_http(status, raw)
        return True

    def report_result(self, job_id: str, result: dict) -> bool:
        """POST /mesh/job/{job_id}/result with {"result": result}."""
        path = PATH_JOB_RESULT_FMT.format(job_id=urllib.parse.quote(job_id, safe=""))
        payload = {"result": result or {}}
        status, raw = self._request("POST", path, body=payload)
        if status >= 400:
            _raise_http(status, raw)
        return True

    def report_failure(self, job_id: str, error: str) -> bool:
        """POST /mesh/job/{job_id}/result with {"error": error}."""
        path = PATH_JOB_RESULT_FMT.format(job_id=urllib.parse.quote(job_id, safe=""))
        payload = {"error": error or ""}
        status, raw = self._request("POST", path, body=payload)
        if status >= 400:
            _raise_http(status, raw)
        return True
