# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Web-fetch tool. urllib only (stdlib). Refuses non-http(s) schemes, refuses
#   private / loopback / link-local IPs (SSRF prevention), refuses redirects
#   into a refused-target. Returns at most _MAX_BYTES of decoded text body.
# - This tool intentionally does NOT execute JavaScript, does NOT follow more
#   than _MAX_REDIRECTS hops, and trims to a byte cap before returning. The
#   model can ask for more bytes via the `length` arg if needed.
# - To suppress SSRF: every resolved IP is checked against the private ranges
#   defined in ipaddress.ip_address.is_private / is_loopback / is_link_local.
# veritate_mri/agent/tools/fetch.py
# ------------------------------------------------------------------------------------
# Imports:

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

from . import Tool

# ------------------------------------------------------------------------------------
# Constants

_HTTP_TIMEOUT_SECS = 15
_MAX_BYTES         = 32 * 1024     # 32 kB default
_MAX_CAP           = 1024 * 1024   # 1 MB cap
_MAX_REDIRECTS     = 3
_ALLOWED_SCHEMES   = ("http", "https")
_UA                = "veritate-agent/1.0"

# ------------------------------------------------------------------------------------
# Functions


def _validate_url(url: str) -> str:
    """Parse + validate. Returns the (possibly normalized) URL. Raises ValueError."""
    if not isinstance(url, str):
        raise ValueError(f"url must be string, got {type(url).__name__}")
    if not url:
        raise ValueError("empty url")
    p = urllib.parse.urlparse(url)
    if p.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"only http(s) allowed; got scheme {p.scheme!r}")
    if not p.hostname:
        raise ValueError("missing hostname")
    # Resolve hostname; reject if it points to a private/loopback/link-local IP.
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"dns resolution failed: {e}")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError(f"refused: {p.hostname} resolves to non-public IP {addr}")
    return url


def _fetch(url: str, length: int = _MAX_BYTES) -> str:
    try:
        url = _validate_url(url)
    except ValueError as e:
        return f"error: {e}"

    length = max(0, min(int(length), _MAX_CAP))
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    # urllib follows redirects by default; we need to validate each hop.
    # The simple-but-correct approach: use a custom opener with a redirect
    # handler that re-validates each new URL.

    class _SafeRedirect(urllib.request.HTTPRedirectHandler):
        def __init__(self):
            self.hops = 0

        def http_error_302(self, req, fp, code, msg, headers):
            if self.hops >= _MAX_REDIRECTS:
                raise urllib.error.HTTPError(req.full_url, code, "too many redirects", headers, fp)
            self.hops += 1
            new_url = headers.get("Location") or headers.get("location") or ""
            new_url = urllib.parse.urljoin(req.full_url, new_url)
            try:
                _validate_url(new_url)
            except ValueError as e:
                raise urllib.error.HTTPError(req.full_url, 403, f"refused redirect: {e}", headers, fp)
            return super().http_error_302(req, fp, code, msg, headers)
        http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    opener = urllib.request.build_opener(_SafeRedirect())
    try:
        with opener.open(req, timeout=_HTTP_TIMEOUT_SECS) as resp:
            ctype = resp.headers.get("Content-Type", "")
            # Heuristic: refuse obvious binary blobs.
            if any(b in ctype.lower() for b in ("image/", "audio/", "video/", "application/octet-stream")):
                return f"error: refused content-type {ctype}"
            raw = resp.read(length)
    except urllib.error.HTTPError as e:
        return f"error: HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"error: network: {e.reason}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

    # Decode using a best-effort charset.
    text = raw.decode("utf-8", errors="replace")
    return text


def _execute(args: Dict[str, Any]) -> str:
    url = args.get("url")
    if url is None:
        return "error: missing required arg 'url'"
    length = args.get("length", _MAX_BYTES)
    try:
        length = int(length)
    except (TypeError, ValueError):
        return "error: 'length' must be an integer"
    return _fetch(url, length=length)


TOOL = Tool(
    name="fetch",
    description="Fetch an HTTP/HTTPS URL and return the response body as UTF-8 text. Private/loopback IPs are blocked.",
    args_schema={
        "url": {"type": "string", "required": True,
                "doc": "http:// or https:// URL. Private/loopback IPs are refused."},
        "length": {"type": "integer", "required": False,
                   "doc": f"Maximum response bytes to return. Default {_MAX_BYTES}, capped at {_MAX_CAP}."},
    },
    execute=_execute,
)
