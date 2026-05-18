# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Filesystem-read tool. Jailed to a single root directory; rejects any path
#   that resolves outside the root after symlink expansion. Read-only. Returns
#   at most _MAX_BYTES of file content as a string (utf-8, replacement on bad
#   bytes), with truncation marker.
# - Binary files are detected by a heuristic (presence of NUL in first 512 bytes)
#   and rejected. The agent should use `fetch` for HTTP resources instead.
# - Path is provided as a path RELATIVE to the root. Absolute paths are
#   rejected. ".." segments are rejected. Symlinks pointing outside the root
#   are rejected.
# veritate_mri/agent/tools/filesystem.py
# ------------------------------------------------------------------------------------
# Imports:

import os
from typing import Any, Dict

from . import Tool

# ------------------------------------------------------------------------------------
# Constants

_MAX_BYTES = 32 * 1024  # 32 kB per read; agent can ask for more if needed
_BINARY_PROBE_BYTES = 512

# ------------------------------------------------------------------------------------
# Functions


def _safe_resolve(root: str, rel_path: str) -> str:
    """Resolve `rel_path` against `root`. Return the absolute path on success,
    or raise ValueError with a user-readable reason."""
    if not isinstance(rel_path, str):
        raise ValueError(f"path must be string, got {type(rel_path).__name__}")
    if not rel_path:
        raise ValueError("empty path")
    if rel_path.startswith("/") or (len(rel_path) > 1 and rel_path[1] == ":"):
        raise ValueError("absolute paths are not allowed; pass a path relative to root")
    if any(part == ".." for part in rel_path.split(os.sep)):
        raise ValueError("'..' segments are not allowed")
    root_abs = os.path.realpath(root)
    target_abs = os.path.realpath(os.path.join(root_abs, rel_path))
    if not target_abs.startswith(root_abs + os.sep) and target_abs != root_abs:
        raise ValueError("path resolves outside the root")
    return target_abs


def _read(root: str, rel_path: str, start: int = 0, length: int = _MAX_BYTES) -> str:
    try:
        abs_path = _safe_resolve(root, rel_path)
    except ValueError as e:
        return f"error: {e}"
    if not os.path.exists(abs_path):
        return f"error: file does not exist: {rel_path}"
    if not os.path.isfile(abs_path):
        return f"error: not a regular file: {rel_path}"
    try:
        with open(abs_path, "rb") as f:
            if start:
                f.seek(int(start))
            head = f.read(_BINARY_PROBE_BYTES)
            if b"\x00" in head:
                return f"error: binary file rejected: {rel_path}"
            remainder = f.read(max(0, int(length) - len(head)))
        chunk = head + remainder
    except (OSError, ValueError) as e:
        return f"error: {type(e).__name__}: {e}"
    text = chunk[: max(0, int(length))].decode("utf-8", errors="replace")
    sz = os.path.getsize(abs_path)
    if start + len(chunk) < sz:
        text += f"\n... [truncated, {sz - (start + len(chunk))} bytes remaining]"
    return text


def make_tool(root: str) -> Tool:
    """Build a filesystem tool jailed to `root`. The root must exist and be a
    directory."""
    if not os.path.isdir(root):
        raise ValueError(f"fs_read root does not exist: {root}")
    root_abs = os.path.realpath(root)

    def _execute(args: Dict[str, Any]) -> str:
        path = args.get("path")
        if path is None:
            return "error: missing required arg 'path'"
        start = args.get("start", 0)
        length = args.get("length", _MAX_BYTES)
        try:
            start = int(start)
            length = int(length)
        except (TypeError, ValueError):
            return "error: 'start' and 'length' must be integers"
        if start < 0 or length < 0 or length > 1024 * 1024:
            return "error: 'start' must be >=0, 'length' must be 0..1048576"
        return _read(root_abs, path, start=start, length=length)

    return Tool(
        name="fs_read",
        description=f"Read a UTF-8 text file under {os.path.basename(root_abs)}/. Read-only, jailed to that directory.",
        args_schema={
            "path": {"type": "string", "required": True,
                     "doc": "Path relative to the jail root. No '..' segments, no absolute paths."},
            "start": {"type": "integer", "required": False,
                      "doc": "Byte offset to start reading from. Default 0."},
            "length": {"type": "integer", "required": False,
                       "doc": f"Maximum bytes to read. Default {_MAX_BYTES}. Capped at 1048576."},
        },
        execute=_execute,
    )
