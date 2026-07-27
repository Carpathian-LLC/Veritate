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
from typing import Any

from . import ERROR_PREFIX, Tool

# ------------------------------------------------------------------------------------
# Constants

_MAX_BYTES = 32 * 1024  # 32 kB per read; agent can ask for more if needed
_MAX_CAP   = 1024 * 1024  # hard ceiling on a single `length` request
_BINARY_PROBE_BYTES = 512
_NUL = b"\x00"

_ARG_PATH   = "path"
_ARG_START  = "start"
_ARG_LENGTH = "length"
_START_DEFAULT = 0

_TOOL_NAME = "fs_read"
_TOOL_DESCRIPTION_FMT = ("Read a UTF-8 text file under {root}/. "
                         "Read-only, jailed to that directory.")
_PATH_DOC       = "Path relative to the jail root. No '..' segments, no absolute paths."
_START_DOC_FMT  = "Byte offset to start reading from. Default {default}."
_LENGTH_DOC_FMT = "Maximum bytes to read. Default {default}. Capped at {cap}."
_TRUNCATED_FMT  = "\n... [truncated, {n} bytes remaining]"

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


def _read(root: str, rel_path: str, start: int = _START_DEFAULT, length: int = _MAX_BYTES) -> str:
    try:
        abs_path = _safe_resolve(root, rel_path)
    except ValueError as e:
        return f"{ERROR_PREFIX}{e}"
    if not os.path.exists(abs_path):
        return f"{ERROR_PREFIX}file does not exist: {rel_path}"
    if not os.path.isfile(abs_path):
        return f"{ERROR_PREFIX}not a regular file: {rel_path}"
    try:
        with open(abs_path, "rb") as f:
            if start:
                f.seek(int(start))
            head = f.read(_BINARY_PROBE_BYTES)
            if _NUL in head:
                return f"{ERROR_PREFIX}binary file rejected: {rel_path}"
            remainder = f.read(max(0, int(length) - len(head)))
        chunk = head + remainder
    except (OSError, ValueError) as e:
        return f"{ERROR_PREFIX}{type(e).__name__}: {e}"
    text = chunk[: max(0, int(length))].decode("utf-8", errors="replace")
    sz = os.path.getsize(abs_path)
    if start + len(chunk) < sz:
        text += _TRUNCATED_FMT.format(n=sz - (start + len(chunk)))
    return text


def make_tool(root: str) -> Tool:
    """Build a filesystem tool jailed to `root`. The root must exist and be a
    directory."""
    if not os.path.isdir(root):
        raise ValueError(f"fs_read root does not exist: {root}")
    root_abs = os.path.realpath(root)

    def _execute(args: dict[str, Any]) -> str:
        path = args.get(_ARG_PATH)
        if path is None:
            return f"{ERROR_PREFIX}missing required arg {_ARG_PATH!r}"
        start = args.get(_ARG_START, _START_DEFAULT)
        length = args.get(_ARG_LENGTH, _MAX_BYTES)
        try:
            start = int(start)
            length = int(length)
        except (TypeError, ValueError):
            return f"{ERROR_PREFIX}{_ARG_START!r} and {_ARG_LENGTH!r} must be integers"
        if start < 0 or length < 0 or length > _MAX_CAP:
            return (f"{ERROR_PREFIX}{_ARG_START!r} must be >=0, "
                    f"{_ARG_LENGTH!r} must be 0..{_MAX_CAP}")
        return _read(root_abs, path, start=start, length=length)

    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION_FMT.format(root=os.path.basename(root_abs)),
        args_schema={
            _ARG_PATH:   {"type": "string", "required": True, "doc": _PATH_DOC},
            _ARG_START:  {"type": "integer", "required": False,
                          "doc": _START_DOC_FMT.format(default=_START_DEFAULT)},
            _ARG_LENGTH: {"type": "integer", "required": False,
                          "doc": _LENGTH_DOC_FMT.format(default=_MAX_BYTES, cap=_MAX_CAP)},
        },
        execute=_execute,
    )
