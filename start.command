#!/bin/bash
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Double-click in Finder to install dependencies (first run) and launch the
#   Veritate dashboard. The .command extension makes Finder open it in Terminal.
#   Re-runs skip already-satisfied steps and just relaunch.
# start.command
# ------------------------------------------------------------------------------------

set -u
cd "$(cd "$(dirname "$0")" && pwd)"

PY=""
# Prefer the newest Python in the supported range [3.10, 3.13]. Veritate's
# launcher does its own tier-aware version check; this just keeps the
# bootstrap from picking a too-new Python (e.g. 3.14 from a fresh brew install)
# when an in-range one is also present.
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
        case "$ver" in
            3.10|3.11|3.12|3.13) PY="$cand"; break ;;
        esac
    fi
done

# Fall back to whatever python3 we can find — veritate.py will print a clear
# tier-specific error if it's out of range.
if [ -z "$PY" ]; then
    if   command -v python3 >/dev/null 2>&1; then PY="python3"
    elif command -v python  >/dev/null 2>&1; then PY="python"
    fi
fi

if [ -z "$PY" ]; then
    echo
    echo "Python 3.10+ is required but was not found on PATH."
    echo "Install with:  xcode-select --install"
    echo "Or download:   https://www.python.org/downloads/"
    echo
    read -r -p "Press Enter to close..." _ || true
    exit 1
fi

echo "[start.command] using $PY ($("$PY" -V 2>&1))"

"$PY" veritate.py "$@"
status=$?

if [ "$status" -ne 0 ]; then
    echo
    echo "veritate exited with code $status."
    read -r -p "Press Enter to close..." _ || true
fi
exit "$status"
