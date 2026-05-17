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
if   command -v python3 >/dev/null 2>&1; then PY="python3"
elif command -v python  >/dev/null 2>&1; then PY="python"
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

"$PY" veritate.py "$@"
status=$?

if [ "$status" -ne 0 ]; then
    echo
    echo "veritate exited with code $status."
    read -r -p "Press Enter to close..." _ || true
fi
exit "$status"
