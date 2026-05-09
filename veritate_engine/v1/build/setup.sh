#!/bin/sh
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - one-time POSIX toolchain install. macOS uses Apple's xcrun clang (already
#   on every developer Mac); brew clang only if you want the LLVM upstream.
#   Linux uses the distro package manager.
# veritate_engine/build/setup.sh
# ------------------------------------------------------------------------------------

set -e

OS_RAW="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS_RAW" in
  darwin*)
    if xcrun --find clang >/dev/null 2>&1; then
      echo "setup.sh: macOS Apple clang found at $(xcrun --find clang)"
      echo "setup.sh: ready. run build.sh next."
      exit 0
    fi
    echo "setup.sh: Apple clang missing. install Xcode CLT: xcode-select --install"
    exit 1
    ;;
  linux*)
    if command -v clang >/dev/null 2>&1; then
      echo "setup.sh: clang found at $(command -v clang)"
      echo "setup.sh: ready. run build.sh next."
      exit 0
    fi
    if   command -v apt-get >/dev/null 2>&1; then SUDO="sudo"; PKG="apt-get install -y clang"
    elif command -v dnf     >/dev/null 2>&1; then SUDO="sudo"; PKG="dnf install -y clang"
    elif command -v pacman  >/dev/null 2>&1; then SUDO="sudo"; PKG="pacman -S --noconfirm clang"
    else
      echo "setup.sh: no known package manager. install clang manually." >&2
      exit 1
    fi
    echo "setup.sh: installing clang via $PKG"
    $SUDO $PKG
    ;;
  *)
    echo "setup.sh: unsupported OS: $OS_RAW" >&2
    exit 1
    ;;
esac

echo "setup.sh: done. run build.sh next."
