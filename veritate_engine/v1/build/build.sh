#!/bin/sh
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POSIX build script (Linux + macOS). detects uname -s/-m, picks the
#   matching kernel TUs, invokes clang. binary lands at
#   veritate_engine/bin/<os>/<arch>/veritate.
# - macOS arm64: -mcpu=apple-m1 (NEON + SDOT implied). no NASM needed.
# - macOS x86_64 / Linux x86_64: AVX-512 + VNNI flags as on Windows.
# veritate_engine/build/build.sh
# ------------------------------------------------------------------------------------

set -e

OS_RAW="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS_RAW" in
  linux*)  OS_DIR="linux"  ;;
  darwin*) OS_DIR="macos"  ;;
  *) echo "build.sh: unsupported OS: $OS_RAW" >&2; exit 1 ;;
esac

ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
  x86_64|amd64)   ARCH_DIR="x86_64" ;;
  arm64|aarch64)  ARCH_DIR="arm64"  ;;
  *) echo "build.sh: unsupported arch: $ARCH_RAW" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/bin/$OS_DIR/$ARCH_DIR"
mkdir -p "$OUT_DIR"

CC="${CC:-clang}"
if ! command -v "$CC" >/dev/null 2>&1; then
  echo "build.sh: $CC not found. run setup.sh first." >&2
  exit 1
fi

CFLAGS_COMMON="-O3 -Wall -Wextra -Wno-unused-parameter -DVERITATE_VERIFY_DECODE -DVERITATE_GELU_ZERO_THRESH=4"
LDFLAGS_COMMON="-lm -lpthread"

SHARED_SRC="
  $ROOT/src/main.c
  $ROOT/src/dispatch.c
  $ROOT/src/model.c
  $ROOT/src/alloc.c
  $ROOT/src/threadpool.c
  $ROOT/kernels/scalar/matmul_scalar.c
  $ROOT/kernels/scalar/transformer_scalar.c
"

case "$ARCH_DIR" in
  x86_64)
    CFLAGS_ARCH="-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni"
    KERNEL_SRC="
      $ROOT/kernels/x86_64/matmul_avx2.c
      $ROOT/kernels/x86_64/matmul_vnni.c
      $ROOT/kernels/x86_64/matmul_int4.c
      $ROOT/kernels/x86_64/transformer_avx512.c
    "
    ;;
  arm64)
    if [ "$OS_DIR" = "macos" ]; then
      CFLAGS_ARCH="-arch arm64 -mcpu=apple-m1"
    else
      CFLAGS_ARCH="-march=armv8.2-a+dotprod"
    fi
    KERNEL_SRC="
      $ROOT/kernels/scalar/matmul_int4_scalar.c
      $ROOT/kernels/scalar/hadamard_scalar.c
      $ROOT/kernels/arm64/matmul_neon_sdot.c
      $ROOT/kernels/arm64/matmul_int4_neon.c
      $ROOT/kernels/arm64/transformer_neon.c
    "
    ;;
esac

OUT_BIN="$OUT_DIR/veritate"
echo "build.sh: target $OS_DIR/$ARCH_DIR -> $OUT_BIN"
echo "build.sh: $CC $CFLAGS_COMMON $CFLAGS_ARCH"

# shellcheck disable=SC2086
"$CC" $CFLAGS_COMMON $CFLAGS_ARCH \
    $SHARED_SRC \
    $KERNEL_SRC \
    -o "$OUT_BIN" \
    $LDFLAGS_COMMON

echo "build.sh: done. $OUT_BIN"
