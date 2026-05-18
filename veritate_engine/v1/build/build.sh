#!/bin/sh
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POSIX build script (Linux + macOS). Compiles each TU separately so we can
#   give baseline ISA flags to shared code (main, dispatch, model, scalar
#   kernels) and per-ISA flags only to the kernel TUs that need them. Runtime
#   dispatch in src/dispatch.c picks kernels based on CPUID, so a binary built
#   with the highest kernel-ISA still runs on older CPUs without that ISA.
# - x86_64 baseline: SSE4.2 (Nehalem 2008+). Covers every 64-bit Intel Mac
#   including Ivy Bridge / Sandy Bridge that lack AVX2.
# - arm64 macOS: -mcpu=apple-m1 (NEON + SDOT implied).
# - arm64 Linux: armv8-a baseline; the SDOT kernel gets +dotprod individually.
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

CFLAGS_COMMON="-O3 -flto=full -Wall -Wextra -Wno-unused-parameter -DVERITATE_VERIFY_DECODE -DVERITATE_GELU_ZERO_THRESH=4"
LDFLAGS_COMMON="-lm -lpthread"

# Files compiled with baseline ISA flags. Safe to run on any CPU that satisfies
# the OS/arch baseline (Nehalem+ for x86_64; ARMv8+ for arm64).
SHARED_SRC="
  $ROOT/src/main.c
  $ROOT/src/dispatch.c
  $ROOT/src/model.c
  $ROOT/src/alloc.c
  $ROOT/src/threadpool.c
  $ROOT/src/addons.c
  $ROOT/src/addons/slot_table.c
  $ROOT/kernels/scalar/matmul_scalar.c
  $ROOT/kernels/scalar/matmul_ternary_scalar.c
  $ROOT/kernels/scalar/transformer_scalar.c
"

case "$ARCH_DIR" in
  x86_64)
    CFLAGS_BASELINE="-msse4.2"
    ;;
  arm64)
    if [ "$OS_DIR" = "macos" ]; then
      CFLAGS_BASELINE="-arch arm64 -mcpu=apple-m1"
    else
      CFLAGS_BASELINE="-march=armv8-a"
    fi
    ;;
esac

OUT_BIN="$OUT_DIR/veritate"
TMP_OBJ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/veritate-build.XXXXXX")"
trap 'rm -rf "$TMP_OBJ_DIR"' EXIT

OBJS=""

echo "build.sh: target $OS_DIR/$ARCH_DIR -> $OUT_BIN"
echo "build.sh: baseline $CFLAGS_BASELINE; per-kernel flags applied to specialized TUs only"

# Pass 1: shared TUs at baseline.
for src in $SHARED_SRC; do
  obj="$TMP_OBJ_DIR/$(basename "$src" .c).o"
  # shellcheck disable=SC2086
  "$CC" $CFLAGS_COMMON $CFLAGS_BASELINE -c "$src" -o "$obj"
  OBJS="$OBJS $obj"
done

# Pass 2: per-kernel TUs with their own ISA flags. dispatch.c gates entry to
# each at runtime by CPUID, so a binary compiled with -mavx512vnni still runs
# on a CPU without VNNI (it just won't enter that kernel).
compile_kernel() {
  src="$1"; flags="$2"
  obj="$TMP_OBJ_DIR/$(basename "$src" .c).o"
  # shellcheck disable=SC2086
  "$CC" $CFLAGS_COMMON $CFLAGS_BASELINE $flags -c "$src" -o "$obj"
  OBJS="$OBJS $obj"
}

case "$ARCH_DIR" in
  x86_64)
    compile_kernel "$ROOT/kernels/x86_64/matmul_avx2.c"          "-mavx2"
    compile_kernel "$ROOT/kernels/x86_64/matmul_vnni.c"          "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni"
    compile_kernel "$ROOT/kernels/x86_64/matmul_int4.c"          "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni"
    compile_kernel "$ROOT/kernels/x86_64/matmul_ternary_vnni.c"  "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni"
    compile_kernel "$ROOT/kernels/x86_64/transformer_avx512.c"   "-mavx512f -mavx512bw -mavx512vl"
    ;;
  arm64)
    # Scalar fallback kernels go through the shared baseline already; only the
    # NEON+SDOT-specific ones need extra flags.
    SDOT_FLAGS=""
    if [ "$OS_DIR" != "macos" ]; then
      SDOT_FLAGS="-march=armv8.2-a+dotprod"
    fi
    compile_kernel "$ROOT/kernels/scalar/matmul_int4_scalar.c"   ""
    compile_kernel "$ROOT/kernels/scalar/hadamard_scalar.c"      ""
    compile_kernel "$ROOT/kernels/arm64/matmul_neon_sdot.c"      "$SDOT_FLAGS"
    compile_kernel "$ROOT/kernels/arm64/matmul_int4_neon.c"      "$SDOT_FLAGS"
    compile_kernel "$ROOT/kernels/arm64/transformer_neon.c"      "$SDOT_FLAGS"
    ;;
esac

# Pass 2.5 (macOS only): ObjC bridge + (optional) .metal shader compile.
# - Bridge ALWAYS compiles on macOS so the engine's metal_* symbols resolve at
#   link time. If the metallib isn't built, the bridge reports a clean runtime
#   error when invoked.
# - Shader compile needs `xcrun metal`, which lives in full Xcode (NOT in the
#   Xcode Command Line Tools). On a CLT-only system we skip it and tell the
#   user. The engine still builds and CPU paths still work.
METAL_LDFLAGS=""
if [ "$OS_DIR" = "macos" ]; then
  METAL_BRIDGE_SRC="$ROOT/src/metal_dispatch.m"
  if [ -f "$METAL_BRIDGE_SRC" ]; then
    bridge_obj="$TMP_OBJ_DIR/metal_dispatch.o"
    # shellcheck disable=SC2086
    "$CC" $CFLAGS_COMMON $CFLAGS_BASELINE -ObjC -fobjc-arc -c "$METAL_BRIDGE_SRC" -o "$bridge_obj"
    OBJS="$OBJS $bridge_obj"
    METAL_LDFLAGS="-framework Metal -framework Foundation -framework CoreGraphics"
  fi

  METAL_SRC_DIR="$ROOT/kernels/metal"
  if [ -d "$METAL_SRC_DIR" ] && [ -n "$(ls -1 "$METAL_SRC_DIR"/*.metal 2>/dev/null)" ]; then
    if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
      AIR_DIR="$TMP_OBJ_DIR/metal_air"
      mkdir -p "$AIR_DIR"
      shader_err=0
      for m in "$METAL_SRC_DIR"/*.metal; do
        airfile="$AIR_DIR/$(basename "$m" .metal).air"
        if ! xcrun -sdk macosx metal -c "$m" -o "$airfile"; then
          echo "build.sh: metal compile FAILED for $(basename "$m") (see stderr above)"
          shader_err=1
          break
        fi
      done
      if [ "$shader_err" = "0" ]; then
        METALLIB_OUT="$OUT_DIR/default.metallib"
        if xcrun -sdk macosx metallib "$AIR_DIR"/*.air -o "$METALLIB_OUT"; then
          echo "build.sh: built $METALLIB_OUT"
        else
          echo "build.sh: metallib link FAILED (see stderr above)"
        fi
      fi
    else
      echo "build.sh: 'xcrun metal' not available. Install full Xcode (not just CLT) to enable GPU path."
      echo "build.sh:   sudo xcode-select --install                 (CLT only — what you have)"
      echo "build.sh:   then install Xcode from the App Store and run:"
      echo "build.sh:   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
      echo "build.sh: engine will still build; verify-metal will report 'default.metallib not found'."
    fi
  fi
fi

# Pass 3: link.
# shellcheck disable=SC2086
"$CC" $CFLAGS_COMMON $CFLAGS_BASELINE $OBJS -o "$OUT_BIN" $LDFLAGS_COMMON $METAL_LDFLAGS

echo "build.sh: done. $OUT_BIN"
