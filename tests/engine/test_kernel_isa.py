# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - un-gated load helpers defined in AVX-512 kernel TUs must stay at the SSE4.2
#   baseline (VERITATE_BASELINE_CODEGEN): they run on every x86_64 CPU at model
#   load, so any auto-vectorized AVX/AVX-512 in them is a SIGILL on AVX2-only
#   boxes. asserts the built binary's disassembly for those symbols carries no
#   zmm/ymm registers. linux x86_64 only (needs objdump + an x86 binary).
# tests/engine/test_kernel_isa.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import platform
import re
import shutil
import subprocess
import sys

import pytest
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

BASELINE_SYMBOLS = ("prep_b", "prep_b_keep_raw", "prep_b_int4")
WIDE_REG = re.compile(r"%[yz]mm\d+")
SYM_LINE = re.compile(r"^[0-9a-f]+ <(.+)>:$")

# ------------------------------------------------------------------------------------
# Functions


def _disassembly():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        pytest.skip("baseline-ISA pin is x86_64 codegen; disassembly check needs linux objdump")
    if shutil.which("objdump") is None:
        pytest.skip("objdump not available")
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        pytest.skip(f"engine binary not built: {exe}")
    out = subprocess.run(["objdump", "-d", exe], capture_output=True, timeout=120)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    return out.stdout.decode(errors="replace")


def _function_body(disasm, symbol):
    lines, active = [], False
    for line in disasm.splitlines():
        m = SYM_LINE.match(line)
        if m:
            active = m.group(1) == symbol
            continue
        if active and line.strip():
            lines.append(line)
    return lines


@pytest.mark.parametrize("symbol", BASELINE_SYMBOLS)
def test_ungated_load_helper_has_no_wide_vectors(symbol):
    """Un-gated prep helpers in AVX-512 TUs disassemble free of zmm/ymm registers."""
    body = _function_body(_disassembly(), symbol)
    if not body:
        # LTO may inline a helper into its baseline-TU caller entirely (observed for
        # prep_b_keep_raw); the inlined copy inherits the caller's baseline codegen.
        pytest.skip(f"{symbol} has no standalone symbol (inlined by LTO)")
    wide = [ln for ln in body if WIDE_REG.search(ln)]
    assert not wide, f"{symbol} uses wide vector registers:\n" + "\n".join(wide[:10])
