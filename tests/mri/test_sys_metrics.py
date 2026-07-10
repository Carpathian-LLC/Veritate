# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for linux gpu name resolution in sys_metrics: lspci description
#   cleaning and the slot->name map parse. pure functions, no subprocess.
# tests/mri/test_sys_metrics.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from runtime import sys_metrics

# ------------------------------------------------------------------------------------
# Constants

LSPCI_FIXTURE = (
    "0000:00:02.0 VGA compatible controller: Intel Corporation CoffeeLake-S GT2 [UHD Graphics 630] (rev 02)\n"
    "0000:01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)\n"
    "0000:00:1f.0 ISA bridge: Intel Corporation Q370 Chipset LPC/eSPI Controller\n"
)

# ------------------------------------------------------------------------------------
# Functions

def test_clean_gpu_name_intel_bracket():
    """Intel description resolves to the bracketed model with vendor prefix."""
    assert sys_metrics._clean_gpu_name(
        "Intel Corporation CoffeeLake-S GT2 [UHD Graphics 630] (rev 02)"
    ) == "Intel UHD Graphics 630"


def test_clean_gpu_name_nvidia_bracket():
    """NVIDIA description resolves to the GeForce model name."""
    assert sys_metrics._clean_gpu_name(
        "NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)"
    ) == "NVIDIA GeForce RTX 3090"


def test_clean_gpu_name_amd_uses_last_bracket():
    """AMD's double-bracket format takes the model bracket, not the vendor alias."""
    assert sys_metrics._clean_gpu_name(
        "Advanced Micro Devices, Inc. [AMD/ATI] Navi 21 [Radeon RX 6800 XT]"
    ) == "AMD Radeon RX 6800 XT"


def test_clean_gpu_name_no_bracket_falls_back():
    """A description with no bracket returns the trimmed text unchanged."""
    assert sys_metrics._clean_gpu_name("Some Vendor Mystery Chip 9000") == "Some Vendor Mystery Chip 9000"


def test_lspci_gpu_names_maps_only_display_controllers(monkeypatch):
    """lspci parse keys display/VGA/3D controllers by PCI slot and skips others."""
    monkeypatch.setattr(sys_metrics, "_run", lambda cmd, timeout=2.0: LSPCI_FIXTURE)
    names = sys_metrics._lspci_gpu_names()
    assert names == {
        "0000:00:02.0": "Intel UHD Graphics 630",
        "0000:01:00.0": "NVIDIA GeForce RTX 3090",
    }


def test_lspci_gpu_names_empty_when_lspci_absent(monkeypatch):
    """No lspci on PATH (_run returns None) yields an empty map, never raises."""
    monkeypatch.setattr(sys_metrics, "_run", lambda cmd, timeout=2.0: None)
    assert sys_metrics._lspci_gpu_names() == {}
