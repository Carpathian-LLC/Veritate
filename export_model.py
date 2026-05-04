#!/usr/bin/env python3
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "veritate_mri"))

from export import export_checkpoint

try:
    result = export_checkpoint('pg19_120m_bf16_v1', 9800)
    print("Export successful:")
    for k, v in result.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Export failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
