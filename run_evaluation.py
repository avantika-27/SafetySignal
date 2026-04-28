#!/usr/bin/env python3
"""Run B737 / NTSB-style benchmark evaluation and write JSON report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    script = ROOT / "evaluation" / "run_b737_evaluation.py"
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
