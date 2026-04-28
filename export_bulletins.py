#!/usr/bin/env python3
"""CLI: export professional safety bulletins (Markdown + HTML; optional PDF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "frontend_app") not in sys.path:
    sys.path.insert(0, str(ROOT / "frontend_app"))

from utils.bulletin_export import export_bulletins
from utils.data_loader import load_data_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export alert bulletins as Markdown/HTML (optional PDF).")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "exports" / "bulletins",
        help="Output directory (default: exports/bulletins)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write PDF (requires weasyprint or xhtml2pdf).",
    )
    parser.add_argument(
        "--regs-k",
        type=int,
        default=6,
        help="Regulation citations per bulletin (default 6).",
    )
    args = parser.parse_args()

    paths = export_bulletins(
        load_data_bundle(),
        args.out,
        regs_top_k=args.regs_k,
        write_pdf=args.pdf,
    )
    if not paths:
        print("No alerts to export.", file=sys.stderr)
        return 1
    print(f"Wrote {len(paths)} file(s) under {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
