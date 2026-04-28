#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline_lib import DATA_DIR, ROOT, _safe_parquet_write


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_csv(path))


def main() -> int:
    required = {
        "clustered_reports_csv": DATA_DIR / "clustered_reports.csv",
        "cluster_summary_csv": DATA_DIR / "cluster_summary.csv",
        "monthly_cluster_trends_csv": DATA_DIR / "monthly_cluster_trends.csv",
        "alerts_csv": DATA_DIR / "alerts.csv",
        "cluster_labels_report_csv": ROOT / "cluster_labels_report.csv",
        "recent_spikes_report_csv": ROOT / "recent_spikes_report.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    for _, path in required.items():
        df = pd.read_csv(path)
        _safe_parquet_write(df, path.with_suffix(".parquet"))

    report = {
        "status": "success",
        "files": {
            name: {"path": str(path), "rows": _row_count(path), "bytes": path.stat().st_size}
            for name, path in required.items()
        },
    }
    out_path = ROOT / "pipeline_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Agent 5 complete: validation report at {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
