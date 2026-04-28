#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from pipeline_lib import DATA_DIR, ROOT, _safe_parquet_write, build_cluster_labels_report, build_cluster_summary


def main() -> int:
    clustered_path = DATA_DIR / "clustered_reports.csv"
    clustered = pd.read_csv(clustered_path)

    labels = build_cluster_labels_report(clustered)
    summary = build_cluster_summary(labels)

    labels.to_csv(ROOT / "cluster_labels_report.csv", index=False)
    summary.to_csv(DATA_DIR / "cluster_summary.csv", index=False)
    _safe_parquet_write(labels, ROOT / "cluster_labels_report.parquet")
    _safe_parquet_write(summary, DATA_DIR / "cluster_summary.parquet")

    print(f"Agent 2 complete: labels={len(labels)}, summary={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
