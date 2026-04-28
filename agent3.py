#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from pipeline_lib import DATA_DIR, ROOT, _safe_parquet_write, build_monthly_trends, build_recent_spikes


def main() -> int:
    clustered = pd.read_csv(DATA_DIR / "clustered_reports.csv")
    trends = build_monthly_trends(clustered)
    spikes = build_recent_spikes(trends)

    trends.to_csv(DATA_DIR / "monthly_cluster_trends.csv", index=False)
    spikes.to_csv(ROOT / "recent_spikes_report.csv", index=False)
    _safe_parquet_write(trends, DATA_DIR / "monthly_cluster_trends.parquet")
    _safe_parquet_write(spikes, ROOT / "recent_spikes_report.parquet")

    print(f"Agent 3 complete: trends={len(trends)}, spikes={len(spikes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
