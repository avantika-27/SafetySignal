#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from frontend_app.utils.alerts_engine import build_rule_based_alerts
from pipeline_lib import DATA_DIR, ROOT, _safe_parquet_write


def main() -> int:
    trends = pd.read_csv(DATA_DIR / "monthly_cluster_trends.csv")
    labels = pd.read_csv(ROOT / "cluster_labels_report.csv")
    labels_for_alerts = labels.rename(columns={"cluster_id": "cluster"}).copy()

    alerts = build_rule_based_alerts(trends=trends, cluster_labels_report=labels_for_alerts)
    min_cols = ["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"]
    for col in min_cols:
        if col not in alerts.columns:
            alerts[col] = ""
    alerts = alerts.sort_values(["severity", "month_key"], ascending=[False, False])

    alerts.to_csv(DATA_DIR / "alerts.csv", index=False)
    _safe_parquet_write(alerts, DATA_DIR / "alerts.parquet")

    print(f"Agent 4 complete: alerts={len(alerts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
