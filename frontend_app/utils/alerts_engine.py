from __future__ import annotations

from typing import List
import re

import numpy as np
import pandas as pd

ALERT_ROW_COLUMNS = [
    "alert_id",
    "cluster",
    "cluster_label",
    "alert_type",
    "severity",
    "priority",
    "confidence",
    "month_key",
    "growth_pct",
    "incidents",
    "message",
    "root_cause",
    "regulation_mapping",
    "relevance_score",
]


def _exclude_noise_and_residual_alert_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop HDBSCAN noise (-1) and id 0 from alert tables only (map/trends unchanged)."""
    if df.empty or "cluster" not in df.columns:
        return df
    c = pd.to_numeric(df["cluster"], errors="coerce")
    mask = c.notna() & ~c.isin([-1, 0])
    if "cluster_label" in df.columns:
        lbl = df["cluster_label"].astype(str).str.strip().str.lower()
        mask &= ~lbl.eq("noise")
    return df.loc[mask].reset_index(drop=True)


def _sort_alert_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Critical above high (string sort puts 'high' before 'critical'); then strongest signal first."""
    if df.empty:
        return df
    out = df.copy()
    sev = out["severity"].astype(str).str.lower()
    out["_sev_rank"] = sev.map({"critical": 0, "high": 1}).fillna(9).astype(int)
    sort_cols = ["_sev_rank", "growth_pct", "confidence"]
    asc = [True, False, False]
    if "incidents" in out.columns:
        sort_cols.append("incidents")
        asc.append(False)
    out = out.sort_values(sort_cols, ascending=asc).drop(columns=["_sev_rank"])
    return out.reset_index(drop=True)


def build_rule_based_alerts(
    trends: pd.DataFrame,
    cluster_labels_report: pd.DataFrame,
    min_total_volume: int = 15,
) -> pd.DataFrame:
    if trends.empty:
        return pd.DataFrame(columns=ALERT_ROW_COLUMNS)

    work = trends.copy()
    work["count"] = pd.to_numeric(work.get("count"), errors="coerce").fillna(0.0)
    # Some exports can contain duplicate (cluster, month_key) rows; collapse first for stable math.
    group_cols = [c for c in ["cluster", "cluster_label", "month_key"] if c in work.columns]
    if group_cols:
        agg_map = {
            "count": "sum",
            "total_reports_month": "max",
            "normalized_rate": "max",
            "max_rate_z": "max",
            "pct_change": "max",
            "rolling_mean_rate": "max",
        }
        agg_map = {k: v for k, v in agg_map.items() if k in work.columns}
        work = work.groupby(group_cols, as_index=False).agg(agg_map)
    work = work.sort_values(["cluster", "month_key"])
    work["rolling_3m_avg"] = (
        work.groupby("cluster")["count"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    work["ratio_to_baseline"] = work["count"] / work["rolling_3m_avg"].replace(0, np.nan)
    work["growth_pct"] = ((work["count"] - work["rolling_3m_avg"]) / work["rolling_3m_avg"].replace(0, np.nan) * 100).fillna(0)

    latest_month = sorted(work["month_key"].dropna().unique())[-1]
    latest = work[work["month_key"] == latest_month].copy()
    latest = latest[latest["total_reports_month"] >= min_total_volume]
    _cid = pd.to_numeric(latest["cluster"], errors="coerce")
    latest = latest[_cid.notna() & ~_cid.isin([-1, 0])].copy()
    if "cluster_label" in latest.columns:
        _lbl = latest["cluster_label"].astype(str).str.strip().str.lower()
        latest = latest[~_lbl.eq("noise")].copy()
    latest["severity"] = np.where(
        latest["ratio_to_baseline"] > 2.0,
        "critical",
        np.where(latest["ratio_to_baseline"] > 1.5, "high", "none"),
    )
    latest["alert_mode"] = "baseline"
    latest = latest[latest["severity"] != "none"].copy()
    if latest.empty:
        # Fallback when history is short: surface top z-score spikes as high priority.
        if "max_rate_z" not in work.columns:
            return pd.DataFrame(columns=ALERT_ROW_COLUMNS)
        ranked = (
            work.sort_values("max_rate_z", ascending=False)
            .drop_duplicates(["cluster_label"])
            .copy()
        )
        _rc = pd.to_numeric(ranked["cluster"], errors="coerce")
        ranked = ranked[_rc.notna() & ~_rc.isin([-1, 0])]
        if "cluster_label" in ranked.columns:
            _rl = ranked["cluster_label"].astype(str).str.strip().str.lower()
            ranked = ranked[~_rl.eq("noise")]
        ranked = ranked.head(3)
        ranked["severity"] = np.where(ranked["max_rate_z"] >= 2.2, "critical", "high")
        ranked["growth_pct"] = ranked.get("pct_change", 0.0)
        ranked["alert_mode"] = "fallback_z"
        latest = ranked

    # Prefer cluster causal_summary from cluster_labels_report.csv for causal synopsis.
    root_source_col = (
        "causal_summary"
        if "causal_summary" in cluster_labels_report.columns
        else ("description" if "description" in cluster_labels_report.columns else ("name" if "name" in cluster_labels_report.columns else "cluster_label"))
    )
    root_cause_map = cluster_labels_report.set_index("cluster")[root_source_col].to_dict()
    evidence_map = (
        cluster_labels_report.set_index("cluster")["evidence_bullets"].to_dict()
        if "evidence_bullets" in cluster_labels_report.columns
        else {}
    )
    keywords_map = (
        cluster_labels_report.set_index("cluster")["keywords"].to_dict()
        if "keywords" in cluster_labels_report.columns
        else {}
    )
    description_map = (
        cluster_labels_report.set_index("cluster")["description"].to_dict()
        if "description" in cluster_labels_report.columns
        else {}
    )
    rows: List[dict] = []
    for _, row in latest.iterrows():
        mode = str(row.get("alert_mode", "baseline"))
        if mode == "fallback_z":
            z = float(row.get("max_rate_z", 0.0)) if pd.notna(row.get("max_rate_z", np.nan)) else 0.0
            confidence = int(min(99, max(55, round(55 + (max(0.0, z) * 18)))))
        else:
            ratio = float(row["ratio_to_baseline"]) if pd.notna(row["ratio_to_baseline"]) else 1.0
            confidence = int(min(99, max(55, round((ratio - 1.0) * 75 + 55))))
        # Use data month directly (not current clock year) to avoid misleading alert dates.
        month_key = str(row["month_key"])
        alert_id = f"AVIATION SAFETY ALERT #{month_key}-{int(row['cluster']):03d}"
        root_cause = root_cause_map.get(int(row["cluster"]))
        if isinstance(root_cause, str):
            rc = root_cause.strip()
            # Many exports use placeholder text like "Common pattern: Cluster 27."
            if re.fullmatch(r"(?i)common pattern:\s*cluster\s*\d+\.?", rc):
                ev = str(evidence_map.get(int(row["cluster"]), "")).strip()
                kw = str(keywords_map.get(int(row["cluster"]), "")).strip()
                desc = str(description_map.get(int(row["cluster"]), "")).strip()
                if ev:
                    root_cause = ev
                elif kw:
                    root_cause = f"Observed keywords: {kw}"
                elif desc:
                    root_cause = desc
        if not isinstance(root_cause, str) or not root_cause.strip():
            root_cause = row["cluster_label"]
        incidents = int(row["count"])
        prev = int(round(float(row["rolling_3m_avg"]))) if pd.notna(row["rolling_3m_avg"]) else 0
        message = (
            f"TREND: {row['cluster_label']}\n"
            f"Growth: {row['growth_pct']:+.1f}% ({prev}->{incidents} incidents/month)\n"
            f"Period: Previous 3 months vs {row['month_key']}"
        )
        rows.append(
            {
                "alert_id": alert_id,
                "cluster": int(row["cluster"]),
                "cluster_label": row["cluster_label"],
                "alert_type": "spike",
                "severity": row["severity"],
                "priority": row["severity"].upper(),
                "confidence": confidence,
                "month_key": row["month_key"],
                "growth_pct": float(row["growth_pct"]),
                "incidents": incidents,
                "message": message,
                "root_cause": root_cause,
                "regulation_mapping": "FAA/CFR RAG mapping available in notebook export",
                "relevance_score": round(min(0.99, 0.6 + (confidence / 200)), 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=ALERT_ROW_COLUMNS)
    out = _sort_alert_rows(pd.DataFrame(rows))
    return _exclude_noise_and_residual_alert_clusters(out)
