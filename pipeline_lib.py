from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "frontend_app" / "data"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _stable_unit(seed: str) -> float:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _safe_parquet_write(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False, engine="fastparquet")
    except Exception:
        # Keep pipeline runnable in environments without parquet engine.
        pass


def read_raw_asrs(raw_path: Path | None = None) -> pd.DataFrame:
    path = raw_path or (ROOT / "master_asrs.csv")
    # Use python parser for robustness on very wide ASRS exports.
    df = pd.read_csv(path, low_memory=False, engine="python")
    for required in ["ACN", "Date", "Narrative", "Synopsis", "Anomaly", "Flight Phase", "Make Model Name"]:
        if required not in df.columns:
            df[required] = ""
    return df


def build_clustered_reports(raw_df: pd.DataFrame, top_clusters: int = 25) -> pd.DataFrame:
    df = raw_df.copy()
    df["report_id"] = df["ACN"].astype(str).str.strip()
    _d = pd.to_numeric(df["Date"], errors="coerce")
    _d_str = np.where(_d.notna(), np.floor(_d).astype(np.int64).astype(str), np.nan)
    _parsed = pd.to_datetime(pd.Series(_d_str, index=df.index), format="%Y%m", errors="coerce")
    df["month_key"] = _parsed.dt.strftime("%Y-%m")
    df["date"] = _parsed.dt.strftime("%Y-%m-01")
    df["text"] = df["Narrative"].map(_clean_text)
    empty_text = df["text"].eq("")
    df.loc[empty_text, "text"] = df.loc[empty_text, "Synopsis"].map(_clean_text)
    df["flight_phase"] = df["Flight Phase"].map(_clean_text).replace("", "Unknown Phase")
    df["anomaly"] = df["Anomaly"].map(_clean_text).replace("", "Unspecified Event")
    df["vessel_type"] = df.get("Aircraft Operator", "").astype(str).replace("nan", "")
    df["cluster_key"] = df["flight_phase"] + " | " + df["anomaly"]

    top_keys = df["cluster_key"].value_counts().head(top_clusters).index.tolist()
    cluster_map = {key: i + 1 for i, key in enumerate(top_keys)}
    df["cluster"] = df["cluster_key"].map(cluster_map).fillna(0).astype(int)
    df["cluster_label"] = np.where(df["cluster"] == 0, "Noise / Other", df["cluster_key"])

    centers = {}
    for cid in sorted(df["cluster"].unique()):
        ux = (_stable_unit(f"center-x-{cid}") * 16.0) - 8.0
        uy = (_stable_unit(f"center-y-{cid}") * 16.0) - 8.0
        centers[cid] = (ux, uy)

    xs = []
    ys = []
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        cx, cy = centers[cid]
        jitter_x = (_stable_unit(f"{row['report_id']}-x") - 0.5) * 1.2
        jitter_y = (_stable_unit(f"{row['report_id']}-y") - 0.5) * 1.2
        xs.append(cx + jitter_x)
        ys.append(cy + jitter_y)
    df["umap_x"] = xs
    df["umap_y"] = ys

    out = df[
        [
            "report_id",
            "date",
            "month_key",
            "text",
            "cluster",
            "cluster_label",
            "umap_x",
            "umap_y",
            "vessel_type",
            "flight_phase",
            "anomaly",
            "Make Model Name",
        ]
    ].rename(columns={"Make Model Name": "make_model_name"})
    out = out.dropna(subset=["month_key"]).reset_index(drop=True)
    return out


def build_cluster_labels_report(clustered_reports: pd.DataFrame) -> pd.DataFrame:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "were",
        "was",
        "have",
        "had",
        "into",
        "during",
        "while",
        "flight",
        "pilot",
    }
    rows = []
    for cid, group in clustered_reports.groupby("cluster", sort=True):
        texts = " ".join(group["text"].dropna().astype(str).tolist()).lower()
        tokens = [t for t in re.findall(r"[a-z]{4,}", texts) if t not in stopwords]
        top_terms = pd.Series(tokens).value_counts().head(8).index.tolist() if tokens else []
        top_phase = group["flight_phase"].mode().iat[0] if "flight_phase" in group and not group["flight_phase"].mode().empty else ""
        label = group["cluster_label"].iat[0]
        rows.append(
            {
                "cluster_id": int(cid),
                "size": int(len(group)),
                "name": label,
                "description": f"Cluster with {len(group)} ASRS reports centered on {label}.",
                "keywords": ", ".join(top_terms),
                "causal_summary": f"Common pattern: {label}.",
                "evidence_bullets": f"Top terms: {', '.join(top_terms[:5])}" if top_terms else "",
                "limitations": "Descriptive pattern from text fields; not causal proof.",
                "top_phase": top_phase,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def build_cluster_summary(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": labels["cluster_id"].astype(int),
            "cluster_label": labels["name"].astype(str),
            "cluster_size": labels["size"].astype(int),
            "description": labels["description"].astype(str),
            "keywords": labels["keywords"].astype(str),
            "is_noise": labels["cluster_id"].astype(int).eq(-1)
            | (
                labels["cluster_id"].astype(int).eq(0)
                & labels["name"].astype(str).str.contains("noise", case=False, na=False)
            ),
        }
    )


def build_monthly_trends(clustered_reports: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        clustered_reports.groupby(["month_key", "cluster", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total = monthly.groupby("month_key", as_index=False)["count"].sum().rename(columns={"count": "total_reports_month"})
    monthly = monthly.merge(total, on="month_key", how="left")
    monthly["normalized_rate"] = monthly["count"] / monthly["total_reports_month"].replace(0, np.nan)
    monthly["normalized_rate"] = monthly["normalized_rate"].fillna(0.0)
    monthly = monthly.sort_values(["cluster", "month_key"]).reset_index(drop=True)
    monthly["pct_change"] = (
        monthly.groupby("cluster")["normalized_rate"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100
    )
    monthly["rolling_mean_rate"] = (
        monthly.groupby("cluster")["normalized_rate"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    monthly["rolling_mean_rate"] = monthly["rolling_mean_rate"].fillna(monthly["normalized_rate"])

    z_parts = []
    for _, g in monthly.groupby("cluster"):
        std = g["normalized_rate"].std(ddof=0)
        if std and std > 0:
            z = (g["normalized_rate"] - g["normalized_rate"].mean()) / std
        else:
            z = pd.Series(np.zeros(len(g)), index=g.index)
        z_parts.append(z)
    monthly["max_rate_z"] = pd.concat(z_parts).sort_index() if z_parts else 0.0
    return monthly


def build_recent_spikes(monthly_trends: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    work = monthly_trends.copy()
    spikes = work[work["max_rate_z"] >= 1.5].copy()
    if spikes.empty:
        spikes = work.sort_values("max_rate_z", ascending=False).head(max(3, min(top_n, len(work))))
    spikes = spikes.sort_values("max_rate_z", ascending=False).head(top_n)
    return pd.DataFrame(
        {
            "cluster_id": spikes["cluster"].astype(int),
            "max_rate_z": spikes["max_rate_z"].astype(float),
            "month": spikes["month_key"].astype(str),
            "count": spikes["count"].astype(int),
            "rate": spikes["normalized_rate"].astype(float),
            "name": spikes["cluster_label"].astype(str),
        }
    ).reset_index(drop=True)


def write_dashboard_exports(
    clustered_reports: pd.DataFrame,
    cluster_labels_report: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    monthly_cluster_trends: pd.DataFrame,
    recent_spikes_report: pd.DataFrame,
    alerts: pd.DataFrame | None = None,
) -> None:
    ensure_data_dir()
    clustered_reports.to_csv(DATA_DIR / "clustered_reports.csv", index=False)
    cluster_summary.to_csv(DATA_DIR / "cluster_summary.csv", index=False)
    monthly_cluster_trends.to_csv(DATA_DIR / "monthly_cluster_trends.csv", index=False)
    (alerts if alerts is not None else pd.DataFrame(columns=["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"])).to_csv(
        DATA_DIR / "alerts.csv", index=False
    )
    cluster_labels_report.to_csv(ROOT / "cluster_labels_report.csv", index=False)
    recent_spikes_report.to_csv(ROOT / "recent_spikes_report.csv", index=False)

    _safe_parquet_write(clustered_reports, DATA_DIR / "clustered_reports.parquet")
    _safe_parquet_write(cluster_summary, DATA_DIR / "cluster_summary.parquet")
    _safe_parquet_write(monthly_cluster_trends, DATA_DIR / "monthly_cluster_trends.parquet")
    _safe_parquet_write(alerts if alerts is not None else pd.DataFrame(), DATA_DIR / "alerts.parquet")
    _safe_parquet_write(cluster_labels_report, ROOT / "cluster_labels_report.parquet")
    _safe_parquet_write(recent_spikes_report, ROOT / "recent_spikes_report.parquet")
