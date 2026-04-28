#!/usr/bin/env python3
"""
Rebuild dashboard CSVs/Parquet from master_asrs.csv + embeddings_cache.npz
using the same fingerprint, PCA/UMAP/HDBSCAN settings as worksofar_updated.ipynb.

Run from project root:
  python export_dashboard_from_embeddings.py

Writes:
  - frontend_app/data/clustered_reports.csv (+ parquet)
  - frontend_app/data/cluster_summary.csv (+ parquet)
  - frontend_app/data/monthly_cluster_trends.csv (+ parquet)
  - frontend_app/data/alerts.csv (+ parquet)
  - cluster_labels_report.csv (+ parquet)
  - recent_spikes_report.csv (+ parquet)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from frontend_app.utils.alerts_engine import build_rule_based_alerts
import re

from pipeline_lib import ROOT, build_cluster_summary, build_monthly_trends, build_recent_spikes, write_dashboard_exports

# --- Mirror worksofar_updated.ipynb (embedding / DR / clustering cells) ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
ASRS_EMBED_MAX_CHARS = 1200
ASRS_EMBED_DEDUPE = True
PCA_DIMS = 50
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
RANDOM_STATE = 42
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
CHOSEN_MIN_CLUSTER_SIZE = 30

EMBEDDINGS_NPZ = ROOT / "embeddings_cache.npz"
DATA_CSV = ROOT / "master_asrs.csv"


def _data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    if "ACN" in df.columns:
        h.update(pd.util.hash_pandas_object(df["ACN"], index=False).values.tobytes())
    h.update(str(int(df["full_text"].str.len().sum())).encode())
    h.update(str(EMBEDDING_MODEL).encode())
    h.update(str(ASRS_EMBED_MAX_CHARS).encode())
    h.update(str(ASRS_EMBED_DEDUPE).encode())
    return h.hexdigest()


def _load_filtered_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if "Synopsis" in df.columns:
        df["full_text"] = (df["Narrative"].fillna("") + " " + df["Synopsis"].fillna("")).str.lower()
    else:
        df["full_text"] = df["Narrative"].fillna("").str.lower()
    df = df[df["full_text"].str.len() >= 30].copy()
    return df


def _build_cluster_labels_report_no_noise(clustered: pd.DataFrame) -> pd.DataFrame:
    """Same as pipeline_lib.build_cluster_labels_report but skip HDBSCAN noise (-1)."""
    stopwords: set[str] = {
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
    for cid, group in clustered.groupby("cluster", sort=True):
        if int(cid) == -1:
            continue
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
                "purity_reference_col": "",
                "purity_context": "",
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def main() -> int:
    if not DATA_CSV.is_file():
        print("Missing", DATA_CSV, file=sys.stderr)
        return 1
    if not EMBEDDINGS_NPZ.is_file():
        print("Missing", EMBEDDINGS_NPZ, file=sys.stderr)
        return 1

    df = _load_filtered_df()
    fp = _data_fingerprint(df)
    z = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    emb = z["embeddings"]
    if str(z["fingerprint"]) != fp or emb.shape[0] != len(df):
        print("Embeddings cache does not match filtered dataframe (fingerprint or row count).", file=sys.stderr)
        print("  df rows:", len(df), "emb rows:", emb.shape[0], file=sys.stderr)
        print("  fp match:", str(z["fingerprint"]) == fp, file=sys.stderr)
        return 1

    print("Loaded embeddings", emb.shape, "for", len(df), "reports (fingerprint OK).")

    pca = PCA(n_components=PCA_DIMS, random_state=RANDOM_STATE)
    embeddings_pca = pca.fit_transform(emb)
    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    embeddings_umap = umap_model.fit_transform(embeddings_pca)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CHOSEN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
    )
    cluster_ids = clusterer.fit_predict(embeddings_umap)

    umap_2d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=2,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    xy = umap_2d.fit_transform(embeddings_pca)

    df = df.reset_index(drop=True)
    df["cluster_id"] = cluster_ids.astype(int)
    df["umap_x"] = xy[:, 0]
    df["umap_y"] = xy[:, 1]
    df["cluster_label"] = np.where(df["cluster_id"].eq(-1), "Noise", "Cluster " + df["cluster_id"].astype(str))

    noise_pct = float((df["cluster_id"] == -1).mean() * 100)
    n_clust = int(df["cluster_id"].nunique() - (1 if (df["cluster_id"] == -1).any() else 0))
    print(f"HDBSCAN: clusters (excl. noise)={n_clust}, noise_pct={noise_pct:.2f}%")

    df["report_id"] = df["ACN"].astype(str).str.strip()
    # CSV often reads YYYYMM as float (e.g. 202401.0); str(float) breaks %Y%m parsing.
    _d = pd.to_numeric(df["Date"], errors="coerce")
    _d_str = np.where(_d.notna(), np.floor(_d).astype(np.int64).astype(str), np.nan)
    _parsed = pd.to_datetime(pd.Series(_d_str, index=df.index), format="%Y%m", errors="coerce")
    df["month_key"] = _parsed.dt.strftime("%Y-%m")
    df["date"] = _parsed.dt.strftime("%Y-%m-01")
    df["text"] = df["Narrative"].fillna("").astype(str).str.strip()
    empty = df["text"].eq("")
    df.loc[empty, "text"] = df.loc[empty, "Synopsis"].fillna("").astype(str).str.strip()
    df["flight_phase"] = df.get("Flight Phase", "").fillna("").astype(str).replace("", "Unknown Phase")
    df["vessel_type"] = df.get("Aircraft Operator", "").fillna("").astype(str).replace("nan", "")

    clustered = df[
        [
            "report_id",
            "date",
            "month_key",
            "text",
            "cluster_id",
            "cluster_label",
            "umap_x",
            "umap_y",
            "vessel_type",
            "flight_phase",
        ]
    ].rename(columns={"cluster_id": "cluster"}).dropna(subset=["month_key"]).reset_index(drop=True)

    labels = _build_cluster_labels_report_no_noise(clustered)
    summary = build_cluster_summary(labels)

    trends = build_monthly_trends(clustered)
    spikes = build_recent_spikes(trends)
    labels_for_alerts = labels.rename(columns={"cluster_id": "cluster"})
    alerts = build_rule_based_alerts(trends=trends, cluster_labels_report=labels_for_alerts)
    min_cols = ["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"]
    for col in min_cols:
        if col not in alerts.columns:
            alerts[col] = ""

    write_dashboard_exports(clustered, labels, summary, trends, spikes, alerts)

    print("Wrote dashboard artifacts under frontend_app/data and cluster_labels_report.csv")
    print("  clustered_reports:", len(clustered), "rows")
    print("  cluster_labels_report:", len(labels), "clusters")
    print("  monthly_trends:", len(trends), "rows")
    print("  alerts:", len(alerts), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
