from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import re

import numpy as np
import pandas as pd
import json
import base64


@dataclass(frozen=True)
class DataBundle:
    clustered_reports: pd.DataFrame
    monthly_cluster_trends: pd.DataFrame
    cluster_summary: pd.DataFrame
    alerts: pd.DataFrame
    cluster_labels_report: pd.DataFrame
    notebook_kpis: Dict[str, float]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    return _root_dir() / "data"


def _preferred_file(filename: str) -> Path:
    local_path = _data_dir() / filename
    # Treat tiny/header-only local files as placeholders and fall back to notebook exports at project root.
    if local_path.exists() and local_path.stat().st_size > 200:
        return local_path
    return _root_dir() / filename


def _require_columns(df: pd.DataFrame, required: Iterable[str], filename: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def _load_cluster_labels_report() -> pd.DataFrame:
    path = _preferred_file("cluster_labels_report.csv")
    df = pd.read_csv(path)
    _require_columns(df, ["cluster_id", "size", "name", "description"], path.name)
    df = df.rename(columns={"cluster_id": "cluster", "size": "cluster_size", "name": "cluster_label"})
    return df


def _load_recent_spikes_report() -> pd.DataFrame:
    path = _preferred_file("recent_spikes_report.csv")
    if not path.exists():
        return pd.DataFrame(
            columns=["cluster", "month_key", "count", "normalized_rate", "max_rate_z", "cluster_label"]
        )
    df = pd.read_csv(path)
    _require_columns(df, ["cluster_id", "month", "count", "rate"], path.name)
    df = df.rename(
        columns={
            "cluster_id": "cluster",
            "month": "month_key",
            "rate": "normalized_rate",
            "name": "cluster_label",
        }
    )
    df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce").dt.strftime("%Y-%m")
    return df


def _build_cluster_summary(cluster_labels: pd.DataFrame) -> pd.DataFrame:
    cols = ["cluster", "cluster_label", "cluster_size", "description", "keywords"]
    for col in cols:
        if col not in cluster_labels.columns:
            cluster_labels[col] = ""
    summary = cluster_labels[cols].copy()
    cid = pd.to_numeric(summary["cluster"], errors="coerce")
    cl = summary["cluster_label"].astype(str).str.strip()
    # HDBSCAN noise is -1 in the notebook. Cluster id 0 is a normal cluster there — do not treat 0 as noise.
    summary["is_noise"] = cid.eq(-1) | cl.str.fullmatch("(?i)noise")
    return summary


def _build_monthly_trends(spikes_df: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["month_key", "cluster", "cluster_label", "count", "normalized_rate", "max_rate_z"]
    keep_cols = [c for c in keep_cols if c in spikes_df.columns]

    notebook_trends = _load_monthly_trends_from_notebook(cluster_summary)
    if not notebook_trends.empty:
        df = notebook_trends.copy()
    else:
        df = spikes_df[keep_cols].copy() if not spikes_df.empty else pd.DataFrame()

    # If still sparse, derive from actual clustered report points.
    if df.empty or df["month_key"].nunique() <= 3:
        reports = _build_clustered_reports(cluster_summary, spikes_df)
        if not reports.empty and {"month_key", "cluster", "cluster_label"}.issubset(reports.columns):
            monthly = (
                reports.groupby(["month_key", "cluster", "cluster_label"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            df = monthly

    df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce").dt.strftime("%Y-%m")
    if "total_reports_month" not in df.columns:
        total = df.groupby("month_key", as_index=False)["count"].sum().rename(columns={"count": "total_reports_month"})
        df = df.merge(total, on="month_key", how="left")
    if "normalized_rate" not in df.columns:
        df["normalized_rate"] = df["count"] / df["total_reports_month"].replace(0, np.nan)
    df["normalized_rate"] = df["normalized_rate"].fillna(0.0)

    if "max_rate_z" not in df.columns:
        df["max_rate_z"] = np.nan
    if not spikes_df.empty and {"cluster_label", "month_key", "max_rate_z"}.issubset(spikes_df.columns):
        spike_z = spikes_df[["cluster_label", "month_key", "max_rate_z"]].copy()
        spike_z["month_key"] = pd.to_datetime(spike_z["month_key"], errors="coerce").dt.strftime("%Y-%m")
        df = df.merge(
            spike_z,
            on=["cluster_label", "month_key"],
            how="left",
            suffixes=("", "_spike"),
        )
        df["max_rate_z"] = df["max_rate_z"].fillna(df["max_rate_z_spike"])
        if "max_rate_z_spike" in df.columns:
            df = df.drop(columns=["max_rate_z_spike"])

    df = df.sort_values(["cluster", "month_key"])
    df["pct_change"] = (
        df.groupby("cluster")["normalized_rate"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100
    )
    if "rolling_mean_rate" not in df.columns:
        df["rolling_mean_rate"] = np.nan
    return df


def _decode_plotly_array(arr_obj) -> np.ndarray:
    if isinstance(arr_obj, list):
        return np.asarray(arr_obj, dtype=object)
    if isinstance(arr_obj, dict) and "bdata" in arr_obj and "dtype" in arr_obj:
        raw = base64.b64decode(arr_obj["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(arr_obj["dtype"]))
    return np.array([], dtype=float)


def _load_clustered_reports_from_notebook(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    nb_path = _root_dir() / "worksofar_updated.ipynb"
    if not nb_path.exists():
        return pd.DataFrame()

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    rows = []
    label_to_cluster = {
        str(r["cluster_label"]): int(r["cluster"])
        for _, r in cluster_summary.iterrows()
        if pd.notna(r["cluster_label"])
    }

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            fig = out.get("data", {}).get("application/vnd.plotly.v1+json")
            if not fig:
                continue
            for trace in fig.get("data", []):
                hover = str(trace.get("hovertemplate", ""))
                if "cluster_label" not in hover or "umap_x" not in hover or "umap_y" not in hover:
                    continue

                xs = _decode_plotly_array(trace.get("x"))
                ys = _decode_plotly_array(trace.get("y"))
                cds = trace.get("customdata", [])
                n = min(len(xs), len(ys), len(cds))
                if n == 0:
                    continue

                for i in range(n):
                    cd = cds[i] if i < len(cds) else []
                    cluster_id_text = str(cd[0]) if len(cd) > 0 else ""
                    topic_name = str(cd[1]) if len(cd) > 1 else str(trace.get("name", "Unknown"))
                    desc = str(cd[2]) if len(cd) > 2 else ""
                    acn = str(cd[3]) if len(cd) > 3 else f"row-{i}"
                    date_raw = str(cd[4]) if len(cd) > 4 else ""
                    phase = str(cd[5]) if len(cd) > 5 else ""
                    model = str(cd[6]) if len(cd) > 6 else ""
                    anomaly = str(cd[7]) if len(cd) > 7 else ""

                    date = pd.to_datetime(date_raw, format="%Y%m", errors="coerce")
                    month_key = date.strftime("%Y-%m") if pd.notna(date) else ""
                    parsed_cluster_id = None
                    match = re.search(r"(\d+)", cluster_id_text)
                    if match:
                        parsed_cluster_id = int(match.group(1))

                    rows.append(
                        {
                            "report_id": acn,
                            "date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
                            "month_key": month_key,
                            "text": desc,
                            "cluster": parsed_cluster_id if parsed_cluster_id is not None else label_to_cluster.get(topic_name, -1),
                            "cluster_label": topic_name,
                            "name": topic_name,
                            "flight_phase": phase,
                            "make_model_name": model,
                            "anomaly": anomaly,
                            "umap_x": float(xs[i]),
                            "umap_y": float(ys[i]),
                        }
                    )

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows)
    out_df = out_df.drop_duplicates(subset=["report_id", "cluster_label", "umap_x", "umap_y"], keep="last")
    return out_df


def _plotly_layout_title_text(layout: dict) -> str:
    title = layout.get("title") if isinstance(layout, dict) else None
    if isinstance(title, dict):
        return str(title.get("text") or "")
    return str(title or "")


def _parse_cluster_id_from_notebook_trend_title(title_text: str) -> Optional[int]:
    """
    Notebook titles look like:
    'Cluster 17 — <label> — Normalized Rate Trend' (em dash U+2014 between sections).
    """
    m = re.search(r"Cluster\s+(\d+)\s*\u2014", title_text)
    if not m:
        m = re.search(r"Cluster\s+(\d+)\s*-\s*.+\s*-\s*Normalized Rate Trend", title_text)
    if not m:
        return None
    return int(m.group(1))


def _load_monthly_trends_from_notebook(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Read per-cluster monthly count + normalized rate series from Plotly outputs whose layout title
    ends with 'Normalized Rate Trend'. Cluster id is taken from the figure title so truncated
    customdata labels still align with cluster_labels_report.csv.

    The notebook uses either count on y (with rate in customdata) or normalized rate on y (with count
    in customdata); both are supported.
    """
    nb_path = _root_dir() / "worksofar_updated.ipynb"
    if not nb_path.exists():
        return pd.DataFrame()

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    id_to_label = (
        cluster_summary.drop_duplicates("cluster").set_index("cluster")["cluster_label"].to_dict()
        if not cluster_summary.empty
        else {}
    )

    rows = []
    roll_rows: list[dict] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            fig = out.get("data", {}).get("application/vnd.plotly.v1+json")
            if not fig:
                continue
            layout = fig.get("layout", {}) or {}
            title_text = _plotly_layout_title_text(layout)
            if "Normalized Rate Trend" not in title_text:
                continue
            cluster_id = _parse_cluster_id_from_notebook_trend_title(title_text)
            if cluster_id is None:
                continue

            # Second trace in notebook: rolling mean of normalized rate (red line).
            for trace in fig.get("data", []):
                if str(trace.get("name", "")).strip() != "Rolling Mean Rate":
                    continue
                x_roll = _decode_plotly_array(trace.get("x"))
                y_roll = _decode_plotly_array(trace.get("y"))
                n_roll = min(len(x_roll), len(y_roll))
                for i in range(n_roll):
                    ts = pd.to_datetime(x_roll[i], errors="coerce")
                    if pd.isna(ts):
                        continue
                    mk = ts.strftime("%Y-%m")
                    yv = y_roll[i]
                    if yv is None or (isinstance(yv, (float, np.floating)) and np.isnan(float(yv))):
                        rmean = np.nan
                    else:
                        try:
                            rmean = float(yv)
                        except (TypeError, ValueError):
                            rmean = np.nan
                    roll_rows.append(
                        {"cluster": cluster_id, "month_key": mk, "rolling_mean_rate": rmean}
                    )
                break

            for trace in fig.get("data", []):
                hover = str(trace.get("hovertemplate", ""))
                if "month_key=%{x}" not in hover:
                    continue
                is_count_y = "count=%{y}" in hover
                is_rate_y = "rate=%{y" in hover
                if not (is_count_y or is_rate_y):
                    continue

                x_vals = _decode_plotly_array(trace.get("x"))
                y_vals = _decode_plotly_array(trace.get("y"))
                cds = trace.get("customdata", [])
                n = min(len(x_vals), len(y_vals), len(cds))
                if n == 0:
                    continue

                for i in range(n):
                    cd = cds[i] if i < len(cds) else []
                    ts = pd.to_datetime(x_vals[i], errors="coerce")
                    if pd.isna(ts):
                        continue
                    month_key = ts.strftime("%Y-%m")

                    if is_count_y:
                        count = int(y_vals[i])
                        total_reports_month = cd[1] if len(cd) > 1 else None
                        normalized_rate = float(cd[2]) if len(cd) > 2 and cd[2] is not None else None
                    else:
                        normalized_rate = float(y_vals[i])
                        count = int(cd[1]) if len(cd) > 1 and cd[1] is not None else None
                        total_reports_month = cd[2] if len(cd) > 2 else None

                    if total_reports_month is not None:
                        try:
                            total_reports_month = int(total_reports_month)
                        except (TypeError, ValueError):
                            total_reports_month = None
                    if count is None and normalized_rate is not None and total_reports_month:
                        count = int(round(normalized_rate * total_reports_month))
                    if normalized_rate is None and count is not None and total_reports_month:
                        normalized_rate = float(count) / float(total_reports_month)

                    # z_rate in notebook hover (customdata[6]) drives spike markers; CSV spikes often omit clusters 22/27.
                    max_rate_z = np.nan
                    if len(cd) > 6 and cd[6] is not None:
                        try:
                            max_rate_z = float(cd[6])
                        except (TypeError, ValueError):
                            max_rate_z = np.nan

                    rows.append(
                        {
                            "month_key": month_key,
                            "cluster": cluster_id,
                            "cluster_label": id_to_label.get(cluster_id, ""),
                            "count": 0 if count is None else int(count),
                            "total_reports_month": total_reports_month,
                            "normalized_rate": 0.0 if normalized_rate is None else float(normalized_rate),
                            "max_rate_z": max_rate_z,
                        }
                    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["cluster", "month_key"], keep="last")
    df["cluster_label"] = df["cluster"].map(id_to_label).fillna(df["cluster_label"])
    if roll_rows:
        rdf = pd.DataFrame(roll_rows)
        rdf = rdf.drop_duplicates(subset=["cluster", "month_key"], keep="last")
        df = df.merge(rdf, on=["cluster", "month_key"], how="left")
    else:
        df["rolling_mean_rate"] = np.nan
    return df


def _build_clustered_reports(cluster_summary: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    prepared = _preferred_file("clustered_reports.csv")
    if prepared.exists():
        df = pd.read_csv(prepared)
        _require_columns(
            df,
            ["report_id", "date", "month_key", "text", "cluster", "cluster_label", "umap_x", "umap_y"],
            prepared.name,
        )
        return df

    notebook_df = _load_clustered_reports_from_notebook(cluster_summary)
    if not notebook_df.empty:
        return notebook_df

    rows = []
    for _, row in cluster_summary.head(25).iterrows():
        cluster = int(row["cluster"])
        label = row["cluster_label"]
        size = int(max(15, min(150, row["cluster_size"] // 8 if row["cluster_size"] else 20)))
        rng = np.random.default_rng(cluster + 7)
        center_x, center_y = rng.normal(0, 4), rng.normal(0, 4)
        cluster_trends = trends[trends["cluster"] == cluster]
        month_pool = cluster_trends["month_key"].tolist() or ["2025-11"]
        for i in range(size):
            month = month_pool[i % len(month_pool)]
            date = f"{month}-15"
            rows.append(
                {
                    "report_id": f"C{cluster}-{i+1}",
                    "date": date,
                    "month_key": month,
                    "text": (row.get("description", "") or "")[:220],
                    "cluster": cluster,
                    "cluster_label": label,
                    "vessel_type": "Commercial Fixed Wing",
                    "umap_x": center_x + rng.normal(0, 0.9),
                    "umap_y": center_y + rng.normal(0, 0.9),
                }
            )
    return pd.DataFrame(rows)


def _load_alerts_file() -> pd.DataFrame:
    path = _preferred_file("alerts.csv")
    if not path.exists():
        return pd.DataFrame(columns=["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"])
    df = pd.read_csv(path)
    _require_columns(df, ["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"], path.name)
    return df


def report_noise_pct(reports: pd.DataFrame, cluster_labels: Optional[pd.DataFrame] = None) -> float:
    """
    Fraction of reports that are HDBSCAN-style noise: prefer ``cluster == -1`` or label ``Noise``;
    else pipeline rows with ``cluster == 0`` and label ``Noise`` only.
    If the report table has no noise rows but ``cluster_labels`` sizes sum below the report count,
    infer noise as the residual (notebook exports often omit a ``-1`` row from Plotly traces).
    """
    if reports is None or reports.empty or "cluster" not in reports.columns:
        return 0.0
    total = int(len(reports))
    if total <= 0:
        return 0.0
    c = pd.to_numeric(reports["cluster"], errors="coerce")
    if (c == -1).any():
        return float(100.0 * (c == -1).sum() / total)
    if "cluster_label" in reports.columns:
        lbl = reports["cluster_label"].astype(str).str.strip()
        if lbl.str.fullmatch("(?i)noise").any():
            return float(100.0 * lbl.str.fullmatch("(?i)noise").sum() / total)
        m0 = (c == 0) & lbl.str.lower().eq("noise")
        if m0.any():
            return float(100.0 * m0.sum() / total)
    if cluster_labels is not None and "cluster_size" in cluster_labels.columns:
        ls = int(pd.to_numeric(cluster_labels["cluster_size"], errors="coerce").fillna(0).sum())
        if 0 < ls < total:
            return float(100.0 * (total - ls) / total)
    return 0.0


def _load_notebook_kpis() -> Dict[str, float]:
    kpis: Dict[str, float] = {}
    reports_path = _preferred_file("clustered_reports.csv")
    summary_path = _preferred_file("cluster_summary.csv")
    if reports_path.exists():
        try:
            reports = pd.read_csv(reports_path)
            kpis["total_incidents"] = float(len(reports))
            lbls = _load_cluster_labels_report()
            kpis["noise_pct"] = report_noise_pct(reports, lbls)
        except Exception:
            pass
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if "is_noise" in summary.columns:
                non_noise = summary[~summary["is_noise"].astype(bool)]
                kpis["cluster_count_excl_noise"] = float(non_noise["cluster"].nunique())
            else:
                kpis["cluster_count_excl_noise"] = float(summary["cluster"].nunique())
        except Exception:
            pass
    return kpis


def load_relevant_regulations_from_notebook(limit: int = 4) -> list[dict]:
    spikes = _load_recent_spikes_report()
    if spikes.empty:
        return []
    first = spikes.sort_values("max_rate_z", ascending=False).iloc[0]
    label = str(first.get("cluster_label", ""))
    root = label
    regs = load_cluster_regulations_rag(cluster_label=label, root_cause=root, top_k=limit)
    return [
        {"title": str(r.get("citation", "")), "relevance": float(r.get("relevance", 0.0)), "snippet": str(r.get("text", ""))}
        for r in regs
    ]


def load_top_spike_clusters_from_notebook(limit: int = 3) -> list[int]:
    spikes = _load_recent_spikes_report()
    if spikes.empty:
        return []
    ids = []
    for _, row in spikes.sort_values("max_rate_z", ascending=False).iterrows():
        cid = int(row["cluster"])
        if cid not in ids:
            ids.append(cid)
        if len(ids) >= limit:
            break
    return ids


def load_top_spike_rows_from_notebook(limit: int = 3) -> pd.DataFrame:
    spikes = _load_recent_spikes_report()
    if spikes.empty:
        return pd.DataFrame(columns=["cluster", "month_key", "count", "month_total", "rate", "rate_z", "rate_growth_pct"])

    trends_path = _preferred_file("monthly_cluster_trends.csv")
    trends = pd.read_csv(trends_path) if trends_path.exists() else pd.DataFrame()
    month_totals = (
        trends.drop_duplicates("month_key").set_index("month_key")["total_reports_month"].to_dict()
        if not trends.empty and "total_reports_month" in trends.columns
        else {}
    )

    top = spikes.sort_values("max_rate_z", ascending=False).head(limit).copy()
    top["month_total"] = top["month_key"].map(month_totals).fillna(0)
    out = pd.DataFrame(
        {
            "cluster": top["cluster"].astype(int),
            "month_key": top["month_key"].astype(str),
            "count": top["count"].astype(float),
            "month_total": top["month_total"].astype(float),
            "rate": top["normalized_rate"].astype(float),
            "rate_z": top["max_rate_z"].astype(float),
            "rate_growth_pct": 0.0,
        }
    )
    return out


def _faa_chroma_dir() -> Path:
    env = os.environ.get("FAA_CHROMA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _root_dir() / "chroma_faa_regulations"


def _faa_rag_mode() -> str:
    """chroma | hybrid | keyword — default hybrid (embedding + lexical boost)."""
    m = os.environ.get("FAA_RAG_MODE", "hybrid").strip().lower()
    return m if m in {"chroma", "hybrid", "keyword"} else "hybrid"


def _extract_regulation_query_terms(cluster_label: str, root_cause: str) -> list[str]:
    query = f"{cluster_label} {root_cause}".lower()
    terms = [t for t in re.findall(r"[a-z0-9]{3,}", query) if t not in {"the", "and", "for", "with", "from"}]
    if not terms:
        terms = re.findall(r"[a-z0-9]{3,}", cluster_label.lower())
    return terms


def _keyword_overlap_score(chunk_lower: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    hits = sum(chunk_lower.count(t) for t in terms)
    return min(1.0, hits / max(8.0, len(terms) * 2.5))


def _keyword_regulation_chunks(cluster_label: str, root_cause: str, top_k: int) -> list[dict[str, Any]]:
    """
    Lexical fallback over all files under ``faa_corpus`` (txt/pdf), not only Part 91.
    This keeps citations useful even when embedding retrieval is unavailable.
    """
    corpus_dir = _root_dir() / "faa_corpus"
    if not corpus_dir.is_dir():
        return []

    terms = _extract_regulation_query_terms(cluster_label, root_cause)
    if not terms:
        return []

    def _read_text(path: Path) -> str:
        suf = path.suffix.lower()
        if suf == ".txt":
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        if suf == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                parts: list[str] = []
                for pg in reader.pages:
                    try:
                        parts.append(pg.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(parts)
            except Exception:
                return ""
        return ""

    chunk_size = 1200
    step = 900
    scored: list[tuple[int, str, str]] = []  # (hits, chunk, source_file)
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".pdf"} or not path.is_file():
            continue
        raw = re.sub(r"\s+", " ", _read_text(path)).strip()
        if not raw:
            continue
        chunks = [raw[i : i + chunk_size] for i in range(0, max(1, len(raw) - chunk_size + 1), step)]
        if not chunks:
            chunks = [raw[:chunk_size]]
        for ch in chunks:
            low = ch.lower()
            hits = sum(low.count(t) for t in terms)
            if hits > 0:
                scored.append((hits, ch, path.name))

    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    max_hits = max(1, scored[0][0])

    out: list[dict[str, Any]] = []
    for hits, ch, source_name in scored[: max(top_k * 3, top_k)]:
        out.append(
            {
                "citation": f"{source_name} (lexical)",
                "doc_type": "KEYWORD",
                "source_file": source_name,
                "relevance": round(min(0.99, 0.45 + (hits / max_hits) * 0.35), 4),
                "text": ch[:700],
            }
        )
    return _dedupe_hits(out, top_k)


def _import_query_trend_for_regulations():
    root = _root_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from faa_regulation_rag import query_trend_for_regulations

    return query_trend_for_regulations


def _chroma_regulation_hits(trend_text: str, top_k: int) -> list[dict[str, Any]]:
    chroma_dir = _faa_chroma_dir()
    if not chroma_dir.is_dir():
        return []
    query_fn = _import_query_trend_for_regulations()
    euron_key = (os.environ.get("EURON_API_KEY") or os.environ.get("EURON_EMBEDDINGS_API_KEY") or "").strip()
    euron_url = (os.environ.get("EURON_EMBEDDINGS_URL") or "").strip()
    euron_model = (os.environ.get("EURON_EMBEDDINGS_MODEL") or "").strip()
    # IMPORTANT: must match ingest embedding space.
    # Notebook FAA ingest uses BAAI/bge-base-en-v1.5 by default in this project.
    embed_model_env = (os.environ.get("FAA_EMBEDDING_MODEL") or "").strip()
    model_candidates = [m for m in [embed_model_env, "BAAI/bge-base-en-v1.5", "all-MiniLM-L6-v2"] if m]
    collection = (os.environ.get("FAA_CHROMA_COLLECTION") or "faa_regulations").strip()
    seen: set[str] = set()
    for embed_model in model_candidates:
        if embed_model in seen:
            continue
        seen.add(embed_model)
        try:
            hits = query_fn(
                trend_text,
                chroma_persist_dir=str(chroma_dir),
                embedding_model_name=embed_model,
                collection_name=collection,
                top_k=top_k,
                euron_api_key=euron_key or None,
                euron_embeddings_url=euron_url or None,
                euron_embeddings_model=euron_model or None,
            )
            if hits:
                return hits
        except Exception:
            continue
    return []


def _normalize_chroma_hit(h: dict[str, Any]) -> dict[str, Any]:
    cite = str(h.get("citation") or h.get("source_file") or "FAA corpus")
    return {
        "citation": cite,
        "doc_type": str(h.get("doc_type", "")),
        "source_file": str(h.get("source_file", "")),
        "relevance": float(h.get("relevance", 0.0)),
        "text": str(h.get("text", ""))[:700],
    }


def _merge_hybrid_rank(
    chroma_hits: list[dict[str, Any]],
    terms: list[str],
    top_k: int,
    embed_weight: float,
) -> list[dict[str, Any]]:
    """Re-rank Chroma hits using a small lexical boost so citations stay embedding-ordered but query terms matter."""
    out: list[dict[str, Any]] = []
    w = max(0.0, min(1.0, embed_weight))
    for h in chroma_hits:
        base = _normalize_chroma_hit(h)
        low = base["text"].lower()
        lex = _keyword_overlap_score(low, terms)
        emb = float(base["relevance"])
        base["relevance"] = round(min(0.99, w * emb + (1.0 - w) * lex), 4)
        out.append(base)
    out.sort(key=lambda x: -x["relevance"])
    return out[:top_k]


def _dedupe_hits(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in hits:
        key = f"{h.get('citation', '')}|{(h.get('text') or '')[:120]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def load_cluster_regulations_rag(cluster_label: str, root_cause: str, top_k: int = 4) -> list[dict]:
    """
    FAA regulation RAG for dashboard: prefer Chroma + embedding retrieval (same index as ``faa_regulation_rag``),
    with optional hybrid lexical boost. Set ``FAA_RAG_MODE=keyword`` to force legacy keyword-only behavior.

    Environment:
    - ``FAA_CHROMA_DIR``: Chroma persist directory (default: ``<project>/chroma_faa_regulations``)
    - ``FAA_EMBEDDING_MODEL``: must match ingest (default ``all-MiniLM-L6-v2``)
    - ``FAA_CHROMA_COLLECTION``: collection name (default ``faa_regulations``)
    - ``FAA_RAG_MODE``: ``chroma`` | ``hybrid`` | ``keyword``
    - ``FAA_RAG_HYBRID_EMBED_WEIGHT``: 0–1, weight on embedding vs lexical in hybrid (default 0.78)
    - EurON (optional): ``EURON_API_KEY``, ``EURON_EMBEDDINGS_URL``, ``EURON_EMBEDDINGS_MODEL``
    """
    mode = _faa_rag_mode()
    trend_text = f"{cluster_label}\n{root_cause}".strip()
    terms = _extract_regulation_query_terms(cluster_label, root_cause)
    embed_w = float(os.environ.get("FAA_RAG_HYBRID_EMBED_WEIGHT", "0.78"))
    embed_w = max(0.0, min(1.0, embed_w))

    if mode == "keyword":
        return _keyword_regulation_chunks(cluster_label, root_cause, top_k)

    fetch_k = max(top_k * 2, top_k + 4)
    chroma_raw = _chroma_regulation_hits(trend_text, fetch_k)

    if not chroma_raw:
        return _keyword_regulation_chunks(cluster_label, root_cause, top_k)

    if mode == "chroma":
        ranked = [_normalize_chroma_hit(h) for h in chroma_raw]
        ranked.sort(key=lambda x: -x["relevance"])
        return _dedupe_hits(ranked, top_k)

    # hybrid: Chroma + lexical boost, optional fill from keyword if sparse
    boosted = _merge_hybrid_rank(chroma_raw, terms, top_k, embed_weight=embed_w)
    if len(boosted) < top_k:
        kw_extra = _keyword_regulation_chunks(cluster_label, root_cause, top_k * 2)
        merged = boosted + kw_extra
        merged.sort(key=lambda x: -float(x.get("relevance", 0.0)))
        return _dedupe_hits(merged, top_k)
    return boosted


def cluster_color_map(cluster_labels: Iterable[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    mapping: Dict[str, str] = {}
    for idx, label in enumerate(sorted(set(str(x) for x in cluster_labels))):
        mapping[label] = palette[idx % len(palette)]
    return mapping


def load_data_bundle() -> DataBundle:
    cluster_labels = _load_cluster_labels_report()
    spikes = _load_recent_spikes_report()
    cluster_summary = _build_cluster_summary(cluster_labels)
    trends = _build_monthly_trends(spikes, cluster_summary)
    clustered_reports = _build_clustered_reports(cluster_summary, trends)
    alerts = _load_alerts_file()
    notebook_kpis = _load_notebook_kpis()
    return DataBundle(
        clustered_reports=clustered_reports,
        monthly_cluster_trends=trends,
        cluster_summary=cluster_summary,
        alerts=alerts,
        cluster_labels_report=cluster_labels,
        notebook_kpis=notebook_kpis,
    )


def latest_month_pair(month_keys: pd.Series) -> Tuple[str, str]:
    keys = sorted([k for k in month_keys.dropna().astype(str).unique() if k])
    if not keys:
        return "", ""
    if len(keys) == 1:
        return keys[0], keys[0]
    return keys[-1], keys[-2]
