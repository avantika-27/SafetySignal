"""
Single-page dashboard section renderers (Overview, Cluster map, Trend, Research).
Plotly charts use plotly_dark for consistency with the app theme.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.alerts_engine import build_rule_based_alerts
from utils.bulletin_export import export_bulletins, recommended_actions
from utils.data_loader import (
    DataBundle,
    cluster_color_map,
    latest_month_pair,
    load_cluster_regulations_rag,
    load_top_spike_clusters_from_notebook,
    load_top_spike_rows_from_notebook,
    report_noise_pct,
)

_PLOTLY_TEMPLATE = "plotly_dark"


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_cluster_regulations(cluster_label: str, root_cause: str, top_k: int) -> list[dict]:
    """Cache expensive regulation retrieval so alert UI remains responsive."""
    return load_cluster_regulations_rag(
        cluster_label=str(cluster_label),
        root_cause=str(root_cause),
        top_k=int(top_k),
    )


def _phase_keywords_for_cluster(bundle: DataBundle, cluster_id: int) -> Tuple[Optional[str], Optional[str]]:
    lbl_df = bundle.cluster_labels_report
    if lbl_df.empty or "cluster" not in lbl_df.columns:
        return None, None
    match = lbl_df[lbl_df["cluster"].astype(int) == int(cluster_id)]
    if match.empty:
        return None, None
    tp = match.iloc[0].get("top_phase")
    kw = match.iloc[0].get("keywords")
    tp_out = None if tp is None or (isinstance(tp, float) and pd.isna(tp)) else str(tp).strip() or None
    kw_out = None if kw is None or (isinstance(kw, float) and pd.isna(kw)) else str(kw).strip() or None
    return tp_out, kw_out


def render_overview(bundle: DataBundle) -> None:
    reports = bundle.clustered_reports.copy()
    trends = bundle.monthly_cluster_trends.copy()
    summary = bundle.cluster_summary.copy()
    labels = bundle.cluster_labels_report.copy()

    latest_month, previous_month = latest_month_pair(trends["month_key"])
    latest_trends = trends[trends["month_key"] == latest_month].copy()

    noise_mask = summary["is_noise"].fillna(False)
    cluster_count = int(summary.loc[~noise_mask, "cluster"].nunique())
    total_incidents = int(len(reports))

    labels["cluster_size"] = pd.to_numeric(labels.get("cluster_size"), errors="coerce")
    noise_pct = report_noise_pct(reports, labels)

    nb_kpis = bundle.notebook_kpis or {}
    if "cluster_count_excl_noise" in nb_kpis:
        cluster_count = int(nb_kpis["cluster_count_excl_noise"])

    if not latest_trends.empty:
        _em = latest_trends.sort_values("pct_change", ascending=False).iloc[0]
        top_emerging_full = f"{str(_em['cluster_label'])} ({float(_em['pct_change']):+.1f}%)"
        highest_spike_text = top_emerging_full
    else:
        top_emerging_full = "N/A"
        highest_spike_text = "N/A"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total incidents", f"{total_incidents:,}")
    k2.metric("Clusters", f"{cluster_count:,}")
    k3.metric("Noise", f"{noise_pct:.1f}%")
    k4.metric("Top emerging", top_emerging_full)
    if top_emerging_full and top_emerging_full != "N/A":
        k4.markdown(
            f'<div title="{html_escape(top_emerging_full)}" '
            'style="margin-top:-0.25rem;font-size:0.78rem;color:#8b949e;line-height:1.25;white-space:normal;">'
            f'{html_escape(top_emerging_full)}</div>',
            unsafe_allow_html=True,
        )
    k5.metric("Strongest spike", highest_spike_text)
    if highest_spike_text and highest_spike_text != "N/A":
        k5.markdown(
            f'<div title="{html_escape(highest_spike_text)}" '
            'style="margin-top:-0.25rem;font-size:0.78rem;color:#8b949e;line-height:1.25;white-space:normal;">'
            f'{html_escape(highest_spike_text)}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([2, 1])
    with left:
        # Use actual report rows as source-of-truth for narrative volume.
        top_source = reports.copy()
        if "cluster" in top_source.columns:
            cnum = pd.to_numeric(top_source["cluster"], errors="coerce")
            top_source = top_source[cnum.ne(-1)]
        if "cluster_label" in top_source.columns:
            top_source = top_source[~top_source["cluster_label"].astype(str).str.strip().str.lower().eq("noise")]
        if not top_source.empty and "cluster_label" in top_source.columns:
            top3 = (
                top_source.groupby("cluster_label", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
                .head(3)
            )
        else:
            top3 = (
                labels[["cluster_label", "cluster_size"]]
                .rename(columns={"cluster_size": "count"})
                .sort_values("count", ascending=False)
                .head(3)
            )
        fig = px.bar(top3, x="cluster_label", y="count", title="Top clusters by narrative volume", template=_PLOTLY_TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Operational snapshot**")
        st.markdown(
            f"- Volume: **{total_incidents:,}** narratives in this snapshot.\n"
            f"- Leading shift: **{highest_spike_text}**.\n"
            f"- Focus clusters with sustained positive normalized rates."
        )


def render_cluster_map(bundle: DataBundle) -> None:
    df = bundle.clustered_reports.copy()
    trends = bundle.monthly_cluster_trends.copy()

    if df.empty:
        st.warning("No clustered report coordinates to plot.")
        return

    months = sorted(df["month_key"].dropna().unique().tolist())
    clusters = sorted(df["cluster_label"].dropna().unique().tolist())
    vessels = sorted(df["vessel_type"].dropna().unique().tolist()) if "vessel_type" in df.columns else []

    c1, c2, c3, c4 = st.columns(4)
    selected_month = c1.selectbox("Month", ["All"] + months, index=0, key="map_month")
    selected_cluster = c2.selectbox("Cluster", ["All"] + clusters, index=0, key="map_cluster")
    selected_vessel = c3.selectbox("Vessel type", ["All"] + vessels, index=0, key="map_vessel") if vessels else "All"
    hide_noise = c4.toggle("Hide HDBSCAN noise (−1)", value=False, key="map_hide_noise")

    filtered = df.copy()
    if selected_month != "All":
        filtered = filtered[filtered["month_key"] == selected_month]
    if selected_cluster != "All":
        filtered = filtered[filtered["cluster_label"] == selected_cluster]
    if selected_vessel != "All" and "vessel_type" in filtered.columns:
        filtered = filtered[filtered["vessel_type"] == selected_vessel]
    if hide_noise and "cluster" in filtered.columns:
        cnum = pd.to_numeric(filtered["cluster"], errors="coerce")
        filtered = filtered[cnum.ne(-1)]

    colors = cluster_color_map(filtered["cluster_label"].tolist())
    fig = px.scatter(
        filtered,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        color_discrete_map=colors,
        hover_data=["report_id", "cluster_label", "date", "month_key", "text"],
        title="Semantic map — each point is one ASRS-aligned narrative",
        template=_PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.75})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Notebook spike clusters — rate snapshot**")
    if trends.empty:
        st.info("No trend data.")
        return
    # Keep this snapshot aligned with rule-based alert ranking used elsewhere in the dashboard.
    top_alerts = build_rule_based_alerts(trends, bundle.cluster_labels_report).head(3).copy()
    if not top_alerts.empty:
        top_spike = top_alerts[["cluster", "cluster_label", "month_key", "growth_pct", "confidence"]].copy()
        top_spike["cluster"] = pd.to_numeric(top_spike["cluster"], errors="coerce").fillna(-1).astype(int)
        top_spike = top_spike.rename(columns={"cluster": "cluster_id"})
    else:
        top_spike_ids = load_top_spike_clusters_from_notebook(limit=3)
        if top_spike_ids:
            label_map = (
                bundle.cluster_summary[["cluster", "cluster_label"]]
                .drop_duplicates("cluster")
                .set_index("cluster")["cluster_label"]
                .to_dict()
            )
            rows = [{"cluster_id": cid, "cluster_label": label_map.get(cid, f"Cluster {cid}")} for cid in top_spike_ids]
            top_spike = pd.DataFrame(rows)
        elif "max_rate_z" in trends.columns and trends["max_rate_z"].notna().any():
            top_spike = trends.sort_values(["max_rate_z", "normalized_rate"], ascending=[False, False]).head(3)[
                ["month_key", "cluster_label", "normalized_rate", "max_rate_z"]
            ]
        else:
            top_spike = (
                trends.sort_values(["month_key", "pct_change"], ascending=[False, False])
                .drop_duplicates(["cluster"])
                .head(3)[["month_key", "cluster_label", "normalized_rate", "pct_change"]]
            )
    st.dataframe(top_spike, use_container_width=True, hide_index=True)


def render_time_series(bundle: DataBundle) -> None:
    trends = bundle.monthly_cluster_trends.copy()
    top_spike_ids = load_top_spike_clusters_from_notebook(limit=3)
    top_spike_rows = load_top_spike_rows_from_notebook(limit=3)
    top_alerts = build_rule_based_alerts(trends, bundle.cluster_labels_report).head(3).copy()

    id_set = {int(x) for x in trends["cluster"].dropna().unique()}
    preferred_ids: list[int] = []
    if not top_alerts.empty and "cluster" in top_alerts.columns:
        for cid in top_alerts["cluster"].tolist():
            c = int(cid)
            if c in id_set and c not in preferred_ids:
                preferred_ids.append(c)
    id_to_label = {
        int(r.cluster): str(r.cluster_label)
        for r in trends.drop_duplicates("cluster")[["cluster", "cluster_label"]].itertuples(index=False)
    }
    ordered_cluster_ids: list[int] = []
    primary_ids = preferred_ids or (top_spike_ids or [])
    for cid in primary_ids:
        c = int(cid)
        if c in id_set and c not in ordered_cluster_ids:
            ordered_cluster_ids.append(c)
    for c in sorted(id_set):
        if c not in ordered_cluster_ids:
            ordered_cluster_ids.append(c)

    default_idx = 0
    if primary_ids:
        for i, cid in enumerate(ordered_cluster_ids):
            if cid in [int(x) for x in primary_ids]:
                default_idx = i
                break
    elif "max_rate_z" in trends.columns and trends["max_rate_z"].notna().any():
        top_c = int(trends.sort_values(["max_rate_z", "normalized_rate"], ascending=[False, False])["cluster"].iloc[0])
        if top_c in ordered_cluster_ids:
            default_idx = ordered_cluster_ids.index(top_c)

    if not ordered_cluster_ids:
        st.info("No cluster monthly trends are available.")
        return

    def _fmt(cid: int) -> str:
        return f"Cluster {cid} — {id_to_label.get(cid, '')}"

    selected_id = st.selectbox(
        "Select cluster",
        options=ordered_cluster_ids,
        index=min(default_idx, len(ordered_cluster_ids) - 1),
        format_func=_fmt,
        key="ts_cluster",
    )
    selected_label = str(id_to_label.get(selected_id, ""))

    cluster_df = trends[trends["cluster"] == selected_id].sort_values("month_key").copy()
    if not cluster_df.empty:
        month_index = pd.period_range(cluster_df["month_key"].min(), cluster_df["month_key"].max(), freq="M").astype(str)
        cluster_df = (
            cluster_df.set_index("month_key")
            .reindex(month_index)
            .rename_axis("month_key")
            .reset_index()
        )
        cluster_df["cluster_label"] = selected_label
        cluster_df["cluster"] = selected_id
        cluster_df["count"] = cluster_df["count"].fillna(0).astype(float)
        if "total_reports_month" in cluster_df.columns:
            cluster_df["total_reports_month"] = cluster_df["total_reports_month"].fillna(0)
        if "normalized_rate" not in cluster_df.columns:
            cluster_df["normalized_rate"] = 0.0
        cluster_df["normalized_rate"] = cluster_df["normalized_rate"].fillna(0.0)
        cluster_df["pct_change"] = cluster_df["pct_change"].fillna(0.0)
        if "rolling_mean_rate" in cluster_df.columns:
            cluster_df["rolling_mean_rate"] = pd.to_numeric(cluster_df["rolling_mean_rate"], errors="coerce")
        else:
            cluster_df["rolling_mean_rate"] = pd.Series(float("nan"), index=cluster_df.index)

    chart_title = f"Cluster {selected_id} — {selected_label} — normalized share of monthly reports"

    if cluster_df.empty:
        st.info("No time series rows for this cluster.")
        fig = go.Figure()
        fig.update_layout(title=chart_title, template=_PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        return

    month_tot = (
        cluster_df["total_reports_month"].fillna(0).astype(int)
        if "total_reports_month" in cluster_df.columns
        else pd.Series(0, index=cluster_df.index)
    )
    custom = list(zip(cluster_df["count"].fillna(0).astype(int), month_tot))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cluster_df["month_key"],
            y=cluster_df["normalized_rate"],
            mode="lines+markers",
            name="Normalized rate",
            customdata=custom,
            hovertemplate=(
                "month=%{x}<br>rate=%{y:.4f}<br>count=%{customdata[0]}<br>month_total=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    roll_series = cluster_df["rolling_mean_rate"] if "rolling_mean_rate" in cluster_df.columns else None
    if roll_series is not None and roll_series.notna().any():
        fig.add_trace(
            go.Scatter(
                x=cluster_df["month_key"],
                y=roll_series,
                mode="lines",
                name="Rolling mean",
                line={"color": "#ff9f43", "width": 2},
                connectgaps=False,
                hovertemplate="month=%{x}<br>rolling=%{y:.4f}<extra></extra>",
            )
        )
    else:
        roll_fb = cluster_df["normalized_rate"].rolling(3, min_periods=2).mean()
        if roll_fb.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=cluster_df["month_key"],
                    y=roll_fb,
                    mode="lines",
                    name="Rolling mean (3-mo)",
                    line={"color": "#ff9f43", "width": 2, "dash": "dash"},
                    connectgaps=False,
                    hovertemplate="month=%{x}<br>rolling=%{y:.4f}<extra></extra>",
                )
            )
    if "max_rate_z" in cluster_df.columns:
        spike_df = cluster_df[cluster_df["max_rate_z"].fillna(0) > 1.5]
        if not spike_df.empty:
            stot = (
                spike_df["total_reports_month"].fillna(0).astype(int)
                if "total_reports_month" in spike_df.columns
                else pd.Series(0, index=spike_df.index)
            )
            sp_custom = list(zip(spike_df["count"].fillna(0).astype(int), stot))
            fig.add_trace(
                go.Scatter(
                    x=spike_df["month_key"],
                    y=spike_df["normalized_rate"],
                    mode="markers",
                    name="Spike month",
                    marker={"size": 13, "symbol": "diamond", "color": "#00d4aa"},
                    customdata=sp_custom,
                    hovertemplate=(
                        "month=%{x}<br>rate=%{y:.4f}<br>count=%{customdata[0]}<br>month_total=%{customdata[1]}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title=chart_title,
        xaxis_title="Month",
        yaxis_title="Cluster share of monthly reports",
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    latest_month, previous_month = latest_month_pair(cluster_df["month_key"])
    current_row = cluster_df[cluster_df["month_key"] == latest_month]
    prev_row = cluster_df[cluster_df["month_key"] == previous_month]

    if not top_spike_rows.empty:
        spike_label_map = (
            trends[["cluster", "cluster_label"]]
            .drop_duplicates("cluster")
            .set_index("cluster")["cluster_label"]
            .to_dict()
        )
        spike_rows_with_labels = top_spike_rows.copy()
        spike_rows_with_labels["cluster"] = spike_rows_with_labels["cluster"].astype(int)
        spike_rows_with_labels["cluster_label"] = spike_rows_with_labels["cluster"].map(spike_label_map)
        spike_for_selected = spike_rows_with_labels[spike_rows_with_labels["cluster"].astype(int) == int(selected_id)]
        if not spike_for_selected.empty:
            spike_row = spike_for_selected.iloc[0]
            spike_month = spike_row["month_key"]
            current_row = cluster_df[cluster_df["month_key"] == spike_month]
            prev_month_for_spike = (pd.Period(spike_month, freq="M") - 1).strftime("%Y-%m")
            prev_row = cluster_df[cluster_df["month_key"] == prev_month_for_spike]

    current_count = float(current_row["count"].iloc[0]) if not current_row.empty else 0.0
    prev_count = float(prev_row["count"].iloc[0]) if not prev_row.empty else 0.0
    current_rate = float(current_row["normalized_rate"].iloc[0]) if not current_row.empty else 0.0
    prev_rate = float(prev_row["normalized_rate"].iloc[0]) if not prev_row.empty else 0.0

    # If latest month is zero for this cluster, show the most recent active month in KPI cards.
    kpi_month = latest_month
    if current_count <= 0 and "count" in cluster_df.columns:
        active = cluster_df[pd.to_numeric(cluster_df["count"], errors="coerce").fillna(0) > 0].copy()
        if not active.empty:
            kpi_month = str(active["month_key"].iloc[-1])
            current_row = cluster_df[cluster_df["month_key"] == kpi_month]
            prev_for_kpi = (pd.Period(kpi_month, freq="M") - 1).strftime("%Y-%m")
            prev_row = cluster_df[cluster_df["month_key"] == prev_for_kpi]
            current_count = float(current_row["count"].iloc[0]) if not current_row.empty else 0.0
            prev_count = float(prev_row["count"].iloc[0]) if not prev_row.empty else 0.0
            current_rate = float(current_row["normalized_rate"].iloc[0]) if not current_row.empty else 0.0
            prev_rate = float(prev_row["normalized_rate"].iloc[0]) if not prev_row.empty else 0.0
    count_delta = current_count - prev_count
    rate_delta_pct = ((current_rate - prev_rate) / prev_rate * 100) if prev_rate else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Count vs prior month", f"{int(current_count)}", delta=f"{count_delta:+.0f}")
    m2.metric("Normalized rate", f"{current_rate:.3f}", delta=f"{rate_delta_pct:+.1f}%")
    if kpi_month != latest_month:
        st.caption(f"KPI month adjusted to {kpi_month} (latest active month for this cluster).")

    st.markdown("**Latest month — cross-cluster snapshot**")
    if not trends.empty:
        lm = sorted(trends["month_key"].dropna().unique())[-1]
        latest_df = trends[trends["month_key"] == lm].copy()
        if preferred_ids and not top_alerts.empty:
            rate_map = trends.set_index(["cluster", "month_key"])[["normalized_rate"]].to_dict("index")
            snapshot_rows = []
            cluster_num = pd.to_numeric(trends["cluster"], errors="coerce")
            for _, arow in top_alerts.iterrows():
                cid = int(arow["cluster"])
                m = str(arow["month_key"])
                prev_m = (pd.Period(m, freq="M") - 1).strftime("%Y-%m")

                curr_count_series = trends[(cluster_num == cid) & (trends["month_key"] == m)]["count"]
                prev_count_series = trends[(cluster_num == cid) & (trends["month_key"] == prev_m)]["count"]
                curr_count = int(curr_count_series.iloc[0]) if not curr_count_series.empty else int(arow.get("incidents", 0))
                prev_c = int(prev_count_series.iloc[0]) if not prev_count_series.empty else 0
                curr_rate = float(rate_map.get((cid, m), {}).get("normalized_rate", 0.0))
                prev_r = float(rate_map.get((cid, prev_m), {}).get("normalized_rate", 0.0))

                snapshot_rows.append(
                    {
                        "cluster_id": cid,
                        "cluster_label": str(arow.get("cluster_label", f"Cluster {cid}")),
                        "count": curr_count,
                        "prev_count": prev_c,
                        "count_delta": curr_count - prev_c,
                        "normalized_rate": curr_rate,
                        "prev_rate": prev_r,
                        "rate_delta": curr_rate - prev_r,
                    }
                )
            top3 = pd.DataFrame(snapshot_rows)
        elif top_spike_ids and not top_spike_rows.empty:
            label_map = (
                trends[["cluster", "cluster_label"]]
                .drop_duplicates("cluster")
                .set_index("cluster")["cluster_label"]
                .to_dict()
            )
            rate_map = trends.set_index(["cluster", "month_key"])[["normalized_rate"]].to_dict("index")
            snapshot_rows = []
            for cid in top_spike_ids:
                sp = top_spike_rows[top_spike_rows["cluster"].astype(int) == int(cid)]
                if sp.empty:
                    continue
                sp = sp.iloc[0]
                m = str(sp["month_key"])
                prev_m = (pd.Period(m, freq="M") - 1).strftime("%Y-%m")
                curr_count = int(sp["count"])
                prev_count_series = trends[(trends["cluster"] == cid) & (trends["month_key"] == prev_m)]["count"]
                prev_c = int(prev_count_series.iloc[0]) if not prev_count_series.empty else 0
                curr_rate = float(rate_map.get((cid, m), {}).get("normalized_rate", 0.0))
                prev_r = float(rate_map.get((cid, prev_m), {}).get("normalized_rate", 0.0))
                snapshot_rows.append(
                    {
                        "cluster_label": label_map.get(cid, f"Cluster {cid}"),
                        "count": curr_count,
                        "prev_count": prev_c,
                        "count_delta": curr_count - prev_c,
                        "normalized_rate": curr_rate,
                        "prev_rate": prev_r,
                        "rate_delta": curr_rate - prev_r,
                    }
                )
            top3 = pd.DataFrame(snapshot_rows)
        elif "max_rate_z" in latest_df.columns and latest_df["max_rate_z"].notna().any():
            top3 = latest_df.sort_values(["max_rate_z", "normalized_rate"], ascending=[False, False]).head(3)
        else:
            top3 = latest_df.sort_values(["normalized_rate", "count"], ascending=[False, False]).head(3)

        if "prev_count" not in top3.columns:
            prev_month = sorted(trends["month_key"].dropna().unique())[-2] if trends["month_key"].nunique() > 1 else lm
            prev_df = trends[trends["month_key"] == prev_month][["cluster_label", "count", "normalized_rate"]].rename(
                columns={"count": "prev_count", "normalized_rate": "prev_rate"}
            )
            top3 = top3.merge(prev_df, on="cluster_label", how="left")
            top3["prev_count"] = top3["prev_count"].fillna(0).astype(int)
            top3["prev_rate"] = top3["prev_rate"].fillna(0.0)
            top3["count"] = top3["count"].fillna(0).astype(int)
            top3["normalized_rate"] = top3["normalized_rate"].fillna(0.0)
            top3["count_delta"] = top3["count"] - top3["prev_count"]
            top3["rate_delta"] = top3["normalized_rate"] - top3["prev_rate"]
        display_cols = [
            "cluster_id",
            "cluster_label",
            "count",
            "prev_count",
            "count_delta",
            "normalized_rate",
            "prev_rate",
            "rate_delta",
        ]
        if "cluster_id" not in top3.columns:
            id_map = (
                trends[["cluster", "cluster_label"]]
                .drop_duplicates("cluster_label")
                .set_index("cluster_label")["cluster"]
                .to_dict()
            )
            top3["cluster_id"] = top3["cluster_label"].map(id_map).fillna(-1).astype(int)
        st.dataframe(top3[display_cols], use_container_width=True, hide_index=True)

    if "max_rate_z" in cluster_df.columns and cluster_df["max_rate_z"].notna().any():
        st.markdown("**Detected spike events (z > 1.5)**")
        spike_rows = cluster_df[cluster_df["max_rate_z"].fillna(0) > 1.5][
            ["month_key", "count", "normalized_rate", "max_rate_z"]
        ]
        if spike_rows.empty:
            st.caption("No spike months for this cluster at the current threshold.")
        else:
            st.dataframe(spike_rows.sort_values("max_rate_z", ascending=False), use_container_width=True, hide_index=True)


def _build_research_alerts(bundle: DataBundle, top_spike_rows: pd.DataFrame) -> pd.DataFrame:
    alerts = build_rule_based_alerts(bundle.monthly_cluster_trends, bundle.cluster_labels_report)
    if top_spike_rows.empty:
        return alerts
    top_spike_rows = top_spike_rows.copy()
    top_spike_rows["cluster"] = top_spike_rows["cluster"].astype(int)
    top_spike_rows = top_spike_rows[~top_spike_rows["cluster"].isin([-1, 0])].reset_index(drop=True)
    if top_spike_rows.empty:
        return alerts
    alerts["cluster"] = alerts["cluster"].astype(int)
    base_by_cluster = alerts.set_index("cluster").to_dict("index") if not alerts.empty else {}
    label_map = (
        bundle.cluster_summary[["cluster", "cluster_label"]]
        .drop_duplicates("cluster")
        .set_index("cluster")["cluster_label"]
        .to_dict()
    )
    root_map = (
        bundle.cluster_labels_report[["cluster", "causal_summary"]]
        .drop_duplicates("cluster")
        .set_index("cluster")["causal_summary"]
        .to_dict()
        if "causal_summary" in bundle.cluster_labels_report.columns
        else {}
    )
    rebuilt = []
    for _, s in top_spike_rows.iterrows():
        cid = int(s["cluster"])
        base = base_by_cluster.get(cid, {})
        rebuilt.append(
            {
                "cluster": cid,
                "cluster_label": label_map.get(cid, base.get("cluster_label", f"Cluster {cid}")),
                "month_key": s["month_key"],
                "incidents": int(s["count"]),
                "alert_id": f"SIG #{s['month_key']}-{cid:03d}",
                "priority": base.get("priority", "HIGH"),
                "confidence": base.get("confidence", 85),
                "growth_pct": base.get("growth_pct", 0.0),
                "root_cause": root_map.get(cid, base.get("root_cause", label_map.get(cid, f"Cluster {cid}"))),
                "regulation_mapping": base.get("regulation_mapping", ""),
                "relevance_score": base.get("relevance_score", 0.0),
            }
        )
    out = pd.DataFrame(rebuilt)
    order = {int(c): i for i, c in enumerate(top_spike_rows["cluster"].tolist())}
    out["__order"] = out["cluster"].map(order).fillna(999).astype(int)
    return out.sort_values("__order").drop(columns=["__order"])


def render_research(bundle: DataBundle) -> None:
    """Interactive alert console with focused detail pane."""
    st.markdown(
        "Live fusion of **ASRS cluster spikes**, **FAA corpus retrieval**, and **SMS-oriented action templates**. "
        "Select a signal card to open a focused alert panel."
    )
    st.markdown(
        """
<style>
.alert-tile {
  border: 1px solid rgba(139, 92, 246, 0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(20, 14, 30, 0.55);
  min-height: 118px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.alert-tile.active {
  border-color: rgba(167, 139, 250, 0.8);
  box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.45), 0 8px 20px rgba(88, 64, 140, 0.3);
}
.alert-tile .alert-title {
  font-size: 0.95rem;
  font-weight: 700;
  margin-top: 2px;
  line-height: 1.22;
  min-height: 2.44em; /* lock 2 lines so all cards align */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.alert-tile .alert-meta {
  font-size: 0.78rem;
  color: #9ca3af;
}
.alert-dialog {
  margin-top: 10px;
  border: 1px solid rgba(167, 139, 250, 0.5);
  border-radius: 14px;
  padding: 14px 16px;
  background: linear-gradient(145deg, rgba(23,16,33,0.88) 0%, rgba(12,18,30,0.9) 100%);
}
</style>
        """,
        unsafe_allow_html=True,
    )

    alerts = build_rule_based_alerts(bundle.monthly_cluster_trends, bundle.cluster_labels_report).head(3).copy()
    if not alerts.empty:
        z_map = (
            bundle.monthly_cluster_trends.sort_values("max_rate_z", ascending=False)
            .drop_duplicates("cluster")
            .set_index("cluster")["max_rate_z"]
            .to_dict()
            if "max_rate_z" in bundle.monthly_cluster_trends.columns
            else {}
        )
        alerts["rate_z"] = alerts["cluster"].map(z_map).fillna(0.0).astype(float)
    else:
        alerts["rate_z"] = 0.0

    with st.expander("Export bulletins (Markdown / HTML / optional PDF)", expanded=False):
        want_pdf = st.checkbox("Include PDF (requires weasyprint or xhtml2pdf)", value=False, key="rs_pdf")
        if st.button("Generate bulletins ZIP", key="rs_zip_btn"):
            with tempfile.TemporaryDirectory() as tmp:
                paths = export_bulletins(bundle, Path(tmp), regs_top_k=6, write_pdf=want_pdf)
                if not paths:
                    st.warning("Nothing was written.")
                    st.session_state.pop("bulletin_zip_bytes", None)
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for p in paths:
                            zf.write(p, arcname=p.name)
                    st.session_state["bulletin_zip_bytes"] = buf.getvalue()
        if st.session_state.get("bulletin_zip_bytes"):
            st.download_button(
                label="Download ZIP",
                data=st.session_state["bulletin_zip_bytes"],
                file_name="aviation_safety_bulletins.zip",
                mime="application/zip",
                key="rs_dl_zip",
            )

    if alerts.empty:
        st.success("No spike signals in the current window — dataset is stable against rule thresholds.")
        return

    alerts = alerts.reset_index(drop=True)

    focus_key = "alert_focus_idx"
    if focus_key not in st.session_state or int(st.session_state.get(focus_key, 0)) >= len(alerts):
        st.session_state[focus_key] = 0

    tile_cols = st.columns(len(alerts))
    for i, (_, row) in enumerate(alerts.iterrows()):
        active = i == int(st.session_state.get(focus_key, 0))
        with tile_cols[i]:
            st.markdown(
                f'<div class="alert-tile{" active" if active else ""}">'
                f'<div style="font-size:0.76rem;color:#a78bfa;font-weight:700;">SIGNAL {i+1}</div>'
                f'<div class="alert-title">{html_escape(str(row["cluster_label"]))}</div>'
                f'<div class="alert-meta" style="margin-top:4px;">Cluster {int(row["cluster"])} · {row["month_key"]}</div>'
                f'<div class="alert-meta">Growth {float(row["growth_pct"]):+.1f}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"alert_open_{i}", use_container_width=True):
                st.session_state[focus_key] = i

    row = alerts.iloc[int(st.session_state[focus_key])]
    rz = float(row.get("rate_z", 0.0))
    st.markdown('<div class="alert-dialog">', unsafe_allow_html=True)
    st.markdown(
        f"### {row['cluster_label']}  \n"
        f"`Cluster {int(row['cluster'])}` · `{row['month_key']}` · `{int(row['incidents'])} incidents`"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Growth vs baseline", f"{row['growth_pct']:+.1f}%")
    c2.metric("Spike scores", f"{int(row['confidence'])}%")
    c3.metric("Rate z", f"{rz:.2f}")

    t_summary, t_actions, t_regs = st.tabs(["Summary", "Action checklist", "Regulations"])
    with t_summary:
        synopsis = str(row.get("root_cause", "")).strip()
        if synopsis.lower() in {"", "nan", "none"}:
            synopsis = "No descriptive causal synopsis is available for this alert yet."
        st.markdown("**Causal synopsis (descriptive)**")
        synopsis_html = html_escape(synopsis[:1200])
        st.markdown(
            f"""
<div style="
  border: 1px solid rgba(148,163,184,0.35);
  background: rgba(15,23,42,0.72);
  border-radius: 10px;
  padding: 12px 14px;
  color: #e5e7eb;
  font-size: 0.95rem;
  line-height: 1.55;
  white-space: pre-wrap;
">{synopsis_html}</div>
            """,
            unsafe_allow_html=True,
        )
    with t_actions:
        tp, kw = _phase_keywords_for_cluster(bundle, int(row["cluster"]))
        recs = recommended_actions(
            str(row["cluster_label"]),
            tp,
            kw,
            str(row.get("root_cause", "")),
        )
        for i, rec in enumerate(recs, start=1):
            st.checkbox(rec, key=f"alert_action_{int(row['cluster'])}_{i}")
    with t_regs:
        with st.spinner("Loading regulations..."):
            regs = _cached_cluster_regulations(
                cluster_label=str(row["cluster_label"]),
                root_cause=str(row["root_cause"]),
                top_k=12,
            )
        uniq = []
        seen: set[str] = set()
        for reg in regs:
            key = " ".join(str(reg.get("text", "")).lower().split())[:260]
            if key and key not in seen:
                uniq.append(reg)
                seen.add(key)
            if len(uniq) >= 4:
                break
        if not uniq:
            st.caption("No hits in the local Chroma index — ingest or check FAA_RAG_MODE.")
        for reg in uniq:
            title = reg.get("citation") or "Citation"
            st.markdown(f"**{title}** · relevance {float(reg.get('relevance', 0)):.2f}")
            st.caption(str(reg.get("text", ""))[:520] + ("…" if len(str(reg.get("text", ""))) > 520 else ""))
    st.markdown("</div>", unsafe_allow_html=True)


def _uploaded_file_to_text(uploaded) -> str:
    name = str(getattr(uploaded, "name", "")).lower()
    raw = uploaded.getvalue()
    if not raw:
        return ""
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw))
            return df.astype(str).head(1500).to_csv(index=False)
        except Exception:
            pass
    if name.endswith(".json"):
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if name.endswith(".txt") or name.endswith(".md") or name.endswith(".log"):
        return raw.decode("utf-8", errors="replace")
    # Generic fallback for other text-like files.
    return raw.decode("utf-8", errors="replace")


def _chunk_text(text: str, size: int = 900, overlap: int = 180) -> list[str]:
    clean = " ".join((text or "").split())
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]
    step = max(200, size - overlap)
    return [clean[i : i + size] for i in range(0, len(clean), step)]


def _lexical_score(text: str, terms: list[str]) -> float:
    if not text or not terms:
        return 0.0
    low = text.lower()
    hits = sum(low.count(t) for t in terms)
    return float(hits) / max(4.0, float(len(terms)))


def _question_terms(question: str) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "are",
        "how",
        "this",
        "that",
        "when",
        "were",
        "was",
        "has",
        "have",
        "had",
        "does",
        "did",
        "why",
        "who",
        "will",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "been",
        "being",
        "which",
        "into",
        "about",
        "your",
        "some",
        "any",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "each",
        "other",
        "more",
        "most",
        "common",
        "involving",
    }
    return [t for t in re.findall(r"[a-z0-9]{3,}", (question or "").lower()) if t not in stop]


def _extract_cluster_id_from_question(question: str) -> Optional[int]:
    q = (question or "").lower()
    m = re.search(r"\bcluster\s*#?\s*(\d+)\b", q)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _cluster_context(bundle: DataBundle, cluster_id: int) -> dict:
    clr = bundle.cluster_labels_report
    if clr is None or clr.empty:
        return {}
    cid_col = "cluster" if "cluster" in clr.columns else ("cluster_id" if "cluster_id" in clr.columns else None)
    if not cid_col:
        return {}
    sub = clr[pd.to_numeric(clr[cid_col], errors="coerce").fillna(-1).astype(int) == int(cluster_id)]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "cluster_id": int(cluster_id),
        "cluster_label": str(row.get("cluster_label", row.get("name", ""))).strip(),
        "causal_summary": str(row.get("causal_summary", "")).strip(),
        "top_phase": str(row.get("top_phase", "")).strip(),
        "keywords": str(row.get("keywords", "")).strip(),
    }


def _bundle_evidence_snippets(
    question: str, bundle: DataBundle, max_snippets: int = 4, snippet_chars: int = 380
) -> list[tuple[float, str, str]]:
    """Return (score, snippet, source_label) from cluster labels and matching narratives."""
    terms = _question_terms(question)
    if not terms:
        return []
    scored: list[tuple[float, str, str]] = []
    clr = bundle.cluster_labels_report
    if clr is not None and not clr.empty:
        cid_col = "cluster" if "cluster" in clr.columns else ("cluster_id" if "cluster_id" in clr.columns else None)
        text_cols = [
            c for c in ("cluster_label", "description", "keywords", "causal_summary", "evidence_bullets") if c in clr.columns
        ]
        if cid_col and text_cols:
            for _, row in clr.iterrows():
                parts: list[str] = []
                for c in text_cols:
                    v = row.get(c)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        continue
                    s = str(v).strip()
                    if s:
                        parts.append(s)
                if not parts:
                    continue
                blob = " ".join(parts).lower()
                sc = float(sum(blob.count(t) for t in terms))
                if sc <= 0:
                    continue
                label = str(row.get("cluster_label", row.get("name", "")))[:120]
                cid = row.get(cid_col, "")
                body = " ".join(parts)[:1200]
                sn = (body[:snippet_chars] + "…") if len(body) > snippet_chars else body
                scored.append((sc, sn, f"Cluster {cid} · {label}".strip()))

    reports = bundle.clustered_reports
    if reports is not None and not reports.empty and "text" in reports.columns:
        col = reports["text"].astype(str)
        long_terms = sorted({t for t in terms if len(t) >= 4}, key=len, reverse=True)[:6]
        if not long_terms:
            long_terms = terms[:6]
        mask = pd.Series(False, index=reports.index)
        for kt in long_terms:
            try:
                mask = mask | col.str.lower().str.contains(re.escape(kt), case=False, na=False, regex=True)
            except re.error:
                continue
        if bool(mask.any()):
            sub = reports.loc[mask].head(500)
            for _, r in sub.iterrows():
                txt = str(r.get("text", ""))
                low = txt.lower()
                sc = float(sum(low.count(t) for t in terms))
                if sc <= 0:
                    continue
                body = txt[:1200]
                sn = (body[:snippet_chars] + "…") if len(body) > snippet_chars else body
                cl = str(r.get("cluster_label", ""))[:90]
                tail = f" · {cl}" if cl else ""
                scored.append((sc * 0.82, sn, f"ASRS narrative{tail}"))

    scored.sort(key=lambda x: -x[0])
    out: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for sc, sn, src in scored:
        key = sn[:160]
        if key in seen:
            continue
        seen.add(key)
        out.append((sc, sn, src))
        if len(out) >= max_snippets:
            break
    return out


def _rerank_reg_hits_for_question(hits: list[dict], terms: list[str]) -> list[dict]:
    """Boost regulation chunks that literally mention query terms (embedding-only ties were hiding diversity)."""
    if not hits or not terms:
        return hits
    ranked: list[tuple[float, dict]] = []
    for h in hits:
        txt = str(h.get("text", "")).lower()
        lex = float(sum(txt.count(t) for t in terms))
        base = float(h.get("relevance", 0.0))
        ranked.append((lex * 0.22 + base, h))
    ranked.sort(key=lambda x: -x[0])
    return [h for _, h in ranked]


def _build_research_chat_answer(question: str, uploaded_files: list, bundle: DataBundle | None = None) -> tuple[str, list[str]]:
    terms = _question_terms(question)
    cluster_hint = _extract_cluster_id_from_question(question)
    cluster_ctx: dict = {}
    if bundle is not None and cluster_hint is not None:
        cluster_ctx = _cluster_context(bundle, cluster_hint)

    file_chunks: list[tuple[float, str, str]] = []
    for f in uploaded_files or []:
        text = _uploaded_file_to_text(f)
        if not text:
            continue
        for ch in _chunk_text(text):
            sc = _lexical_score(ch, terms)
            if sc > 0:
                file_chunks.append((sc, ch, getattr(f, "name", "uploaded file")))
    file_chunks.sort(key=lambda x: x[0], reverse=True)
    top_file = file_chunks[:3]

    reg_query_label = question
    reg_query_cause = question
    if cluster_ctx:
        reg_query_label = cluster_ctx.get("cluster_label") or question
        reg_query_cause = cluster_ctx.get("causal_summary") or question
    kb_hits = load_cluster_regulations_rag(cluster_label=reg_query_label, root_cause=reg_query_cause, top_k=8)
    kb_hits = _rerank_reg_hits_for_question(kb_hits, terms)[:4]

    bundle_snips: list[tuple[float, str, str]] = []
    if bundle is not None:
        bundle_snips = _bundle_evidence_snippets(question, bundle)
        if cluster_ctx:
            tag = f"Cluster {cluster_ctx.get('cluster_id')}"
            cluster_specific = [x for x in bundle_snips if tag in x[2]]
            if cluster_specific:
                bundle_snips = cluster_specific + [x for x in bundle_snips if x not in cluster_specific]
            bundle_snips = bundle_snips[:4]

    body_lines: list[str] = []
    if bundle_snips:
        body_lines.append("**From your loaded dataset (cluster themes & narratives):**")
        for _, sn, src in bundle_snips:
            body_lines.append(f"- **{src}** — {sn}")
    if top_file:
        body_lines.append("**From uploaded file(s):**")
        for _, ch, src in top_file:
            body_lines.append(f"- [{src}] {ch[:260]}{'...' if len(ch) > 260 else ''}")
    if kb_hits:
        body_lines.append("**From FAA / regulations index (Chroma):**")
        for h in kb_hits[:4]:
            cite = str(h.get("citation", "FAA corpus"))
            txt = str(h.get("text", ""))
            body_lines.append(f"- [{cite}] {txt[:240]}{'...' if len(txt) > 240 else ''}")

    if not body_lines:
        return (
            "I could not find strong matches in the uploaded files or knowledge base for this question. "
            "Try a more specific question (aircraft type, phase, anomaly, cluster, month).",
            [],
        )

    # Build a direct answer first (concise), keep retrieval details out of main reply.
    answer_lines: list[str] = [f"**Your question:** {question}", ""]
    if cluster_ctx:
        answer_lines.append(
            f"**Grounded answer (cluster {cluster_ctx.get('cluster_id')} · {cluster_ctx.get('cluster_label') or 'unknown'}):**"
        )
        if cluster_ctx.get("causal_summary"):
            answer_lines.append(f"- {cluster_ctx.get('causal_summary')}")
        if cluster_ctx.get("top_phase"):
            answer_lines.append(f"- Dominant phase in current snapshot: {cluster_ctx.get('top_phase')}.")
        if cluster_ctx.get("keywords"):
            answer_lines.append(f"- Key terms: {cluster_ctx.get('keywords')}.")
    else:
        answer_lines.append("**Grounded answer:**")
        if bundle_snips:
            answer_lines.append(f"- The strongest matching dataset evidence is from: {bundle_snips[0][2]}.")
            answer_lines.append(f"- Evidence snippet: {bundle_snips[0][1][:260]}{'...' if len(bundle_snips[0][1]) > 260 else ''}")
        else:
            answer_lines.append("- I found relevant regulatory context but limited direct dataset evidence for this wording.")

    # Keep answer compact; no keyword dump or long snippet list in the main text.
    answer = "\n".join(answer_lines)
    sources = (
        [f"dataset:{src}" for _, _, src in bundle_snips]
        + [f"upload:{src}" for _, _, src in top_file]
        + [str(h.get("citation", "FAA corpus")) for h in kb_hits[:4]]
    )
    return answer, sources


def render_research_chatbot(bundle: DataBundle) -> None:
    st.markdown(
        """
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
  <div style="font-weight:700;font-size:1rem;color:#e5e7eb;">✈ Chat</div>
  <div style="font-size:0.78rem;color:#94a3b8;">Aviation Assistant</div>
</div>
<div style="font-size:0.83rem;color:#9ca3af;margin-bottom:10px;">Ask questions using uploaded files + FAA Chroma knowledge base.</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload incident/context files (txt, csv, json, md)",
        type=["txt", "csv", "json", "md", "log"],
        accept_multiple_files=True,
        key="research_uploads",
    )

    chat_key = "research_chat_history"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    q = st.chat_input("Ask about incidents, trends, causes, mitigations, or FAA rules...")
    if q:
        st.session_state[chat_key].append({"role": "user", "text": q})
        ans, sources = _build_research_chat_answer(q, uploaded_files or [], bundle)
        st.session_state[chat_key].append({"role": "assistant", "text": ans, "sources": sources})

    if not st.session_state[chat_key]:
        st.markdown(
            """
<div style="background:rgba(51,65,85,0.36);border:1px solid rgba(148,163,184,0.3);border-radius:12px;padding:10px 12px;margin-bottom:8px;color:#e5e7eb;">
Hi there, ask me about incident trends, clusters, probable causes, or FAA references.
</div>
            """,
            unsafe_allow_html=True,
        )

    for i, msg in enumerate(st.session_state[chat_key]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg["role"] == "assistant" and msg.get("sources"):
                uniq = []
                seen = set()
                for s in msg["sources"]:
                    if s not in seen:
                        uniq.append(s)
                        seen.add(s)
                st.caption("Sources used: " + " | ".join(uniq[:6]))
            if i == len(st.session_state[chat_key]) - 1 and msg["role"] == "assistant":
                pass


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br/>")
    )
