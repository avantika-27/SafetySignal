#!/usr/bin/env python3
"""
NTSB-based validation for Claim 3:
"The system correctly distinguishes alert-worthy safety themes from routine ones."

Three tests:
  Test 1 — Thematic alignment: Do ASRS cluster themes match NTSB accident cause distribution?
  Test 2 — Temporal co-occurrence: When ASRS clusters spike, do NTSB accidents in the same
            theme also increase in the same/following month?
  Test 3 — Alert precision: For each month a theme is "spiking" in ASRS, was that theme
            present in NTSB fatal/serious accidents that same month?

Design rationale:
  ASRS and NTSB cannot be joined by report ID — they are separate systems.
  ASRS = voluntary self-reports of near-misses and incidents (no injury required).
  NTSB = official accident investigations (damage or injury required).
  Validation is therefore done at the THEMATIC + TEMPORAL level, not at report level.
  This is appropriate and acknowledged as a limitation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NTSB_PATH = Path(r"C:\Users\Shreya Sett\Downloads\04762d92-2675-4c47-8940-b9df3ad1644bAviationData.csv")

# ---------------------------------------------------------------------------
# Theme taxonomy  (6 categories shared between NTSB causes and ASRS clusters)
# ---------------------------------------------------------------------------

# Keywords used to classify NTSB ProbableCause text into themes
NTSB_THEME_KEYWORDS: dict[str, list[str]] = {
    "airspace_conflict":    ["separation", "nmac", "near midair", "midair", "collision", "airspace", "conflict", "traffic"],
    "ground_ops":           ["taxi", "runway incursion", "ground collision", "ground event", "ground operation", "ramp"],
    "atc_procedure":        ["atc", "controller", "clearance", "procedure", "checklist", "far violation", "regulatory", "deviation"],
    "equipment_failure":    ["mechanical", "engine failure", "engine power", "hydraulic", "electrical", "avionics",
                             "structural failure", "gear", "malfunction", "component"],
    "weather_environment":  ["weather", "wind", "turbulence", "icing", "wake vortex", "thunderstorm", "fog", "vmcg",
                             "spatial disorientation", "terrain"],
    "loss_of_control":      ["loss of control", "stall", "spin", "upset", "controllability", "loss of aircraft control",
                             "cfit", "controlled flight"],
}

# ASRS cluster IDs mapped to themes (based on cluster name inspection)
ASRS_CLUSTER_THEMES: dict[str, list[int]] = {
    "airspace_conflict":    [2, 3, 4, 18, 23],
    "ground_ops":           [0, 8, 15, 36],
    "atc_procedure":        [1, 14, 26, 27, 28, 29, 32, 37, 38, 39],
    "equipment_failure":    [5, 6, 7, 9, 10, 11, 12, 13, 16, 19, 20, 21, 22, 24, 25, 30, 31, 33, 34, 35, 41, 42, 43],
    "weather_environment":  [17],
    "loss_of_control":      [40],
}

THEMES = list(NTSB_THEME_KEYWORDS.keys())


def classify_ntsb_theme(cause_text: str) -> str:
    """Return the first matching theme, or 'other'."""
    text = (cause_text or "").lower()
    for theme, keywords in NTSB_THEME_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return theme
    return "other"


def is_alert_worthy_ntsb(row: pd.Series) -> int:
    """1 if fatal/serious injury or aircraft destroyed, else 0."""
    fatal = pd.to_numeric(row.get("FatalInjuryCount", 0), errors="coerce") or 0
    serious = pd.to_numeric(row.get("SeriousInjuryCount", 0), errors="coerce") or 0
    damage = str(row.get("AirCraftDamage", "")).lower()
    return int(fatal > 0 or serious > 0 or "destroyed" in damage)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    ntsb = pd.read_csv(NTSB_PATH, encoding="latin-1", low_memory=False)
    ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"], errors="coerce", utc=True)
    ntsb["month_key"] = ntsb["EventDate"].dt.to_period("M").astype(str)
    ntsb["theme"] = ntsb["ProbableCause"].fillna("").apply(classify_ntsb_theme)
    ntsb["y_true"] = ntsb.apply(is_alert_worthy_ntsb, axis=1)

    clustered = pd.read_csv(ROOT / "frontend_app" / "data" / "clustered_reports.csv")
    clustered["month_key"] = pd.to_datetime(clustered["date"], errors="coerce").dt.to_period("M").astype(str)

    # Map ASRS clusters to themes
    cluster_to_theme: dict[int, str] = {}
    for theme, ids in ASRS_CLUSTER_THEMES.items():
        for cid in ids:
            cluster_to_theme[cid] = theme
    clustered["theme"] = clustered["cluster"].map(cluster_to_theme).fillna("other")

    trends = pd.read_csv(ROOT / "frontend_app" / "data" / "monthly_cluster_trends.csv")
    labels = pd.read_csv(ROOT / "cluster_labels_report.csv")

    # Overlap months
    asrs_months = set(clustered["month_key"].dropna())
    ntsb_months = set(ntsb["month_key"].dropna())
    overlap = sorted(asrs_months & ntsb_months)

    return ntsb, clustered, trends, labels, overlap


# ---------------------------------------------------------------------------
# Test 1 — Thematic alignment
# ---------------------------------------------------------------------------

def test1_thematic_alignment(ntsb, clustered, overlap):
    """
    Compare % distribution of themes in NTSB accidents vs ASRS alert-worthy clusters
    during the 23-month overlap window.
    """
    ntsb_ol = ntsb[ntsb["month_key"].isin(overlap)]
    asrs_ol = clustered[clustered["month_key"].isin(overlap) & (clustered["cluster"] != -1)]

    ntsb_alert = ntsb_ol[ntsb_ol["y_true"] == 1]

    ntsb_dist = ntsb_alert["theme"].value_counts(normalize=True).reindex(THEMES, fill_value=0)
    asrs_dist = asrs_ol["theme"].value_counts(normalize=True).reindex(THEMES, fill_value=0)

    rows = []
    for t in THEMES:
        rows.append({
            "theme": t,
            "ntsb_alert_pct": round(ntsb_dist[t] * 100, 1),
            "asrs_pct": round(asrs_dist[t] * 100, 1),
            "abs_diff": round(abs(ntsb_dist[t] - asrs_dist[t]) * 100, 1),
        })
    df_align = pd.DataFrame(rows)

    # Chi-squared test: are distributions similar?
    ntsb_counts = ntsb_alert["theme"].value_counts().reindex(THEMES, fill_value=0).values.astype(float)
    asrs_counts = asrs_ol["theme"].value_counts().reindex(THEMES, fill_value=0).values.astype(float)
    # Normalize ASRS counts to same total as NTSB for chi-sq
    asrs_expected = asrs_counts / asrs_counts.sum() * ntsb_counts.sum()
    chi2, pval = stats.chisquare(f_obs=ntsb_counts + 1, f_exp=asrs_expected + 1)

    return {
        "test": "thematic_alignment",
        "description": "Distribution of themes in NTSB alert-worthy accidents vs ASRS clustered reports (overlap 2024-01 to 2025-11)",
        "n_ntsb_alert_events": int(len(ntsb_alert)),
        "n_asrs_clustered_reports": int(len(asrs_ol)),
        "n_overlap_months": len(overlap),
        "theme_distribution": rows,
        "chi2_stat": round(float(chi2), 3),
        "chi2_pvalue": round(float(pval), 4),
        "interpretation": (
            "p >= 0.05: thematic distributions are not significantly different (alignment supported)"
            if pval >= 0.05 else
            "p < 0.05: distributions differ significantly — ASRS and NTSB weight themes differently"
        ),
    }


# ---------------------------------------------------------------------------
# Test 2 — Temporal co-occurrence
# ---------------------------------------------------------------------------

def test2_temporal_cooccurrence(ntsb, clustered, trends, overlap):
    """
    For each (theme, month) in overlap window:
      - Compute ASRS spike flag: z-score of cluster rate in that theme >= 1.5
      - Compute NTSB alert flag: NTSB fatal/serious accidents in that theme >= monthly median + 1 std
    Report: correlation and precision/recall of ASRS spikes vs NTSB accident elevations.
    """
    # Build monthly ASRS theme counts from trends
    # trends has cluster-level monthly data; map clusters to themes
    cluster_to_theme: dict[int, str] = {}
    for theme, ids in ASRS_CLUSTER_THEMES.items():
        for cid in ids:
            cluster_to_theme[cid] = theme

    trends["theme"] = trends["cluster"].map(cluster_to_theme).fillna("other")
    trends_ol = trends[trends["month_key"].isin(overlap)]

    # Aggregate z-score by theme-month: use max z-score across clusters in theme
    asrs_theme_monthly = (
        trends_ol[trends_ol["theme"] != "other"]
        .groupby(["month_key", "theme"])["max_rate_z"]
        .max()
        .reset_index()
        .rename(columns={"max_rate_z": "theme_max_z"})
    )
    asrs_theme_monthly["asrs_spike"] = (asrs_theme_monthly["theme_max_z"] >= 1.5).astype(int)

    # Build monthly NTSB theme counts
    ntsb_ol = ntsb[(ntsb["month_key"].isin(overlap)) & (ntsb["y_true"] == 1)]
    ntsb_theme_monthly = (
        ntsb_ol.groupby(["month_key", "theme"])
        .size()
        .reset_index(name="ntsb_alert_count")
    )

    # For each theme, compute "elevated" flag: count >= median + 0.5*std
    ntsb_elevated_rows = []
    for theme in THEMES:
        sub = ntsb_theme_monthly[ntsb_theme_monthly["theme"] == theme]
        if sub.empty:
            continue
        median_count = sub["ntsb_alert_count"].median()
        std_count = sub["ntsb_alert_count"].std()
        threshold = median_count + 0.5 * (std_count if not np.isnan(std_count) else 0)
        for _, r in sub.iterrows():
            ntsb_elevated_rows.append({
                "month_key": r["month_key"],
                "theme": r["theme"],
                "ntsb_alert_count": r["ntsb_alert_count"],
                "ntsb_elevated": int(r["ntsb_alert_count"] >= threshold),
                "threshold": round(threshold, 1),
            })
    ntsb_elevated = pd.DataFrame(ntsb_elevated_rows)

    # Join ASRS spikes with NTSB elevated flags
    merged = asrs_theme_monthly.merge(ntsb_elevated, on=["month_key", "theme"], how="inner")

    # Correlation per theme
    theme_results = []
    for theme in THEMES:
        sub = merged[merged["theme"] == theme]
        if len(sub) < 5:
            theme_results.append({
                "theme": theme,
                "n_months": len(sub),
                "note": "insufficient data (<5 months overlap)",
            })
            continue
        corr, pval = stats.spearmanr(sub["theme_max_z"], sub["ntsb_alert_count"])
        tp = int(((sub["asrs_spike"] == 1) & (sub["ntsb_elevated"] == 1)).sum())
        fp = int(((sub["asrs_spike"] == 1) & (sub["ntsb_elevated"] == 0)).sum())
        fn = int(((sub["asrs_spike"] == 0) & (sub["ntsb_elevated"] == 1)).sum())
        tn = int(((sub["asrs_spike"] == 0) & (sub["ntsb_elevated"] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        theme_results.append({
            "theme": theme,
            "n_months": len(sub),
            "spearman_r": round(float(corr), 3) if not np.isnan(corr) else None,
            "spearman_p": round(float(pval), 4) if not np.isnan(pval) else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_asrs_spike_vs_ntsb_elevated": round(precision, 3) if precision is not None else None,
            "recall_asrs_spike_vs_ntsb_elevated": round(recall, 3) if recall is not None else None,
        })

    # Overall across all themes
    if not merged.empty:
        overall_corr, overall_p = stats.spearmanr(merged["theme_max_z"], merged["ntsb_alert_count"])
        overall_tp = int(((merged["asrs_spike"] == 1) & (merged["ntsb_elevated"] == 1)).sum())
        overall_fp = int(((merged["asrs_spike"] == 1) & (merged["ntsb_elevated"] == 0)).sum())
        overall_fn = int(((merged["asrs_spike"] == 0) & (merged["ntsb_elevated"] == 1)).sum())
        overall_tn = int(((merged["asrs_spike"] == 0) & (merged["ntsb_elevated"] == 0)).sum())
        overall_prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else None
        overall_rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else None
    else:
        overall_corr = overall_p = overall_prec = overall_rec = None
        overall_tp = overall_fp = overall_fn = overall_tn = 0

    return {
        "test": "temporal_cooccurrence",
        "description": (
            "Per-theme Spearman correlation between ASRS cluster max z-score and NTSB alert-worthy accident count, "
            "monthly over 23-month overlap (2024-01 to 2025-11). "
            "ASRS spike threshold: z >= 1.5. NTSB elevated: count >= median + 0.5 * std."
        ),
        "overall": {
            "spearman_r": round(float(overall_corr), 3) if overall_corr is not None and not np.isnan(overall_corr) else None,
            "spearman_p": round(float(overall_p), 4) if overall_p is not None and not np.isnan(overall_p) else None,
            "tp": overall_tp, "fp": overall_fp, "fn": overall_fn, "tn": overall_tn,
            "precision": round(overall_prec, 3) if overall_prec is not None else None,
            "recall": round(overall_rec, 3) if overall_rec is not None else None,
        },
        "by_theme": theme_results,
        "caveat": (
            "ASRS reports voluntary incidents (no injury required); NTSB records accidents (injury/damage required). "
            "Correlation measures thematic co-occurrence, not causal prediction. "
            "23-month window limits statistical power."
        ),
    }


# ---------------------------------------------------------------------------
# Test 3 — Alert precision on NTSB accident months
# ---------------------------------------------------------------------------

def test3_alert_precision(ntsb, trends, overlap):
    """
    For each month with >= 1 NTSB fatal/serious accident in a theme:
    Was the ASRS system spiking in that same theme in that month?
    Measures: of months where bad accidents happened, did SafetySignal alert on that theme?
    """
    cluster_to_theme: dict[int, str] = {}
    for theme, ids in ASRS_CLUSTER_THEMES.items():
        for cid in ids:
            cluster_to_theme[cid] = theme
    trends["theme"] = trends["cluster"].map(cluster_to_theme).fillna("other")

    # ASRS spiking theme-months
    trends_ol = trends[trends["month_key"].isin(overlap)]
    asrs_spike_set = set(
        trends_ol[
            (trends_ol["max_rate_z"] >= 1.5) & (trends_ol["theme"] != "other")
        ].apply(lambda r: (r["month_key"], r["theme"]), axis=1)
    )

    # NTSB accident theme-months (alert-worthy only)
    ntsb_ol = ntsb[(ntsb["month_key"].isin(overlap)) & (ntsb["y_true"] == 1) & (ntsb["theme"] != "other")]
    ntsb_accident_months = set(
        ntsb_ol.apply(lambda r: (r["month_key"], r["theme"]), axis=1)
    )

    hits = ntsb_accident_months & asrs_spike_set
    misses = ntsb_accident_months - asrs_spike_set

    total_accident_theme_months = len(ntsb_accident_months)
    hit_count = len(hits)
    coverage = hit_count / total_accident_theme_months if total_accident_theme_months > 0 else 0.0

    # Detail by theme
    theme_detail = []
    for theme in THEMES:
        ntsb_months_t = {m for (m, t) in ntsb_accident_months if t == theme}
        asrs_months_t = {m for (m, t) in asrs_spike_set if t == theme}
        hit_t = ntsb_months_t & asrs_months_t
        theme_detail.append({
            "theme": theme,
            "ntsb_accident_months": len(ntsb_months_t),
            "asrs_spiking_months": len(asrs_months_t),
            "co_occurring": len(hit_t),
            "coverage_pct": round(len(hit_t) / len(ntsb_months_t) * 100, 1) if ntsb_months_t else None,
        })

    return {
        "test": "alert_precision_on_ntsb_accident_months",
        "description": (
            "Of all (theme, month) pairs where NTSB recorded a fatal/serious accident, "
            "what fraction also had a SafetySignal ASRS spike (z >= 1.5) in the same theme?"
        ),
        "total_ntsb_accident_theme_months": total_accident_theme_months,
        "asrs_spike_theme_months": len(asrs_spike_set),
        "co_occurring": hit_count,
        "coverage": round(coverage, 4),
        "coverage_pct": round(coverage * 100, 1),
        "by_theme": theme_detail,
        "interpretation": (
            f"SafetySignal was spiking in the same theme as {coverage*100:.1f}% of NTSB fatal/serious accident months. "
            "This measures whether ASRS trend signals co-occur with accident reports — not whether they predict them."
        ),
        "caveat": (
            "A theme-month 'miss' means ASRS had no spike in that theme that month — not that the system failed. "
            "ASRS and NTSB cover different aircraft populations and event severities."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading data...")
    ntsb, clustered, trends, labels, overlap = load_data()
    print(f"  NTSB rows: {len(ntsb)}  |  ASRS rows: {len(clustered)}  |  Overlap months: {len(overlap)}")
    print(f"  NTSB date range: {min(overlap)} to {max(overlap)}")
    print()

    print("Test 1: Thematic alignment...")
    t1 = test1_thematic_alignment(ntsb, clustered, overlap)
    print(f"  chi2={t1['chi2_stat']}  p={t1['chi2_pvalue']}")
    print(f"  {t1['interpretation']}")
    print()

    print("Test 2: Temporal co-occurrence...")
    t2 = test2_temporal_cooccurrence(ntsb, clustered, trends, overlap)
    o = t2["overall"]
    print(f"  Overall Spearman r={o['spearman_r']}  p={o['spearman_p']}")
    print(f"  TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}")
    print(f"  Precision={o['precision']}  Recall={o['recall']}")
    print()

    print("Test 3: Alert precision on NTSB accident months...")
    t3 = test3_alert_precision(ntsb, trends, overlap)
    print(f"  NTSB accident theme-months: {t3['total_ntsb_accident_theme_months']}")
    print(f"  SafetySignal co-occurring spikes: {t3['co_occurring']} ({t3['coverage_pct']}%)")
    print(f"  {t3['interpretation']}")
    print()

    # Theme-level detail table
    print("Theme detail (Test 3):")
    print(f"  {'Theme':<25} NTSB_months  ASRS_spike_months  Co-occur  Coverage%")
    for row in t3["by_theme"]:
        cov = str(row['coverage_pct'])+'%' if row['coverage_pct'] is not None else 'N/A'
        print(f"  {row['theme']:<25} {row['ntsb_accident_months']:<13} {row['asrs_spiking_months']:<19} {row['co_occurring']:<10} {cov}")

    report = {
        "ntsb_file": str(NTSB_PATH),
        "asrs_clustered_file": str(ROOT / "frontend_app" / "data" / "clustered_reports.csv"),
        "overlap_months": overlap,
        "test1_thematic_alignment": t1,
        "test2_temporal_cooccurrence": t2,
        "test3_alert_precision": t3,
        "overall_disclaimer": (
            "ASRS and NTSB cannot be joined by report ID. Validation is thematic and temporal. "
            "Results show structural consistency between the two systems — not causal prediction. "
            "ASRS covers voluntary near-miss reports; NTSB covers investigated accidents. "
            "Overlap window: 23 months (2024-01 to 2025-11)."
        ),
    }

    out_dir = ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ntsb_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    print(f"\nFull report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
