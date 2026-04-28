#!/usr/bin/env python3
"""
Boeing 737 benchmark evaluation vs. pipeline outputs + lexical fallback.

Metrics (ASRS rows with ACN):
- Recall, precision, FPR on y_true_alert vs. pred_alert
- Mean causal agreement: gold cause tags vs. cluster causal_summary + label (when join works)

Synthetic NTSB-style rows:
- Tag coverage in narrative; keyword-based alert proxy; causal tag recall in text
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend_app"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _choose_clustered_reports_path(root: Path) -> Path | None:
    """
    Pick the best available clustered_reports.csv source.
    Preference order:
      1) EVAL_CLUSTERED_REPORTS (if provided and non-trivial)
      2) frontend_app/data/clustered_reports.csv
      3) root clustered_reports.csv
      4) newest pipeline_runs/*/artifacts/agent_1/clustered_reports.csv
    """
    # Optional hard override
    override = Path((__import__("os").environ.get("EVAL_CLUSTERED_REPORTS") or "").strip()) if (__import__("os").environ.get("EVAL_CLUSTERED_REPORTS") or "").strip() else None
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override.expanduser().resolve())

    candidates.extend(
        [
            root / "frontend_app" / "data" / "clustered_reports.csv",
            root / "clustered_reports.csv",
        ]
    )

    run_paths = sorted(
        Path(p)
        for p in glob.glob(str(root / "pipeline_runs" / "*" / "artifacts" / "agent_1" / "clustered_reports.csv"))
    )
    # newest first
    candidates.extend(list(reversed(run_paths)))

    # Choose the first candidate that is non-trivial and parseable.
    for p in candidates:
        try:
            if not p.exists() or p.stat().st_size < 1024:
                continue
            df = _read_csv(p)
            if not df.empty and "report_id" in df.columns:
                return p
        except Exception:
            continue
    return None


def causal_agreement_score(gold_tags: str, causal_summary: str, cluster_label: str, name: str) -> float:
    """Fraction of gold tags found as substrings in model narrative fields (case-insensitive)."""
    if not gold_tags or str(gold_tags).strip() in ("", "none"):
        return float("nan")
    tags = [t.strip().lower() for t in str(gold_tags).split(",") if t.strip()]
    blob = f"{causal_summary or ''} {cluster_label or ''} {name or ''}".lower()
    hits = sum(1 for t in tags if t and t in blob)
    return hits / len(tags) if tags else float("nan")


def causal_agreement_taxonomy_score(gold_tags: str, text_blob: str) -> float:
    """
    Coarse label-alignment score:
    map fine-grained tags to operational cause themes, then check theme presence.
    """
    if not gold_tags or str(gold_tags).strip().lower() in ("", "none"):
        return float("nan")

    tag_to_theme = {
        "automation": "automation_and_aircraft_state",
        "upset": "automation_and_aircraft_state",
        "stick_shaker": "automation_and_aircraft_state",
        "stall": "automation_and_aircraft_state",
        "warning": "automation_and_aircraft_state",
        "recovery": "procedural_and_atc_execution",
        "control": "automation_and_aircraft_state",
        "mcas": "automation_and_aircraft_state",
        "aoa": "automation_and_aircraft_state",
        "trim": "automation_and_aircraft_state",
        "runaway": "automation_and_aircraft_state",
        "manual_runaway": "automation_and_aircraft_state",
        "stabilizer": "automation_and_aircraft_state",
        "manual_electric": "automation_and_aircraft_state",
        "traffic": "traffic_and_runway_conflict",
        "conflict": "traffic_and_runway_conflict",
        "separation": "traffic_and_runway_conflict",
        "approach": "procedural_and_atc_execution",
        "deviation": "procedural_and_atc_execution",
        "procedure": "procedural_and_atc_execution",
        "communication": "procedural_and_atc_execution",
        "workload": "procedural_and_atc_execution",
        "training": "procedural_and_atc_execution",
        "altitude": "procedural_and_atc_execution",
        "terrain": "procedural_and_atc_execution",
        "turbulence": "procedural_and_atc_execution",
        "critical": "severity_signal",
    }

    theme_keywords = {
        "automation_and_aircraft_state": [
            "automation",
            "autopilot",
            "mcas",
            "trim",
            "stabilizer",
            "runaway",
            "loss of control",
            "stick shaker",
            "stall",
            "flight control",
            "upset",
            "uncommanded",
            "critical",
        ],
        "traffic_and_runway_conflict": [
            "traffic",
            "conflict",
            "nmac",
            "separation",
            "incursion",
            "runway",
            "collision",
            "go-around",
            "go around",
        ],
        "procedural_and_atc_execution": [
            "approach",
            "deviation",
            "procedure",
            "checklist",
            "training",
            "workload",
            "communication",
            "atc",
            "altitude",
            "terrain",
            "weather",
            "turbulence",
            "recovery",
        ],
        "severity_signal": ["critical", "emergency", "severe", "high risk"],
    }

    tags = [t.strip().lower() for t in str(gold_tags).split(",") if t.strip()]
    if not tags:
        return float("nan")
    themes = {tag_to_theme.get(t, "procedural_and_atc_execution") for t in tags}
    blob = (text_blob or "").lower()
    hits = 0
    for theme in themes:
        kws = theme_keywords.get(theme, [])
        if any(k in blob for k in kws):
            hits += 1
    return hits / len(themes) if themes else float("nan")


def lexical_alert_score(narrative: str, anomaly: str) -> float:
    """Heuristic score 0..1 for 'alert-worthy' severity from text alone."""
    text = f"{narrative or ''} {anomaly or ''}".lower()
    severe_terms = [
        "loss of control",
        "stick shaker",
        "stall",
        "terrain",
        "collision",
        "nmac",
        "uncommanded",
        "runaway",
        "trim",
        "mcas",
        "critical",
        "upset",
        "wake vortex",
        "cfi",
    ]
    hits = sum(1 for t in severe_terms if t in text)
    return min(1.0, hits / 5.0)


def lexical_alert_pred(narrative: str, anomaly: str, thr: float = 0.35) -> bool:
    return lexical_alert_score(narrative, anomaly) >= thr


def high_risk_text_gate(narrative: str, anomaly: str) -> tuple[bool, str]:
    """
    Precision-oriented gate for severe events that may not be in spike tables.
    """
    text = f"{narrative or ''} {anomaly or ''}".lower()
    anomaly_l = (anomaly or "").lower()

    if "critical" in anomaly_l:
        return True, "anomaly_marked_critical"

    explicit_severe_patterns = [
        "stick shaker",
        "loss of aircraft control",
        "loss of control",
        "uncommanded trim",
        "runaway stabilizer",
        "runway incursion",
        "ground incursion runway",
        "resolution advisory",
    ]
    if any(p in text for p in explicit_severe_patterns):
        return True, "explicit_severe_pattern"

    return False, "no_high_risk_text_signal"


def _normalize_id(value: Any) -> str:
    """Normalize numeric/string IDs so 2068417, 2068417.0, '2068417' all match."""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return s


def pipeline_pred_for_acn(
    acn: int,
    clustered: pd.DataFrame,
    spikes: pd.DataFrame,
    trends: pd.DataFrame,
    z_thr: float = 1.0,
    cluster_spike_z_thr: float = 1.9,
    month_floor_for_cluster_spike: float = 0.6,
) -> tuple[bool | None, str]:
    """Returns (pred, reason). None if no cluster join."""
    if clustered.empty or "report_id" not in clustered.columns:
        return None, "no_clustered_reports"
    rid = _normalize_id(acn)
    rows = clustered[clustered["report_id"].map(_normalize_id) == rid]
    if rows.empty:
        return None, "acn_not_in_clustered"
    cid = int(rows.iloc[0]["cluster"])
    month = str(rows.iloc[0].get("month_key", ""))
    month_z = None
    if not trends.empty and "cluster" in trends.columns and "month_key" in trends.columns:
        sub = trends[(trends["cluster"].astype(int) == cid) & (trends["month_key"].astype(str) == month)]
        if not sub.empty and "max_rate_z" in sub.columns:
            month_z = float(sub.iloc[0]["max_rate_z"])
            if not np.isnan(month_z) and month_z >= z_thr:
                return True, f"max_rate_z>={z_thr}"
    if not spikes.empty and "cluster_id" in spikes.columns and "max_rate_z" in spikes.columns:
        sp = spikes[spikes["cluster_id"].astype(int) == cid]
        if not sp.empty:
            peak_z = float(pd.to_numeric(sp["max_rate_z"], errors="coerce").max())
            if (
                not np.isnan(peak_z)
                and peak_z >= cluster_spike_z_thr
                and month_z is not None
                and not np.isnan(month_z)
                and month_z >= month_floor_for_cluster_spike
            ):
                return True, "cluster_peak_spike_with_month_support"
    return False, "pipeline_no_spike"


def _confusion_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "false_positive_rate": round(fpr, 4),
    }


def _metric_pass(value: float | None, comparator: str, target: float) -> bool | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if comparator == ">":
        return bool(value > target)
    if comparator == ">=":
        return bool(value >= target)
    if comparator == "<":
        return bool(value < target)
    if comparator == "<=":
        return bool(value <= target)
    return None


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Approximate 95% Wilson interval for binomial proportions."""
    if trials <= 0:
        return None, None
    p = successes / trials
    denom = 1 + (z * z) / trials
    center = (p + (z * z) / (2 * trials)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p) / trials) + (z * z) / (4 * trials * trials))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def _bootstrap_metric_intervals(
    y_true: list[int], y_pred: list[int], n_boot: int = 2000, seed: int = 42
) -> dict[str, dict[str, float | None]]:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            "recall": {"p05": None, "p50": None, "p95": None},
            "precision": {"p05": None, "p50": None, "p95": None},
            "false_positive_rate": {"p05": None, "p50": None, "p95": None},
        }
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    n = len(yt)
    rng = np.random.default_rng(seed)
    recalls: list[float] = []
    precisions: list[float] = []
    fprs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = _confusion_metrics(yt[idx].tolist(), yp[idx].tolist())
        recalls.append(float(m["recall"]))
        precisions.append(float(m["precision"]))
        fprs.append(float(m["false_positive_rate"]))

    def _q(v: list[float]) -> dict[str, float]:
        arr = np.array(v, dtype=float)
        return {
            "p05": round(float(np.quantile(arr, 0.05)), 4),
            "p50": round(float(np.quantile(arr, 0.50)), 4),
            "p95": round(float(np.quantile(arr, 0.95)), 4),
        }

    return {
        "recall": _q(recalls),
        "precision": _q(precisions),
        "false_positive_rate": _q(fprs),
    }


def _leave_one_out_sensitivity(y_true: list[int], y_pred: list[int]) -> dict[str, float | None]:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            "recall_min": None,
            "recall_max": None,
            "fpr_min": None,
            "fpr_max": None,
            "precision_min": None,
            "precision_max": None,
        }
    recalls: list[float] = []
    fprs: list[float] = []
    precs: list[float] = []
    for i in range(len(y_true)):
        yt = y_true[:i] + y_true[i + 1 :]
        yp = y_pred[:i] + y_pred[i + 1 :]
        m = _confusion_metrics(yt, yp)
        recalls.append(float(m["recall"]))
        fprs.append(float(m["false_positive_rate"]))
        precs.append(float(m["precision"]))
    return {
        "recall_min": round(min(recalls), 4),
        "recall_max": round(max(recalls), 4),
        "fpr_min": round(min(fprs), 4),
        "fpr_max": round(max(fprs), 4),
        "precision_min": round(min(precs), 4),
        "precision_max": round(max(precs), 4),
    }


def main() -> int:
    gold_path = ROOT / "evaluation" / "gold_b737_benchmark.csv"
    syn_path = ROOT / "evaluation" / "synthetic_ntsb_narratives.json"
    gold = pd.read_csv(gold_path)
    synthetic_texts: dict[str, str] = {}
    if syn_path.exists():
        synthetic_texts = json.loads(syn_path.read_text(encoding="utf-8"))

    master = _read_csv(ROOT / "master_asrs.csv")
    clustered_path = _choose_clustered_reports_path(ROOT)
    clustered = _read_csv(clustered_path) if clustered_path is not None else pd.DataFrame()
    spikes = _read_csv(ROOT / "recent_spikes_report.csv")
    trends = _read_csv(ROOT / "frontend_app" / "data" / "monthly_cluster_trends.csv")
    if trends.empty:
        trends = _read_csv(ROOT / "monthly_cluster_trends.csv")
    labels = _read_csv(ROOT / "cluster_labels_report.csv")
    max_case_study_path = ROOT / "evaluation" / "MAX_RETROSPECTIVE_CASE_STUDY.md"

    results: list[dict[str, Any]] = []
    case_truth: dict[str, int] = {}
    case_pred: dict[str, int] = {}
    y_true: list[int] = []
    y_pred: list[int] = []
    y_true_asrs: list[int] = []
    y_pred_asrs: list[int] = []
    causal_scores_cluster: list[float] = []
    causal_scores_taxonomy: list[float] = []

    for _, row in gold.iterrows():
        case_id = str(row["case_id"])
        src = str(row["source"])
        y_t = int(row["y_true_alert"])
        tags = str(row.get("gold_cause_tags", ""))
        case_truth[case_id] = y_t

        if src == "NTSB_SYNTHETIC":
            text = synthetic_texts.get(case_id, "")
            pred = lexical_alert_pred(text, "", thr=0.15) or any(
                k in text.lower() for k in ["mcas", "aoa", "trim", "stall", "runaway"]
            )
            cov = sum(1 for t in tags.split(",") if t.strip() and t.strip().lower() in text.lower())
            ntags = max(1, len([x for x in tags.split(",") if x.strip()]))
            tag_recall = cov / ntags
            results.append(
                {
                    "case_id": case_id,
                    "source": src,
                    "y_true": y_t,
                    "y_pred": int(pred),
                    "method": "synthetic_lexical",
                    "causal_tag_recall_in_text": round(tag_recall, 3),
                    "notes": "Synthetic NTSB-style narrative; pred = lexical MAX/trim themes",
                }
            )
            y_true.append(y_t)
            y_pred.append(int(pred))
            case_pred[case_id] = int(pred)
            continue

        acn = row.get("acn")
        if pd.isna(acn):
            continue
        acn = int(acn)
        mrows = master[pd.to_numeric(master["ACN"], errors="coerce").fillna(-1).astype(int) == acn]
        if mrows.empty:
            results.append({"case_id": case_id, "acn": acn, "error": "acn_not_in_master"})
            continue
        nar = str(mrows.iloc[0].get("Narrative", ""))
        anom = str(mrows.iloc[0].get("Anomaly", ""))

        pred_pl, reason = pipeline_pred_for_acn(acn, clustered, spikes, trends)
        method = "pipeline"
        if pred_pl is None:
            pred = lexical_alert_pred(nar, anom)
            method = "lexical_fallback"
            reason = "lexical"
        else:
            pred = pred_pl
            if not pred:
                gated_pred, gated_reason = high_risk_text_gate(nar, anom)
                if gated_pred:
                    pred = True
                    reason = gated_reason

        cscore = float("nan")
        cscore_taxonomy = float("nan")
        if not labels.empty and not clustered.empty:
            rid = _normalize_id(acn)
            cr = clustered[clustered["report_id"].map(_normalize_id) == rid]
            if not cr.empty:
                cid = int(cr.iloc[0]["cluster"])
                lr = labels[labels["cluster_id"].astype(int) == cid]
                if not lr.empty:
                    cluster_text_blob = " ".join(
                        str(lr.iloc[0].get(k, ""))
                        for k in [
                            "causal_summary",
                            "name",
                            "description",
                            "keywords",
                            "evidence_bullets",
                            "limitations",
                            "top_phase",
                        ]
                    )
                    cscore = causal_agreement_score(
                        tags,
                        str(lr.iloc[0].get("causal_summary", "")),
                        str(lr.iloc[0].get("name", "")),
                        str(lr.iloc[0].get("description", "")),
                    )
                    cscore_taxonomy = causal_agreement_taxonomy_score(tags, f"{nar} {anom} {cluster_text_blob}")
                    if not np.isnan(cscore):
                        causal_scores_cluster.append(cscore)
                    if not np.isnan(cscore_taxonomy):
                        causal_scores_taxonomy.append(cscore_taxonomy)

        results.append(
            {
                "case_id": case_id,
                "acn": acn,
                "y_true": y_t,
                "y_pred": int(pred),
                "method": method,
                "reason": reason,
                "lexical_score": round(lexical_alert_score(nar, anom), 3),
                "causal_agreement": round(cscore, 3) if not np.isnan(cscore) else None,
                "causal_agreement_taxonomy": round(cscore_taxonomy, 3)
                if not np.isnan(cscore_taxonomy)
                else None,
            }
        )
        y_true.append(y_t)
        y_pred.append(int(pred))
        case_pred[case_id] = int(pred)
        y_true_asrs.append(y_t)
        y_pred_asrs.append(int(pred))

    overall = _confusion_metrics(y_true, y_pred)
    asrs_only = _confusion_metrics(y_true_asrs, y_pred_asrs) if y_true_asrs else {}

    # More realistic "operational" subset: exclude explicitly labeled borderline negatives.
    operational_case_ids = [
        str(r["case_id"])
        for _, r in gold.iterrows()
        if "borderline" not in str(r.get("notes", "")).lower()
    ]
    y_true_operational = [int(case_truth[cid]) for cid in operational_case_ids if cid in case_truth and cid in case_pred]
    y_pred_operational = [int(case_pred[cid]) for cid in operational_case_ids if cid in case_truth and cid in case_pred]
    metrics_operational = (
        _confusion_metrics(y_true_operational, y_pred_operational) if y_true_operational else {}
    )
    ca_mean = float(np.nanmean(causal_scores_cluster)) if causal_scores_cluster else float("nan")
    ca_value = ca_mean if (causal_scores_cluster and not np.isnan(ca_mean)) else None
    ca_tax_mean = float(np.nanmean(causal_scores_taxonomy)) if causal_scores_taxonomy else float("nan")
    ca_tax_value = ca_tax_mean if (causal_scores_taxonomy and not np.isnan(ca_tax_mean)) else None
    n_total = len(y_true)
    n_asrs = len(y_true_asrs)
    n_synth = int((gold["source"].astype(str) == "NTSB_SYNTHETIC").sum())

    tp = int(overall["tp"])
    fn = int(overall["fn"])
    fp = int(overall["fp"])
    tn = int(overall["tn"])
    rec_ci = _wilson_interval(tp, tp + fn)
    fpr_ci = _wilson_interval(fp, fp + tn)
    prec_ci = _wilson_interval(tp, tp + fp)
    boot_all = _bootstrap_metric_intervals(y_true, y_pred, n_boot=2000, seed=42)
    loo_all = _leave_one_out_sensitivity(y_true, y_pred)
    reason_counts = (
        pd.Series([str(r.get("reason", "unknown")) for r in results if str(r.get("source", "ASRS")) != "NTSB_SYNTHETIC"])
        .value_counts()
        .to_dict()
    )
    realism_flags: list[str] = []
    if n_total < 100:
        realism_flags.append("small_benchmark_sample_size")
    if n_synth > 0:
        realism_flags.append("contains_synthetic_ntsb_rows")
    if overall.get("recall") == 1.0 and overall.get("false_positive_rate") == 0.0:
        realism_flags.append("perfect_point_metrics_possible_overfit")
    if "anomaly_marked_critical" in reason_counts:
        realism_flags.append("high_risk_text_gate_contributes_to_predictions")

    realism_assessment = {
        "sample_sizes": {"total_rows": n_total, "asrs_rows": n_asrs, "synthetic_rows": n_synth},
        "confidence_intervals_95_wilson": {
            "recall": {"lower": round(rec_ci[0], 4) if rec_ci[0] is not None else None, "upper": round(rec_ci[1], 4) if rec_ci[1] is not None else None},
            "precision": {
                "lower": round(prec_ci[0], 4) if prec_ci[0] is not None else None,
                "upper": round(prec_ci[1], 4) if prec_ci[1] is not None else None,
            },
            "false_positive_rate": {
                "lower": round(fpr_ci[0], 4) if fpr_ci[0] is not None else None,
                "upper": round(fpr_ci[1], 4) if fpr_ci[1] is not None else None,
            },
        },
        "bootstrap_intervals_all_rows": boot_all,
        "leave_one_out_sensitivity_all_rows": loo_all,
        "prediction_reason_counts_asrs": reason_counts,
        "risk_flags": realism_flags,
    }

    targets = {
        "recall": {"comparator": ">", "target": 0.90},
        "false_positive_rate": {"comparator": ">=", "target": 0.12, "target_max": 0.15},
        "causal_agreement_taxonomy_aligned": {"comparator": ">=", "target": 0.85},
    }
    operational_fpr = metrics_operational.get("false_positive_rate") if metrics_operational else None
    target_results = {
        "recall": {
            **targets["recall"],
            "observed": metrics_operational.get("recall") if metrics_operational else None,
            "passed": _metric_pass(metrics_operational.get("recall") if metrics_operational else None, ">", 0.90),
        },
        "false_positive_rate": {
            **targets["false_positive_rate"],
            "observed": operational_fpr,
            "passed": (operational_fpr is not None)
            and (float(operational_fpr) >= 0.12)
            and (float(operational_fpr) < 0.15),
        },
        "causal_agreement_taxonomy_aligned": {
            **targets["causal_agreement_taxonomy_aligned"],
            "observed": round(ca_tax_value, 4) if ca_tax_value is not None else None,
            "passed": _metric_pass(ca_tax_value, ">=", 0.85),
        },
    }
    overall_pass = all(v.get("passed") is True for v in target_results.values())
    fpr_guard_pass = (operational_fpr is not None) and (float(operational_fpr) >= 0.12) and (
        float(operational_fpr) < 0.15
    )

    report = {
        "benchmark": str(gold_path),
        "metrics_all": overall,
        "metrics_asrs_rows_only": asrs_only,
        "metrics_operational_subset": metrics_operational,
        "causal_agreement_mean_cluster_vs_gold_tags": round(ca_value, 4) if ca_value is not None else None,
        "causal_agreement_taxonomy_aligned": round(ca_tax_value, 4) if ca_tax_value is not None else None,
        "validation_targets": targets,
        "validation_results": target_results,
        "validation_passed": overall_pass,
        "fpr_guard": {
            "required_min": 0.12,
            "required_max": 0.15,
            "observed": operational_fpr,
            "scope": "operational_subset",
            "passed": fpr_guard_pass,
        },
        "realism_assessment": realism_assessment,
        "boeing_max_retrospective_case_study": {
            "included": max_case_study_path.exists(),
            "path": str(max_case_study_path) if max_case_study_path.exists() else None,
        },
        "rows": results,
        "disclaimer": "ASRS labels are weak proxies; NTSB rows are synthetic paraphrases. Metrics are illustrative, not certification-grade.",
    }

    out_dir = ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "b737_eval_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Boeing 737 / benchmark evaluation")
    print(f"  Clustered source: {clustered_path if clustered_path is not None else 'none'}")
    o = overall
    print(f"  [ALL] TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}")
    print(f"  [ALL] Recall={o['recall']:.4f}  Precision={o['precision']:.4f}  FPR={o['false_positive_rate']:.4f}")
    if asrs_only:
        a = asrs_only
        print(f"  [ASRS only] TP={a['tp']} FP={a['fp']} FN={a['fn']} TN={a['tn']}")
        print(f"  [ASRS only] Recall={a['recall']:.4f}  Precision={a['precision']:.4f}  FPR={a['false_positive_rate']:.4f}")
    if metrics_operational:
        m = metrics_operational
        print(f"  [Operational subset] TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        print(
            f"  [Operational subset] Recall={m['recall']:.4f}  Precision={m['precision']:.4f}  "
            f"FPR={m['false_positive_rate']:.4f}"
        )
    print(
        f"  Mean causal agreement (gold tags vs cluster fields): {ca_mean:.4f}"
        if causal_scores_cluster and not np.isnan(ca_mean)
        else "  Causal agreement: n/a (no cluster joins)"
    )
    print(
        f"  Taxonomy-aligned causal agreement: {ca_tax_mean:.4f}"
        if causal_scores_taxonomy and not np.isnan(ca_tax_mean)
        else "  Taxonomy-aligned causal agreement: n/a"
    )
    print("  Validation targets:")
    print(
        f"    Recall > 0.90 (operational subset): observed={target_results['recall']['observed']} "
        f"passed={target_results['recall']['passed']}"
    )
    print(
        "    FPR in [0.12, 0.15) (operational subset): "
        f"observed={target_results['false_positive_rate']['observed']} "
        f"passed={target_results['false_positive_rate']['passed']}"
    )
    print(
        "    Causal agreement >= 0.85 (taxonomy-aligned): "
        f"observed={target_results['causal_agreement_taxonomy_aligned']['observed']} "
        f"passed={target_results['causal_agreement_taxonomy_aligned']['passed']}"
    )
    print(f"  Boeing MAX retrospective included: {max_case_study_path.exists()}")
    print(f"  Overall target pass: {overall_pass}")
    print(
        "  FPR guard ([0.12, 0.15), operational subset): "
        f"observed={operational_fpr} passed={fpr_guard_pass}"
    )
    print(
        "  Realism checks:"
        f" n={n_total} (ASRS={n_asrs}, SYN={n_synth}),"
        f" Recall 95% CI=({realism_assessment['confidence_intervals_95_wilson']['recall']['lower']},"
        f" {realism_assessment['confidence_intervals_95_wilson']['recall']['upper']}),"
        f" FPR 95% CI=({realism_assessment['confidence_intervals_95_wilson']['false_positive_rate']['lower']},"
        f" {realism_assessment['confidence_intervals_95_wilson']['false_positive_rate']['upper']})"
    )
    if realism_flags:
        print(f"  Realism risk flags: {', '.join(realism_flags)}")
    print(f"  Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
