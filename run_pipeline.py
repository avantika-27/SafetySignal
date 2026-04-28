#!/usr/bin/env python3
"""Orchestrate Agents 1-5 and persist run artifacts/manifest.

Usage:
  python run_pipeline.py
  python run_pipeline.py --config pipeline_config.json
  python run_pipeline.py --output-dir runs --stop-on-error

Config file format (JSON):
{
  "agents": [
    {
      "name": "Agent 1",
      "command": ["python", "agent1.py"],
      "artifacts": ["outputs/agent1_result.json"]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AGENTS = [
    {
        "name": "Agent 1",
        "command": ["python", "agent1.py"],
        "artifacts": ["frontend_app/data/clustered_reports.csv"],
    },
    {
        "name": "Agent 2",
        "command": ["python", "agent2.py"],
        "artifacts": ["cluster_labels_report.csv", "frontend_app/data/cluster_summary.csv"],
    },
    {
        "name": "Agent 3",
        "command": ["python", "agent3.py"],
        "artifacts": ["recent_spikes_report.csv", "frontend_app/data/monthly_cluster_trends.csv"],
    },
    {
        "name": "Agent 4",
        "command": ["python", "agent4.py"],
        "artifacts": ["frontend_app/data/alerts.csv"],
    },
    {
        "name": "Agent 5",
        "command": ["python", "agent5.py"],
        "artifacts": ["pipeline_validation_report.json"],
    },
]


@dataclass
class AgentSpec:
    name: str
    command: list[str]
    artifacts: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_config(config_path: Path | None) -> list[AgentSpec]:
    if config_path is None:
        raw_agents = DEFAULT_AGENTS
    else:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw_agents = data.get("agents", [])

    if len(raw_agents) != 5:
        raise ValueError(
            f"Expected exactly 5 agents (Agent 1-5), found {len(raw_agents)}."
        )

    specs: list[AgentSpec] = []
    for idx, raw in enumerate(raw_agents, start=1):
        name = str(raw.get("name") or f"Agent {idx}")
        command = raw.get("command")
        artifacts = raw.get("artifacts", [])
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError(f"{name}: command must be a list of strings.")
        if not isinstance(artifacts, list) or not all(
            isinstance(item, str) for item in artifacts
        ):
            raise ValueError(f"{name}: artifacts must be a list of strings.")
        specs.append(AgentSpec(name=name, command=command, artifacts=artifacts))
    return specs


def _run_agent(
    agent: AgentSpec, project_root: Path, run_dir: Path
) -> tuple[dict[str, Any], bool]:
    agent_slug = agent.name.lower().replace(" ", "_")
    stdout_path = run_dir / f"{agent_slug}.stdout.log"
    stderr_path = run_dir / f"{agent_slug}.stderr.log"

    started = _utc_now_iso()
    start_monotonic = time.monotonic()

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(
            agent.command,
            cwd=project_root,
            stdout=out,
            stderr=err,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    duration = round(time.monotonic() - start_monotonic, 3)
    finished = _utc_now_iso()
    ok = proc.returncode == 0

    copied_artifacts: list[str] = []
    missing_artifacts: list[str] = []
    artifacts_dir = run_dir / "artifacts" / agent_slug
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for artifact in agent.artifacts:
        src = (project_root / artifact).resolve()
        if src.exists() and src.is_file():
            dest = artifacts_dir / src.name
            dest.write_bytes(src.read_bytes())
            copied_artifacts.append(str(dest.relative_to(run_dir)))
        else:
            missing_artifacts.append(artifact)

    result = {
        "name": agent.name,
        "command": agent.command,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_seconds": duration,
        "status": "success" if ok else "failed",
        "return_code": proc.returncode,
        "stdout_log": str(stdout_path.relative_to(run_dir)),
        "stderr_log": str(stderr_path.relative_to(run_dir)),
        "artifacts_declared": agent.artifacts,
        "artifacts_copied": copied_artifacts,
        "artifacts_missing": missing_artifacts,
    }
    return result, ok


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _is_finite_float(value: str) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _check_required_columns(path: Path, required: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"missing file: {path.name}"]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
    missing = [c for c in required if c not in cols]
    return len(missing) == 0, missing


def _validation_result(gate: str, checks: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    passed = all(bool(c.get("passed")) for c in checks)
    return {"gate": gate, "passed": passed, "checks": checks, "metrics": metrics}


def _validate_agent_1(project_root: Path) -> dict[str, Any]:
    gate = "data_quality"
    target = project_root / "frontend_app" / "data" / "clustered_reports.csv"
    required = ["report_id", "date", "month_key", "text", "cluster", "cluster_label", "umap_x", "umap_y"]
    ok_cols, missing_cols = _check_required_columns(target, required)

    checks: list[dict[str, Any]] = [
        {"name": "clustered_reports_exists_and_columns", "passed": ok_cols, "details": missing_cols}
    ]
    metrics: dict[str, Any] = {"file": str(target), "row_count": 0, "null_report_id_rate": 1.0, "null_text_rate": 1.0}
    if not ok_cols:
        return _validation_result(gate, checks, metrics)

    rows = _read_csv_rows(target)
    row_count = len(rows)
    null_report_id = sum(1 for r in rows if not str(r.get("report_id", "")).strip())
    null_text = sum(1 for r in rows if not str(r.get("text", "")).strip())
    clusters = {r.get("cluster", "") for r in rows if str(r.get("cluster", "")).strip()}
    checks.append({"name": "non_empty_dataset", "passed": row_count > 0, "details": {"row_count": row_count}})
    checks.append(
        {
            "name": "reasonable_cluster_coverage",
            "passed": len(clusters) >= 2,
            "details": {"unique_clusters": len(clusters)},
        }
    )
    checks.append(
        {
            "name": "null_rate_threshold",
            "passed": (null_report_id / row_count if row_count else 1.0) <= 0.02
            and (null_text / row_count if row_count else 1.0) <= 0.05,
            "details": {"max_report_id_null_rate": 0.02, "max_text_null_rate": 0.05},
        }
    )
    metrics.update(
        {
            "row_count": row_count,
            "unique_clusters": len(clusters),
            "null_report_id_rate": round(null_report_id / row_count, 4) if row_count else 1.0,
            "null_text_rate": round(null_text / row_count, 4) if row_count else 1.0,
        }
    )
    return _validation_result(gate, checks, metrics)


def _validate_agent_2(project_root: Path) -> dict[str, Any]:
    gate = "embedding_and_clustering_sanity"
    reports = project_root / "frontend_app" / "data" / "clustered_reports.csv"
    labels = project_root / "cluster_labels_report.csv"
    summary = project_root / "frontend_app" / "data" / "cluster_summary.csv"

    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    ok_reports, missing_reports = _check_required_columns(
        reports, ["cluster", "umap_x", "umap_y", "cluster_label", "report_id"]
    )
    ok_labels, missing_labels = _check_required_columns(labels, ["cluster_id", "size", "name", "description"])
    ok_summary, missing_summary = _check_required_columns(
        summary, ["cluster", "cluster_label", "cluster_size", "description", "keywords", "is_noise"]
    )
    checks.extend(
        [
            {"name": "reports_schema", "passed": ok_reports, "details": missing_reports},
            {"name": "labels_schema", "passed": ok_labels, "details": missing_labels},
            {"name": "summary_schema", "passed": ok_summary, "details": missing_summary},
        ]
    )
    if not (ok_reports and ok_labels and ok_summary):
        return _validation_result(gate, checks, metrics)

    report_rows = _read_csv_rows(reports)
    label_rows = _read_csv_rows(labels)
    summary_rows = _read_csv_rows(summary)

    finite_embed = sum(
        1
        for r in report_rows
        if _is_finite_float(r.get("umap_x", "")) and _is_finite_float(r.get("umap_y", ""))
    )
    finite_ratio = finite_embed / len(report_rows) if report_rows else 0.0
    label_ids = {_to_int(r.get("cluster_id", "")) for r in label_rows}
    label_ids.discard(None)
    report_ids = {_to_int(r.get("cluster", "")) for r in report_rows}
    report_ids.discard(None)
    covered = len(report_ids.intersection(label_ids))
    coverage = covered / len(report_ids) if report_ids else 0.0

    checks.append(
        {"name": "embedding_finite_ratio", "passed": finite_ratio >= 0.99, "details": {"min_ratio": 0.99}}
    )
    checks.append(
        {
            "name": "label_coverage_of_clusters",
            "passed": coverage >= 0.95,
            "details": {"min_ratio": 0.95},
        }
    )
    checks.append(
        {
            "name": "summary_cluster_count_sane",
            "passed": len(summary_rows) >= max(1, len(label_rows) - 1),
            "details": {"summary_rows": len(summary_rows), "label_rows": len(label_rows)},
        }
    )

    metrics.update(
        {
            "reports_row_count": len(report_rows),
            "labels_row_count": len(label_rows),
            "summary_row_count": len(summary_rows),
            "embedding_finite_ratio": round(finite_ratio, 4),
            "cluster_label_coverage_ratio": round(coverage, 4),
            "unique_report_clusters": len(report_ids),
        }
    )
    return _validation_result(gate, checks, metrics)


def _validate_agent_3(project_root: Path) -> dict[str, Any]:
    gate = "trend_sanity"
    trends = project_root / "frontend_app" / "data" / "monthly_cluster_trends.csv"
    spikes = project_root / "recent_spikes_report.csv"
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    ok_trends, miss_trends = _check_required_columns(
        trends, ["month_key", "cluster", "count", "total_reports_month", "normalized_rate"]
    )
    ok_spikes, miss_spikes = _check_required_columns(spikes, ["cluster_id", "max_rate_z", "month", "count", "rate", "name"])
    checks.append({"name": "trends_schema", "passed": ok_trends, "details": miss_trends})
    checks.append({"name": "spikes_schema", "passed": ok_spikes, "details": miss_spikes})
    if not ok_trends:
        return _validation_result(gate, checks, metrics)

    trend_rows = _read_csv_rows(trends)
    spike_rows = _read_csv_rows(spikes) if ok_spikes else []
    bad_rates = 0
    bad_counts = 0
    months = set()
    for r in trend_rows:
        months.add(str(r.get("month_key", "")))
        c = _to_int(r.get("count", ""))
        t = _to_int(r.get("total_reports_month", ""))
        nr_ok = _is_finite_float(r.get("normalized_rate", ""))
        nr = float(r["normalized_rate"]) if nr_ok else -1.0
        if c is None or t is None or c < 0 or t <= 0:
            bad_counts += 1
        if not nr_ok or nr < 0 or nr > 1.000001:
            bad_rates += 1

    checks.append(
        {
            "name": "normalized_rate_bounds",
            "passed": bad_rates == 0,
            "details": {"out_of_bound_rows": bad_rates},
        }
    )
    checks.append(
        {
            "name": "count_total_consistency",
            "passed": bad_counts == 0,
            "details": {"invalid_count_rows": bad_counts},
        }
    )
    checks.append(
        {
            "name": "sufficient_time_coverage",
            "passed": len(months) >= 3,
            "details": {"unique_months": len(months), "min_expected": 3},
        }
    )
    checks.append(
        {
            "name": "spike_file_non_empty",
            "passed": len(spike_rows) > 0,
            "details": {"spike_rows": len(spike_rows)},
        }
    )
    metrics.update(
        {
            "trend_rows": len(trend_rows),
            "spike_rows": len(spike_rows),
            "unique_months": len(months),
            "invalid_rate_rows": bad_rates,
            "invalid_count_rows": bad_counts,
        }
    )
    return _validation_result(gate, checks, metrics)


def _validate_agent_4(project_root: Path) -> dict[str, Any]:
    gate = "alert_completeness"
    alerts = project_root / "frontend_app" / "data" / "alerts.csv"
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    required = ["cluster", "cluster_label", "alert_type", "severity", "month_key", "message"]
    ok_alerts, miss_alerts = _check_required_columns(alerts, required)
    checks.append({"name": "alerts_schema", "passed": ok_alerts, "details": miss_alerts})
    if not ok_alerts:
        return _validation_result(gate, checks, metrics)

    rows = _read_csv_rows(alerts)
    severity_ok = {"low", "medium", "high", "critical"}
    bad_severity = sum(1 for r in rows if str(r.get("severity", "")).strip().lower() not in severity_ok)
    empty_message = sum(1 for r in rows if not str(r.get("message", "")).strip())

    checks.append({"name": "alerts_non_empty", "passed": len(rows) > 0, "details": {"row_count": len(rows)}})
    checks.append(
        {
            "name": "severity_values_valid",
            "passed": bad_severity == 0,
            "details": {"invalid_rows": bad_severity},
        }
    )
    checks.append(
        {
            "name": "message_completeness",
            "passed": empty_message == 0,
            "details": {"empty_message_rows": empty_message},
        }
    )
    metrics.update({"alert_rows": len(rows), "invalid_severity_rows": bad_severity, "empty_message_rows": empty_message})
    return _validation_result(gate, checks, metrics)


def _validate_agent_5(project_root: Path) -> dict[str, Any]:
    gate = "final_pipeline_sanity"
    required_files = [
        project_root / "frontend_app" / "data" / "clustered_reports.csv",
        project_root / "frontend_app" / "data" / "cluster_summary.csv",
        project_root / "frontend_app" / "data" / "monthly_cluster_trends.csv",
        project_root / "frontend_app" / "data" / "alerts.csv",
        project_root / "cluster_labels_report.csv",
        project_root / "recent_spikes_report.csv",
        project_root / "pipeline_validation_report.json",
    ]
    checks: list[dict[str, Any]] = []
    missing = [str(p) for p in required_files if not p.exists()]
    checks.append({"name": "all_required_outputs_present", "passed": len(missing) == 0, "details": missing})
    checks.append(
        {
            "name": "all_required_outputs_nontrivial_size",
            "passed": all(p.exists() and p.stat().st_size > 40 for p in required_files),
            "details": {str(p): (p.stat().st_size if p.exists() else 0) for p in required_files},
        }
    )
    return _validation_result(
        gate,
        checks,
        {
            "required_files_count": len(required_files),
            "present_files_count": len(required_files) - len(missing),
        },
    )


def _validate_agent_step(step_idx: int, project_root: Path) -> dict[str, Any]:
    if step_idx == 1:
        return _validate_agent_1(project_root)
    if step_idx == 2:
        return _validate_agent_2(project_root)
    if step_idx == 3:
        return _validate_agent_3(project_root)
    if step_idx == 4:
        return _validate_agent_4(project_root)
    return _validate_agent_5(project_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Agents 1-5 sequentially and write run_manifest.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config describing exact agent commands/artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline_runs"),
        help="Directory where each pipeline run folder is created.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when an agent fails.",
    )
    parser.add_argument(
        "--ignore-gate-fail",
        action="store_true",
        help="Continue pipeline even if a validation gate fails.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    try:
        agents = _load_config(args.config)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": _utc_now_iso(),
        "project_root": str(project_root),
        "config_path": str(args.config) if args.config else None,
        "output_dir": str(run_dir),
        "status": "running",
        "agents": [],
    }

    any_failed = False
    for idx, agent in enumerate(agents, start=1):
        print(f"Running {agent.name}: {' '.join(agent.command)}")
        result, ok = _run_agent(agent, project_root, run_dir)
        validation = _validate_agent_step(idx, project_root)
        result["validation"] = validation
        if not validation["passed"]:
            result["status"] = "failed_validation"
            ok = False
            print(f"  -> validation gate failed: {validation['gate']}")
        run_manifest["agents"].append(result)
        print(f"  -> {result['status']} (rc={result['return_code']})")
        if not ok:
            any_failed = True
            if args.stop_on_error or (not args.ignore_gate_fail and result["status"] == "failed_validation"):
                print("Stopping early due to failure (command or validation gate).")
                break

    run_manifest["finished_at_utc"] = _utc_now_iso()
    run_manifest["status"] = "failed" if any_failed else "success"
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Run manifest written to: {run_manifest_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
