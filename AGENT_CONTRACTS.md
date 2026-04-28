# Agent 1-5 Contracts

This document defines file-level contracts for the 5-agent pipeline orchestrated by `run_pipeline.py`.

## Global Rules

- **File format:** UTF-8 CSV with headers, comma-delimited.
- **Date keys:** `month_key` uses `YYYY-MM`; `date` uses `YYYY-MM-DD`.
- **Cluster key convention:** final published outputs use `cluster` (int). Legacy handoff files may use `cluster_id` and are normalized downstream.
- **Idempotency:** each agent should overwrite its own outputs atomically (write temp + rename).
- **Failure behavior:** if an agent fails, it must not leave partially written outputs.

## Agent 1 - Report Clustering

- **Purpose:** transform raw incident reports into clustered point-level records.
- **Inputs:**
  - `master_asrs.csv`
- **Outputs:**
  - `frontend_app/data/clustered_reports.csv`

### Output schema: `clustered_reports.csv`

| column | type | required | constraints |
|---|---|---|---|
| `report_id` | string | yes | unique per report |
| `date` | string | yes | `YYYY-MM-DD` |
| `month_key` | string | yes | `YYYY-MM` |
| `text` | string | yes | non-empty narrative/summary |
| `cluster` | integer | yes | cluster id, `>= 0` |
| `cluster_label` | string | yes | human-readable topic label |
| `umap_x` | float | yes | finite numeric |
| `umap_y` | float | yes | finite numeric |
| `vessel_type` | string | no | optional context |

## Agent 2 - Cluster Labeling and Evidence

- **Purpose:** produce cluster-level labels, descriptions, and evidence summaries.
- **Inputs:**
  - `frontend_app/data/clustered_reports.csv`
- **Outputs:**
  - `cluster_labels_report.csv`
  - `frontend_app/data/cluster_summary.csv`

### Output schema: `cluster_labels_report.csv`

| column | type | required | constraints |
|---|---|---|---|
| `cluster_id` | integer | yes | primary key, `>= 0` |
| `size` | integer | yes | cluster size, `>= 0` |
| `name` | string | yes | short label |
| `description` | string | yes | 1-3 sentence summary |
| `keywords` | string | yes | comma-separated terms |
| `causal_summary` | string | no | root-cause style summary |
| `evidence_bullets` | string | no | compressed evidence text |
| `limitations` | string | no | caveats |
| `top_phase` | string | no | dominant flight phase |

### Output schema: `cluster_summary.csv`

| column | type | required | constraints |
|---|---|---|---|
| `cluster` | integer | yes | primary key, `>= 0` |
| `cluster_label` | string | yes | display label |
| `cluster_size` | integer | yes | `>= 0` |
| `description` | string | yes | cluster summary text |
| `keywords` | string | yes | comma-separated terms |
| `is_noise` | boolean | yes | true for noise bucket |

## Agent 3 - Trend and Spike Detection

- **Purpose:** compute monthly cluster trends and identify top spike events.
- **Inputs:**
  - `frontend_app/data/clustered_reports.csv`
  - `cluster_labels_report.csv`
- **Outputs:**
  - `frontend_app/data/monthly_cluster_trends.csv`
  - `recent_spikes_report.csv`

### Output schema: `monthly_cluster_trends.csv`

| column | type | required | constraints |
|---|---|---|---|
| `month_key` | string | yes | `YYYY-MM` |
| `cluster` | integer | yes | `>= 0` |
| `cluster_label` | string | yes | label for plotting |
| `count` | integer | yes | reports in month |
| `total_reports_month` | integer | yes | total corpus reports in month |
| `normalized_rate` | float | yes | `count / total_reports_month` |
| `pct_change` | float | yes | month-over-month percentage |
| `rolling_mean_rate` | float | no | optional smoothed signal |
| `max_rate_z` | float | no | optional z-score signal |

### Output schema: `recent_spikes_report.csv`

| column | type | required | constraints |
|---|---|---|---|
| `cluster_id` | integer | yes | `>= 0` |
| `max_rate_z` | float | yes | spike intensity |
| `month` | string | yes | `YYYY-MM` |
| `count` | integer | yes | cluster reports in spike month |
| `rate` | float | yes | normalized monthly rate |
| `name` | string | yes | cluster label |

## Agent 4 - Alert Generation

- **Purpose:** convert spike/trend signals into analyst-facing alerts.
- **Inputs:**
  - `frontend_app/data/monthly_cluster_trends.csv`
  - `cluster_labels_report.csv`
  - `recent_spikes_report.csv` (optional enrichment)
- **Outputs:**
  - `frontend_app/data/alerts.csv`

### Output schema: `alerts.csv` (minimum contract)

| column | type | required | constraints |
|---|---|---|---|
| `cluster` | integer | yes | `>= 0` |
| `cluster_label` | string | yes | label text |
| `alert_type` | string | yes | e.g. `spike` |
| `severity` | string | yes | enum: `low`, `medium`, `high`, `critical` |
| `month_key` | string | yes | `YYYY-MM` |
| `message` | string | yes | human-readable alert narrative |

### Optional extended alert fields

`alert_id`, `priority`, `confidence`, `growth_pct`, `incidents`, `root_cause`, `regulation_mapping`, `relevance_score`

## Agent 5 - Publish/Package for Dashboard

- **Purpose:** validate and publish the complete dashboard data bundle.
- **Inputs:**
  - `frontend_app/data/clustered_reports.csv`
  - `frontend_app/data/cluster_summary.csv`
  - `frontend_app/data/monthly_cluster_trends.csv`
  - `frontend_app/data/alerts.csv`
  - `cluster_labels_report.csv`
  - `recent_spikes_report.csv`
- **Outputs:**
  - `frontend_app/data/clustered_reports.csv` (validated pass-through or refreshed)
  - `frontend_app/data/cluster_summary.csv` (validated pass-through or refreshed)
  - `frontend_app/data/monthly_cluster_trends.csv` (validated pass-through or refreshed)
  - `frontend_app/data/alerts.csv` (validated pass-through or refreshed)
  - `pipeline_validation_report.json` (recommended)

### Output schema: `pipeline_validation_report.json` (recommended)

```json
{
  "run_id": "20260417T000000Z",
  "status": "success",
  "checks": [
    {
      "file": "frontend_app/data/alerts.csv",
      "row_count": 0,
      "required_columns_present": true,
      "null_violations": []
    }
  ]
}
```

## Handoff Expectations Between Agents

- Agent 2 must not start until Agent 1 output exists and has required columns.
- Agent 3 must not start until Agent 2 outputs exist.
- Agent 4 must not start until Agent 3 outputs exist.
- Agent 5 validates all final deliverables before pipeline completion.
