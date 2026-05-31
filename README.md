# SafetySignal

SafetySignal is an aviation safety analytics project that turns ASRS narratives into actionable cluster intelligence.
It includes:

- clustering and trend/spike detection on incident narratives
- a Streamlit dashboard for exploration and alerts
- FAA regulation retrieval (RAG) for contextual references
- cluster causal summaries and action checklist generation
- Boeing 737/MAX benchmark evaluation with realism diagnostics

## Repository Structure

- `agent1.py` ... `agent5.py`: staged pipeline agents
- `run_pipeline.py`: orchestrates all 5 agents and writes run manifests
- `pipeline_lib.py`: shared pipeline utilities
- `frontend_app/`: Streamlit dashboard app
- `faa_regulation_rag.py`: FAA corpus ingestion/retrieval helpers
- `evaluation/`: benchmark datasets, scripts, and retrospective case study
- `run_evaluation.py`: convenience wrapper to run evaluation

## Prerequisites

- Python 3.10+ recommended
- `pip`
- (Optional) virtual environment tool (`venv` or conda)

## Setup

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r frontend_app/requirements.txt
```

If you use conda, activate your env first and run only the `pip install` commands.

## Run the Data Pipeline

This executes Agent 1-5, validates outputs, and writes artifacts/manifests under `pipeline_runs/`.

```bash
python run_pipeline.py
```

Optional:

```bash
python run_pipeline.py --output-dir pipeline_runs --stop-on-error
```

### Main Pipeline Outputs

- `frontend_app/data/clustered_reports.csv`
- `cluster_labels_report.csv`
- `frontend_app/data/cluster_summary.csv`
- `recent_spikes_report.csv`
- `frontend_app/data/monthly_cluster_trends.csv`
- `frontend_app/data/alerts.csv`

## Run the Streamlit Dashboard

```bash
cd frontend_app
python -m streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Run Evaluation

From project root:

```bash
python run_evaluation.py
# or
python evaluation/run_b737_evaluation.py
```

Evaluation report is written to:

- `evaluation/reports/b737_eval_report.json`

This report includes:

- confusion metrics (`metrics_all`, `metrics_asrs_rows_only`, `metrics_operational_subset`)
- validation targets and pass/fail
- FPR guard checks-
- realism diagnostics (confidence intervals, bootstrap ranges, risk flags)
- Boeing MAX retrospective case study linkage

## FAA Corpus / RAG

- FAA source files are in `faa_corpus/`
- `faa_regulation_rag.py` contains ingestion/retrieval helpers


## Quick Start 

```bash
python -m pip install -r frontend_app/requirements.txt
python run_pipeline.py
python run_evaluation.py
cd frontend_app && python -m streamlit run app.py
```
