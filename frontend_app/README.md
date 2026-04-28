# Streamlit Safety Dashboard

## Run locally

```bash
cd frontend_app
python -m pip install -r requirements.txt
streamlit run app.py
```

## Data inputs

The app reads prepared CSV files from:

1. `frontend_app/data/` (preferred when populated)
2. project root fallback (`cluster_labels_report.csv`, `recent_spikes_report.csv`)

Expected files:

- `clustered_reports.csv`
- `cluster_summary.csv`
- `monthly_cluster_trends.csv`
- `alerts.csv`

If these are not fully available, the app creates safe fallback views from available exports.

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub.
2. In Streamlit Cloud, create a new app.
3. Set `main file path` to `frontend_app/app.py`.
4. Deploy.
