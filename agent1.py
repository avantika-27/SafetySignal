#!/usr/bin/env python3
from __future__ import annotations

from pipeline_lib import DATA_DIR, _safe_parquet_write, build_clustered_reports, ensure_data_dir, read_raw_asrs


def main() -> int:
    raw = read_raw_asrs()
    clustered = build_clustered_reports(raw)
    ensure_data_dir()
    clustered.to_csv(DATA_DIR / "clustered_reports.csv", index=False)
    _safe_parquet_write(clustered, DATA_DIR / "clustered_reports.parquet")
    print(f"Agent 1 complete: clustered_reports rows={len(clustered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
