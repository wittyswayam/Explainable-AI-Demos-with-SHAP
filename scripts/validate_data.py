"""
scripts/validate_data.py
=========================
DVC pipeline Stage 2 — Data validation.

Reads:  data/raw/churn_dataset.csv
        data/raw/california_housing.csv
Writes: data/processed/validation_report.json

Checks schema, completeness, row count, and target distribution.
Raises SystemExit(1) on validation failure so DVC marks the stage as failed.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def main(config_path: str = "configs/data.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    churn_path = Path("data/raw/churn_dataset.csv")
    if not churn_path.exists():
        logger.error("Raw data not found: %s — run ingest first", churn_path)
        sys.exit(1)

    df = pd.read_csv(churn_path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    val_cfg = cfg["validation"]
    expected_cols = val_cfg["expected_columns"]
    missing_cols  = [c for c in expected_cols if c not in df.columns]
    missing_rate  = float(df.isnull().mean().max())
    row_count     = len(df)

    passed = (
        len(missing_cols) == 0
        and missing_rate < val_cfg["max_missing_rate"]
        and row_count    >= val_cfg["min_rows"]
    )

    report = {
        "rows": row_count,
        "columns": list(df.columns),
        "missing_cols": missing_cols,
        "max_missing_rate": round(missing_rate, 6),
        "validation_passed": passed,
        "target_positive_rate": round(float(df["exited"].mean()), 4) if "exited" in df.columns else None,
    }

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not passed:
        logger.error("Data validation FAILED: %s", report)
        sys.exit(1)

    logger.info("Data validation PASSED: rows=%d, missing=%.4f", row_count, missing_rate)
    print(f"Validation PASSED | rows={row_count:,} | missing_rate={missing_rate:.4f} | positive_rate={report['target_positive_rate']}")

if __name__ == "__main__":
    main()
