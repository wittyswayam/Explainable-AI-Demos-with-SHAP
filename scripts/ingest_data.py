"""
scripts/ingest_data.py
=======================
DVC pipeline Stage 1 — Download and cache raw datasets.
Supports Kaggle API download with local fallback.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def ingest_churn(cfg: dict, output_path: Path) -> None:
    """Download Bank Customer Churn dataset."""
    if output_path.exists():
        logger.info("Churn dataset already cached at %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
        kaggle_path = cfg["dataset"]["kaggle_path"]
        logger.info("Downloading from Kaggle: %s", kaggle_path)
        dl_path = kagglehub.dataset_download(kaggle_path)
        csv_files = list(Path(dl_path).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV in Kaggle download")
        df = pd.read_csv(csv_files[0])
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.to_csv(output_path, index=False)
        logger.info("Saved churn dataset: %s (%d rows)", output_path, len(df))
    except Exception as exc:
        logger.warning("Kaggle download failed (%s) — generating synthetic fallback", exc)
        _generate_synthetic_churn(output_path)


def ingest_california_housing(output_path: Path) -> None:
    """Download California Housing via sklearn."""
    if output_path.exists():
        logger.info("California Housing already cached at %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(output_path, index=False)
    logger.info("Saved California Housing: %s (%d rows)", output_path, len(df))


def _generate_synthetic_churn(output_path: Path) -> None:
    """Generate a synthetic churn dataset matching the real schema."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 10_000
    geography = rng.choice(["France", "Spain", "Germany"], n, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], n)
    age = rng.integers(18, 92, n)
    credit_score = rng.integers(300, 850, n)
    tenure = rng.integers(0, 10, n)
    balance = rng.choice([0.0], n, p=[1.0]) * 0 + rng.uniform(0, 250_000, n) * rng.integers(0, 2, n)
    num_products = rng.integers(1, 5, n)
    has_cr_card = rng.integers(0, 2, n)
    is_active_member = rng.integers(0, 2, n)
    estimated_salary = rng.uniform(10_000, 200_000, n)
    exited = (
        (age > 50).astype(int) * 0.3
        + (balance == 0).astype(int) * 0.2
        + (num_products == 1).astype(int) * 0.2
        + rng.uniform(0, 0.3, n)
    ) > 0.5
    df = pd.DataFrame({
        "rownumber": range(1, n + 1),
        "customerid": rng.integers(10_000_000, 20_000_000, n),
        "surname": ["Synthetic"] * n,
        "creditscore": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance.round(2),
        "numofproducts": num_products,
        "hascrcard": has_cr_card,
        "isactivemember": is_active_member,
        "estimatedsalary": estimated_salary.round(2),
        "exited": exited.astype(int),
    })
    df.to_csv(output_path, index=False)
    logger.info("Generated synthetic churn dataset: %s (%d rows)", output_path, n)


if __name__ == "__main__":
    with open("configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    ingest_churn(cfg, Path("data/raw/churn_dataset.csv"))
    ingest_california_housing(Path("data/raw/california_housing.csv"))
    logger.info("Data ingestion complete.")
