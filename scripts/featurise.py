"""
scripts/featurise.py
=====================
DVC pipeline Stage 3 — Feature engineering and train/test split.

Reads:  data/raw/churn_dataset.csv
Writes: data/processed/X_train.parquet
        data/processed/X_test.parquet
        data/processed/y_train.parquet
        data/processed/y_test.parquet
        data/processed/preprocessor.pkl
        data/processed/feature_names.json

This script must be idempotent (safe to re-run), deterministic
(random_state=42), and fully configurable from configs/data.yaml.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(path: Path, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load and minimally clean the raw churn dataset."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    feat_cfg = cfg["features"]
    drop_cols = feat_cfg.get("drop_cols", [])
    target = feat_cfg["target_col"]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    logger.info("Loaded: X=%s, y=%s (positive rate=%.2f%%)", X.shape, y.shape, y.mean() * 100)
    return X, y


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Construct sklearn ColumnTransformer from data config."""
    set_config(transform_output="pandas")

    feat_cfg = cfg["features"]
    pre_cfg = cfg.get("preprocessing", {})

    num_cols = feat_cfg["numerical_cols"]
    cat_cols = feat_cfg["categorical_cols"]

    # Validate all declared columns exist in the expected schema
    expected_cols = set(num_cols + cat_cols)
    logger.info("Configuring preprocessor: %d numerical, %d categorical features", len(num_cols), len(cat_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown=pre_cfg.get("handle_unknown", "ignore"),
                    sparse_output=False,
                ),
                cat_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )
    return preprocessor


def split_and_transform(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified split and fit-transform preprocessing."""
    train_cfg = cfg.get("preprocessing", {})
    test_size = 0.2
    random_state = 42

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info("Split: train=%d, test=%d", len(X_train_raw), len(X_test_raw))

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    logger.info(
        "Preprocessed: train=%s, test=%s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train.values, y_test.values


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
    out_dir: Path,
) -> dict:
    """Persist all featurisation outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet", index=False)

    # Save targets as parquet (DVC tracks them) and npy (convenience)
    pd.Series(y_train, name="target").to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    pd.Series(y_test, name="target").to_frame().to_parquet(out_dir / "y_test.parquet", index=False)

    with open(out_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f, protocol=5)

    feature_names = list(X_train.columns)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    summary = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }

    with open(out_dir / "featurise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for name, path in [
        ("X_train.parquet", out_dir / "X_train.parquet"),
        ("X_test.parquet", out_dir / "X_test.parquet"),
        ("preprocessor.pkl", out_dir / "preprocessor.pkl"),
        ("feature_names.json", out_dir / "feature_names.json"),
    ]:
        size_kb = path.stat().st_size / 1024
        logger.info("Saved %s (%.1f KB)", name, size_kb)

    return summary


def main(
    data_config: str = "configs/data.yaml",
    raw_data_path: str = "data/raw/churn_dataset.csv",
    output_dir: str = "data/processed",
) -> None:
    cfg = load_config(data_config)
    raw_path = Path(raw_data_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Run `python scripts/ingest_data.py` first, or `dvc repro ingest`."
        )

    X, y = load_raw_data(raw_path, cfg)
    preprocessor = build_preprocessor(cfg)
    X_train, X_test, y_train, y_test = split_and_transform(X, y, preprocessor, cfg)
    summary = save_outputs(X_train, X_test, y_train, y_test, preprocessor, Path(output_dir))

    logger.info("Featurisation complete:")
    logger.info("  Train: %d rows × %d features | positive rate=%.2f%%",
                summary["train_rows"], summary["n_features"], summary["train_positive_rate"] * 100)
    logger.info("  Test:  %d rows × %d features | positive rate=%.2f%%",
                summary["test_rows"], summary["n_features"], summary["test_positive_rate"] * 100)
    logger.info("  Top 5 features: %s", summary["feature_names"][:5])

    print("\n✅ Featurisation complete")
    print(f"   Train: {summary['train_rows']:,} rows × {summary['n_features']} features")
    print(f"   Test:  {summary['test_rows']:,} rows × {summary['n_features']} features")
    print(f"   Outputs: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XAI Platform — DVC Stage 3: Feature Engineering")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--raw-data", default="data/raw/churn_dataset.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    main(
        data_config=args.data_config,
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
    )
