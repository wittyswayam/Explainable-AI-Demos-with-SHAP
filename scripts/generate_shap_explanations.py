"""
scripts/generate_shap_explanations.py
=======================================
DVC pipeline Stage 6 — Generate and persist SHAP values.
Loads trained model, computes global + local SHAP attributions,
saves .npz arrays and JSON importance for downstream analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging
from src.explainability.shap_engine import ExplainerConfig, SHAPEngine

configure_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    model_path = Path("data/processed/trained_model.pkl")
    x_train_path = Path("data/processed/X_train.parquet")
    x_test_path = Path("data/processed/X_test.parquet")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artefacts
    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(x_train_path)
    X_test = pd.read_parquet(x_test_path)
    feature_names = list(X_train.columns)

    logger.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)

    # Build engine
    config = ExplainerConfig(background_sample_size=200, check_additivity=True)
    engine = SHAPEngine(model=model, background_data=X_train, config=config)

    # Compute SHAP on test set
    logger.info("Computing SHAP values on %d test instances...", len(X_test))
    result = await engine.explain(X_test, feature_names=feature_names)

    # Persist SHAP arrays
    np.savez_compressed(
        out_dir / "shap_values.npz",
        shap_values=result.shap_values,
        base_values=result.base_values if isinstance(result.base_values, np.ndarray) else np.array([result.base_values]),
        feature_values=result.feature_values,
    )
    logger.info("Saved shap_values.npz")

    # Persist importance JSON
    importance = result.global_feature_importance()
    with open(out_dir / "shap_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    logger.info("Saved shap_importance.json | top feature: %s=%.4f", list(importance.keys())[0], list(importance.values())[0])

    # Summary
    logger.info(
        "SHAP generation complete | explainer=%s | additivity=%s | time_ms=%.1f",
        result.explainer_type.value,
        result.additivity_check_passed,
        result.computation_time_ms,
    )
    print("=" * 60)
    print("SHAP Global Importance (Top 10):")
    for feat, val in list(importance.items())[:10]:
        bar = "█" * int(val / max(importance.values()) * 30)
        print(f"  {feat:<30} {bar} {val:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
