"""
flows/xai_prefect_pipeline.py
================================
Prefect 3 workflow orchestration for the XAI Platform.
Alternative to the Airflow DAG — Prefect is Python-native and
requires no separate scheduler infrastructure.

Features vs Airflow DAG:
- Pure Python — no XML/YAML configuration
- Native async support
- Built-in retry policies per task
- Automatic result caching (cache_key_fn)
- Infrastructure-agnostic (local, Docker, Kubernetes, ECS)
- Real-time UI at http://localhost:4200

Run:
    prefect server start                     # start local server
    python flows/xai_prefect_pipeline.py     # run pipeline
    prefect deploy flows/xai_prefect_pipeline.py:xai_training_flow  # deploy to server
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task, get_run_logger
from prefect.cache_policies import FLOW_PARAMETERS
from prefect.tasks import task_input_hash
from prefect.runtime import flow_run

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(
    name="ingest-data",
    description="Download Kaggle churn dataset and California Housing.",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
    tags=["data", "ingestion"],
)
def ingest_data(config_path: str = "configs/data.yaml") -> Dict[str, int]:
    """Download and cache raw datasets."""
    import sys
    sys.path.insert(0, ".")
    import yaml
    from scripts.ingest_data import ingest_churn, ingest_california_housing

    log = get_run_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    churn_path = Path("data/raw/churn_dataset.csv")
    housing_path = Path("data/raw/california_housing.csv")

    ingest_churn(cfg, churn_path)
    ingest_california_housing(housing_path)

    import pandas as pd
    churn_rows = len(pd.read_csv(churn_path))
    housing_rows = len(pd.read_csv(housing_path))

    log.info("Ingested churn=%d rows, housing=%d rows", churn_rows, housing_rows)
    return {"churn_rows": churn_rows, "housing_rows": housing_rows}


@task(
    name="validate-data",
    description="Schema validation, completeness checks, distribution profiling.",
    retries=2,
    retry_delay_seconds=10,
    tags=["data", "validation"],
)
def validate_data(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Validate dataset quality and schema."""
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    import yaml

    log = get_run_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv("data/raw/churn_dataset.csv")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    expected = cfg["validation"]["expected_columns"]
    missing_cols = [c for c in expected if c not in df.columns]
    missing_rate = float(df.isnull().mean().max())

    validation_passed = (
        len(missing_cols) == 0
        and missing_rate < cfg["validation"]["max_missing_rate"]
        and len(df) >= cfg["validation"]["min_rows"]
    )

    if not validation_passed:
        raise ValueError(f"Data validation FAILED: missing_cols={missing_cols}, missing_rate={missing_rate:.3f}")

    report = {
        "rows": len(df),
        "missing_cols": missing_cols,
        "max_missing_rate": missing_rate,
        "validation_passed": validation_passed,
    }

    with open(OUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info("Data validation PASSED: %s", report)
    return report


@task(
    name="featurise",
    description="Feature engineering and train/test split.",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6),
    tags=["features"],
)
def featurise(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Feature engineering pipeline."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn import set_config
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    set_config(transform_output="pandas")
    log = get_run_logger()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv("data/raw/churn_dataset.csv")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=["rownumber", "customerid", "surname"], errors="ignore")

    feat_cfg = cfg["features"]
    X = df.drop(columns=[feat_cfg["target_col"]])
    y = df[feat_cfg["target_col"]].values

    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), feat_cfg["numerical_cols"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feat_cfg["categorical_cols"]),
    ], remainder="passthrough")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_proc.to_parquet(OUT_DIR / "X_train.parquet")
    X_test_proc.to_parquet(OUT_DIR / "X_test.parquet")
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_test.npy", y_test)

    with open(OUT_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    log.info("Featurised: train=%d rows, test=%d rows, features=%d", len(X_train_proc), len(X_test_proc), X_train_proc.shape[1])
    return {"train_rows": len(X_train_proc), "test_rows": len(X_test_proc), "n_features": X_train_proc.shape[1]}


@task(
    name="train-model",
    description="Optuna HPO + VotingClassifier ensemble + MLflow tracking.",
    retries=1,
    retry_delay_seconds=60,
    timeout_seconds=3600,
    tags=["training", "mlflow"],
)
def train_model(
    config_path: str = "configs/training.yaml",
    mlflow_uri: str = "http://localhost:5000",
) -> Dict[str, Any]:
    """Train model with Optuna HPO and MLflow tracking."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    import yaml
    from src.pipelines.training_pipeline import TrainingConfig, TrainingPipeline
    from scripts.train import churn_model_builder_fixed, churn_model_builder_optuna

    log = get_run_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    X_train = pd.read_parquet(OUT_DIR / "X_train.parquet")
    X_test = pd.read_parquet(OUT_DIR / "X_test.parquet")
    y_train = np.load(OUT_DIR / "y_train.npy")
    y_test = np.load(OUT_DIR / "y_test.npy")

    config = TrainingConfig(
        experiment_name=cfg["experiment"]["name"] + "-prefect",
        run_name=f"prefect-{flow_run.get_flow_run_context().flow_run.name}",
        task="classification",
        cv_folds=cfg["training"]["cv_folds"],
        optuna_trials=cfg["training"]["optuna_trials"],
        metric="roc_auc",
        mlflow_tracking_uri=mlflow_uri,
        shap_sample_size=cfg["shap"]["background_sample_size"],
    )

    def model_builder(trial=None, params=None):
        if trial is not None:
            return churn_model_builder_optuna(trial)
        return churn_model_builder_fixed(params or {})

    pipeline = TrainingPipeline(config=config)
    result = pipeline.run(
        X_train=X_train, y_train=y_train,
        X_val=X_test, y_val=y_test,
        model_builder=model_builder,
        feature_names=list(X_train.columns),
    )

    with open(OUT_DIR / "trained_model.pkl", "wb") as f:
        pickle.dump(result.model, f)

    log.info("Training complete: run_id=%s val_roc_auc=%.4f", result.mlflow_run_id, result.val_metrics.get("roc_auc", 0))
    return {
        "mlflow_run_id": result.mlflow_run_id,
        "val_roc_auc": result.val_metrics.get("roc_auc", 0.0),
        "val_f1": result.val_metrics.get("f1", 0.0),
        "training_time_s": result.training_time_seconds,
    }


@task(
    name="quality-gate",
    description="Enforce ROC-AUC quality threshold before proceeding.",
    tags=["quality"],
)
def quality_gate(train_result: Dict[str, Any], min_roc_auc: float = 0.80) -> bool:
    """Block pipeline if model quality is below threshold."""
    log = get_run_logger()
    val_roc_auc = train_result["val_roc_auc"]
    if val_roc_auc < min_roc_auc:
        raise ValueError(
            f"Quality gate FAILED: ROC-AUC={val_roc_auc:.4f} < threshold={min_roc_auc}"
        )
    log.info("Quality gate PASSED: ROC-AUC=%.4f", val_roc_auc)
    return True


@task(
    name="generate-shap",
    description="Compute SHAP global + local explanations on test set.",
    tags=["explainability", "shap"],
)
async def generate_shap() -> Dict[str, Any]:
    """Compute and persist SHAP values."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    from src.explainability.shap_engine import ExplainerConfig, SHAPEngine

    log = get_run_logger()

    with open(OUT_DIR / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(OUT_DIR / "X_train.parquet")
    X_test = pd.read_parquet(OUT_DIR / "X_test.parquet")
    feature_names = list(X_train.columns)

    engine = SHAPEngine(
        model=model,
        background_data=X_train,
        config=ExplainerConfig(background_sample_size=200),
    )
    result = await engine.explain(X_test, feature_names=feature_names)

    np.savez_compressed(
        OUT_DIR / "shap_values.npz",
        shap_values=result.shap_values,
        base_value=np.array([result.expected_value]),
    )
    with open(OUT_DIR / "shap_importance.json", "w") as f:
        json.dump(result.global_importance, f, indent=2)

    top_feat = list(result.global_importance.keys())[0]
    log.info("SHAP complete: explainer=%s, additivity=%s, top=%s", result.explainer_type.value, result.additivity_check_passed, top_feat)
    return {
        "explainer_type": result.explainer_type.value,
        "additivity_passed": result.additivity_check_passed,
        "top_feature": top_feat,
    }


@task(
    name="detect-drift",
    description="PSI + KS multivariate drift analysis.",
    tags=["monitoring", "drift"],
)
def detect_drift() -> Dict[str, Any]:
    """Run drift detection on test vs training distributions."""
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    from src.monitoring.drift_detector import DriftDetector

    log = get_run_logger()
    X_train = pd.read_parquet(OUT_DIR / "X_train.parquet")
    X_test = pd.read_parquet(OUT_DIR / "X_test.parquet")

    detector = DriftDetector(reference_data=X_train)
    report = detector.detect(production_data=X_test)

    if report.alert_required:
        log.warning("DRIFT ALERT: severity=%s, n_drifted=%d", report.overall_severity.value, report.n_drifted)
    else:
        log.info("Drift check CLEAR: severity=%s", report.overall_severity.value)

    return report.summary()


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(
    name="xai-training-pipeline",
    description="End-to-end XAI Platform ML pipeline: data → features → train → SHAP → drift",
    version="2.0.0",
    log_prints=True,
)
async def xai_training_flow(
    data_config: str = "configs/data.yaml",
    training_config: str = "configs/training.yaml",
    mlflow_uri: str = "http://localhost:5000",
    min_roc_auc: float = 0.80,
) -> Dict[str, Any]:
    """
    Full XAI Platform training and explanation pipeline.

    Stages:
        1. Ingest → 2. Validate → 3. Featurise → 4. Train → 5. Quality Gate
        → 6. SHAP (async) + Drift Detection (parallel)
    """
    # Sequential stages
    ingest_result = ingest_data(data_config)
    validate_result = validate_data(data_config, wait_for=[ingest_result])
    featurise_result = featurise(data_config, wait_for=[validate_result])
    train_result = train_model(training_config, mlflow_uri, wait_for=[featurise_result])

    # Quality gate
    gate_passed = quality_gate(train_result, min_roc_auc)

    # Parallel: SHAP + drift (both wait on quality gate)
    shap_future = generate_shap.submit(wait_for=[gate_passed])
    drift_future = detect_drift.submit(wait_for=[gate_passed])

    shap_result = await shap_future.result()
    drift_result = drift_future.result()

    return {
        "status": "success",
        "ingest": ingest_result,
        "train": train_result,
        "shap": shap_result,
        "drift": drift_result,
    }


if __name__ == "__main__":
    asyncio.run(xai_training_flow())
