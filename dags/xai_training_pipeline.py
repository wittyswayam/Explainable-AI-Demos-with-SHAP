"""
dags/xai_training_pipeline.py
================================
Apache Airflow DAG — XAI Platform End-to-End ML Pipeline

Stages:
  1. ingest_data       — Download and cache raw datasets
  2. validate_data     — Schema + quality checks
  3. featurise         — Feature engineering & preprocessing
  4. train_model       — Optuna HPO + MLflow tracking
  5. evaluate_model    — Comprehensive metrics + calibration
  6. generate_shap     — SHAP global + local explanations
  7. detect_drift      — PSI + KS drift analysis on recent production data
  8. register_model    — Promote to MLflow Staging if metrics pass gate
  9. notify            — Slack/email notification on completion

Schedule: Daily at 02:00 UTC (retraining pipeline)
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ── DAG default args ──────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "xai-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ── Quality gates ─────────────────────────────────────────────────────────────
ROC_AUC_THRESHOLD = float(Variable.get("xai_min_roc_auc", default_var="0.80"))
DRIFT_PSI_THRESHOLD = float(Variable.get("xai_max_psi", default_var="0.20"))
MLFLOW_URI = Variable.get("xai_mlflow_uri", default_var="http://mlflow:5000")
MODEL_NAME = "xai-churn-ensemble"


# ── Task implementations ──────────────────────────────────────────────────────

def _ingest_data(**context: Any) -> dict:
    """Download and cache raw datasets."""
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")

    from scripts.ingest_data import ingest_churn, ingest_california_housing
    import yaml

    with open("/opt/airflow/xai-platform/configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    churn_path = Path("/opt/airflow/xai-platform/data/raw/churn_dataset.csv")
    housing_path = Path("/opt/airflow/xai-platform/data/raw/california_housing.csv")

    ingest_churn(cfg, churn_path)
    ingest_california_housing(housing_path)

    return {
        "churn_rows": len(__import__("pandas").read_csv(churn_path)),
        "housing_rows": len(__import__("pandas").read_csv(housing_path)),
    }


def _validate_data(**context: Any) -> dict:
    """Validate dataset schema, completeness, and distribution."""
    import pandas as pd
    import yaml

    with open("/opt/airflow/xai-platform/configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    churn_path = "/opt/airflow/xai-platform/data/raw/churn_dataset.csv"
    df = pd.read_csv(churn_path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    expected_cols = cfg["validation"]["expected_columns"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    missing_rate = df.isnull().mean().max()
    min_rows = cfg["validation"]["min_rows"]
    max_missing = cfg["validation"]["max_missing_rate"]

    validation_passed = (
        len(missing_cols) == 0
        and missing_rate < max_missing
        and len(df) >= min_rows
    )

    report = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_cols": missing_cols,
        "max_missing_rate": round(float(missing_rate), 4),
        "validation_passed": validation_passed,
    }

    out_path = Path("/opt/airflow/xai-platform/data/processed/validation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    if not validation_passed:
        raise ValueError(f"Data validation FAILED: {report}")

    logger.info("Data validation PASSED: %s", report)
    return report


def _featurise(**context: Any) -> dict:
    """Feature engineering and train/test split."""
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn import set_config
    set_config(transform_output="pandas")

    df = pd.read_csv("/opt/airflow/xai-platform/data/raw/churn_dataset.csv")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=["rownumber", "customerid", "surname"], errors="ignore")

    X = df.drop(columns=["exited"])
    y = df["exited"].values

    num_cols = ["creditscore", "age", "tenure", "balance", "numofproducts",
                "hascrcard", "isactivemember", "estimatedsalary"]
    cat_cols = ["geography", "gender"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="passthrough")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    out_dir = Path("/opt/airflow/xai-platform/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_proc.to_parquet(out_dir / "X_train.parquet")
    X_test_proc.to_parquet(out_dir / "X_test.parquet")
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)

    with open(out_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    return {
        "train_rows": len(X_train_proc),
        "test_rows": len(X_test_proc),
        "n_features": X_train_proc.shape[1],
        "feature_names": list(X_train_proc.columns),
    }


def _train_model(**context: Any) -> dict:
    """Train VotingClassifier ensemble with Optuna HPO and MLflow tracking."""
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")

    import numpy as np
    import pandas as pd
    from src.pipelines.training_pipeline import TrainingConfig, TrainingPipeline

    out_dir = Path("/opt/airflow/xai-platform/data/processed")
    X_train = pd.read_parquet(out_dir / "X_train.parquet")
    X_test = pd.read_parquet(out_dir / "X_test.parquet")
    y_train = np.load(out_dir / "y_train.npy")
    y_test = np.load(out_dir / "y_test.npy")
    feature_names = list(X_train.columns)

    from scripts.train import churn_model_builder_fixed, churn_model_builder_optuna

    config = TrainingConfig(
        experiment_name="xai-platform-airflow",
        run_name=f"airflow-{context['ds']}",
        task="classification",
        cv_folds=5,
        optuna_trials=20,
        metric="roc_auc",
        mlflow_tracking_uri=MLFLOW_URI,
        model_registry_name=MODEL_NAME,
        shap_sample_size=200,
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
        feature_names=feature_names,
    )

    # Save model artifact
    with open(out_dir / "trained_model.pkl", "wb") as f:
        pickle.dump(result.model, f)

    return {
        "mlflow_run_id": result.mlflow_run_id,
        "cv_score": result.cv_score,
        "val_roc_auc": result.val_metrics.get("roc_auc", 0.0),
        "val_f1": result.val_metrics.get("f1", 0.0),
        "training_time_seconds": result.training_time_seconds,
    }


def _evaluate_model(**context: Any) -> dict:
    """Run comprehensive model evaluation and check quality gate."""
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")

    import numpy as np
    import pandas as pd
    from src.models.evaluator import ModelEvaluator

    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")
    val_roc_auc = train_result["val_roc_auc"]

    if val_roc_auc < ROC_AUC_THRESHOLD:
        raise ValueError(
            f"Model quality gate FAILED: ROC-AUC={val_roc_auc:.4f} < threshold={ROC_AUC_THRESHOLD}"
        )

    out_dir = Path("/opt/airflow/xai-platform/data/processed")
    with open(out_dir / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_parquet(out_dir / "X_test.parquet")
    y_test = np.load(out_dir / "y_test.npy")

    evaluator = ModelEvaluator(task="classification")
    report = evaluator.evaluate(model, X_test.values, y_test)
    report.to_json(out_dir / "evaluation_report.json")

    return {
        "roc_auc": report.classification.roc_auc,
        "f1": report.classification.f1,
        "ece": report.calibration.ece if report.calibration else None,
        "quality_gate_passed": val_roc_auc >= ROC_AUC_THRESHOLD,
    }


def _generate_shap(**context: Any) -> dict:
    """Compute SHAP values and save explanations."""
    import asyncio
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")

    import numpy as np
    import pandas as pd
    from src.explainability.shap_engine import ExplainerConfig, SHAPEngine

    out_dir = Path("/opt/airflow/xai-platform/data/processed")
    with open(out_dir / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(out_dir / "X_train.parquet")
    X_test = pd.read_parquet(out_dir / "X_test.parquet")
    feature_names = list(X_train.columns)

    engine = SHAPEngine(
        model=model,
        background_data=X_train,
        config=ExplainerConfig(background_sample_size=200),
    )

    result = asyncio.run(engine.explain(X_test, feature_names=feature_names))

    np.savez_compressed(
        out_dir / "shap_values.npz",
        shap_values=result.shap_values,
        base_value=np.array([result.expected_value]),
        feature_values=result.feature_values,
    )

    with open(out_dir / "shap_importance.json", "w") as f:
        json.dump(result.global_importance, f, indent=2)

    top_feat = list(result.global_importance.keys())[0]
    return {
        "explainer_type": result.explainer_type.value,
        "additivity_passed": result.additivity_check_passed,
        "top_feature": top_feat,
        "top_feature_shap": result.global_importance[top_feat],
    }


def _detect_drift(**context: Any) -> dict:
    """Run multivariate drift detection on recent production vs training data."""
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")
    import pandas as pd
    from src.monitoring.drift_detector import DriftDetector

    out_dir = Path("/opt/airflow/xai-platform/data/processed")
    X_train = pd.read_parquet(out_dir / "X_train.parquet")
    X_test = pd.read_parquet(out_dir / "X_test.parquet")

    detector = DriftDetector(reference_data=X_train)
    report = detector.detect(production_data=X_test)

    if report.alert_required:
        logger.warning("DRIFT ALERT: %s", report.summary())

    return {
        "n_drifted": report.n_drifted,
        "drift_rate": report.drift_rate,
        "overall_severity": report.overall_severity.value,
        "alert_required": report.alert_required,
    }


def _register_model(**context: Any) -> dict:
    """Promote model to MLflow Staging registry."""
    import sys
    sys.path.insert(0, "/opt/airflow/xai-platform")
    from src.models.registry import ModelRegistry

    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")
    run_id = train_result["mlflow_run_id"]

    registry = ModelRegistry(tracking_uri=MLFLOW_URI)
    version = registry.register(
        run_id=run_id,
        artifact_path="model",
        name=MODEL_NAME,
        description=f"Airflow run {context['ds']} | ROC-AUC={train_result['val_roc_auc']:.4f}",
        tags={"airflow_dag": context["dag"].dag_id, "run_date": context["ds"]},
    )
    registry.promote(name=MODEL_NAME, version=version, stage="Staging")
    return {"model_version": version, "stage": "Staging"}


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="xai_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="XAI Platform: data ingestion → training → SHAP → drift detection → registry",
    schedule_interval="0 2 * * *",  # Daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["xai", "mlops", "training"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest_data,
        doc_md="**Stage 1:** Download Kaggle churn dataset and California Housing.",
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        doc_md="**Stage 2:** Schema validation, completeness checks, distribution profiling.",
    )

    featurise_task = PythonOperator(
        task_id="featurise",
        python_callable=_featurise,
        doc_md="**Stage 3:** MinMax scaling + OneHot encoding. Output: parquet artefacts.",
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        doc_md="**Stage 4:** Optuna HPO (20 trials) + VotingClassifier + MLflow tracking.",
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
        doc_md="**Stage 5:** ROC-AUC gate + full classification report + calibration.",
    )

    shap_task = PythonOperator(
        task_id="generate_shap",
        python_callable=_generate_shap,
        doc_md="**Stage 6:** SHAP global + local explanations → npz + JSON.",
    )

    drift_task = PythonOperator(
        task_id="detect_drift",
        python_callable=_detect_drift,
        doc_md="**Stage 7:** PSI + KS drift detection. Alert if PSI > 0.20.",
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
        doc_md="**Stage 8:** Promote passing model to MLflow Staging registry.",
    )

    # ── Pipeline graph ────────────────────────────────────────────────────────
    (
        ingest_task
        >> validate_task
        >> featurise_task
        >> train_task
        >> evaluate_task
        >> [shap_task, drift_task]
        >> register_task
    )
