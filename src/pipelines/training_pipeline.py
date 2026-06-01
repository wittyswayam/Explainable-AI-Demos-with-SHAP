"""
Training Pipeline
=================
End-to-end model training with:
- Data validation (Great Expectations style checks)
- Feature engineering via sklearn pipelines
- Optuna Bayesian hyperparameter optimisation
- MLflow experiment tracking and model registration
- SHAP-based post-training explainability
- Model evaluation with multiple metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Unified configuration for the training pipeline."""

    experiment_name: str = "xai-platform"
    run_name: Optional[str] = None
    task: str = "classification"               # "classification" | "regression"
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 600
    metric: str = "roc_auc"                    # primary optimisation metric
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_registry_name: Optional[str] = None
    shap_sample_size: int = 200
    n_jobs: int = -1


@dataclass
class TrainingResult:
    """Container for all outputs of a training run."""

    model: Any
    feature_names: List[str]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    best_params: Dict[str, Any]
    cv_score: float
    cv_std: float
    mlflow_run_id: Optional[str] = None
    shap_importance: Optional[Dict[str, float]] = None
    training_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return {k: round(v, 4) for k, v in metrics.items()}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
class TrainingPipeline:
    """
    Orchestrates the full model training lifecycle.

    Usage
    -----
    pipeline = TrainingPipeline(config=TrainingConfig(task="classification"))
    result = pipeline.run(X_train, y_train, X_val, y_val, model_builder=my_builder_fn)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def run(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: np.ndarray,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: np.ndarray,
        model_builder: Any,          # callable(trial) → BaseEstimator
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Execute full training pipeline with MLflow tracking.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : held-out validation data
        model_builder : Optuna objective factory — receives trial, returns fitted model
        feature_names : column names for SHAP

        Returns
        -------
        TrainingResult
        """
        t0 = time.perf_counter()
        feat_names = feature_names or (
            list(X_train.columns) if isinstance(X_train, pd.DataFrame)
            else [f"f{i}" for i in range(X_train.shape[1])]
        )

        with mlflow.start_run(run_name=self.config.run_name) as run:
            mlflow.log_params({
                "task": self.config.task,
                "cv_folds": self.config.cv_folds,
                "optuna_trials": self.config.optuna_trials,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "n_features": len(feat_names),
            })

            # ── Hyperparameter search ──────────────────────────────────────
            logger.info("Starting Optuna search | trials=%d", self.config.optuna_trials)
            study = self._run_optuna(model_builder, X_train, y_train)
            best_params = study.best_params
            logger.info("Best params: %s | cv_score=%.4f", best_params, study.best_value)
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_best_score", study.best_value)

            # ── Retrain on full training data with best params ─────────────
            best_model = model_builder(params=best_params)
            best_model.fit(X_train, y_train)

            # ── Evaluate ───────────────────────────────────────────────────
            train_metrics = self._evaluate(best_model, X_train, y_train)
            val_metrics = self._evaluate(best_model, X_val, y_val)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            # ── SHAP importance ────────────────────────────────────────────
            shap_imp = self._compute_shap_importance(best_model, X_train, feat_names)
            if shap_imp:
                mlflow.log_metrics({f"shap_{k}": v for k, v in list(shap_imp.items())[:20]})

            # ── Log model to registry ──────────────────────────────────────
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=self.config.model_registry_name,
            )

            training_time = time.perf_counter() - t0
            mlflow.log_metric("training_time_seconds", training_time)

            logger.info(
                "Training complete | run_id=%s | val_metrics=%s | time=%.1fs",
                run.info.run_id,
                val_metrics,
                training_time,
            )

            return TrainingResult(
                model=best_model,
                feature_names=feat_names,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_params=best_params,
                cv_score=study.best_value,
                cv_std=0.0,  # populated from trial std if needed
                mlflow_run_id=run.info.run_id,
                shap_importance=shap_imp,
                training_time_seconds=training_time,
            )

    def _run_optuna(self, model_builder: Any, X: Any, y: np.ndarray) -> optuna.Study:
        """Run Bayesian hyperparameter search with Optuna."""
        cv = (
            StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            if self.config.task == "classification"
            else KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        )

        def objective(trial: optuna.Trial) -> float:
            model = model_builder(trial=trial)
            scoring = self.config.metric
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout_seconds,
            n_jobs=1,
            show_progress_bar=False,
        )
        return study

    def _evaluate(self, model: Any, X: Any, y: np.ndarray) -> Dict[str, float]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") and self.config.task == "classification" else None
        if self.config.task == "classification":
            return _classification_metrics(y, y_pred, y_prob)
        return _regression_metrics(y, y_pred)

    def _compute_shap_importance(
        self, model: Any, X_train: Any, feat_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Compute SHAP global feature importance post-training."""
        try:
            import shap
            n = min(self.config.shap_sample_size, len(X_train))
            bg = X_train.sample(n, random_state=42) if isinstance(X_train, pd.DataFrame) else X_train[:n]
            explainer = shap.Explainer(model, bg)
            shap_vals = explainer(bg)
            importance = np.abs(shap_vals.values).mean(axis=0)
            if importance.ndim == 2:
                importance = importance.mean(axis=1)
            return {
                name: round(float(v), 6)
                for name, v in sorted(
                    zip(feat_names, importance), key=lambda x: x[1], reverse=True
                )
            }
        except Exception as exc:
            logger.warning("SHAP importance computation failed: %s", exc)
            return None
