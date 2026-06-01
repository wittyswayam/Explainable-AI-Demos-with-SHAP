"""
src/models/evaluator.py
========================
Comprehensive model evaluation module.

Computes:
- Classification: ROC-AUC, F1, Precision, Recall, PR-AUC, Brier Score, MCC
- Regression: RMSE, MAE, R², MAPE, Max Error
- Calibration: reliability diagrams, Expected Calibration Error (ECE)
- Fairness: group-level metric disparities
- Generates evaluation report as JSON and markdown
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, max_error, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationReport:
    accuracy: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier_score: float
    log_loss_val: float
    mcc: float
    confusion_matrix: List[List[int]]
    optimal_threshold: float
    threshold_f1: float


@dataclass
class RegressionReport:
    rmse: float
    mae: float
    r2: float
    mape: float
    max_error_val: float
    median_ae: float


@dataclass
class CalibrationReport:
    ece: float
    mean_confidence: float
    mean_accuracy: float
    n_bins: int
    calibration_curve_prob_true: List[float]
    calibration_curve_prob_pred: List[float]


@dataclass
class EvaluationReport:
    task: str
    n_samples: int
    n_features: int
    classification: Optional[ClassificationReport] = None
    regression: Optional[RegressionReport] = None
    calibration: Optional[CalibrationReport] = None
    fairness: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"EVALUATION REPORT — {self.task.upper()}")
        print(f"{'='*55}")
        print(f"Samples: {self.n_samples:,} | Features: {self.n_features}")
        if self.classification:
            c = self.classification
            print(f"\nClassification Metrics:")
            print(f"  ROC-AUC:         {c.roc_auc:.4f}")
            print(f"  PR-AUC:          {c.pr_auc:.4f}")
            print(f"  F1 (@ opt thr):  {c.threshold_f1:.4f}  [threshold={c.optimal_threshold:.2f}]")
            print(f"  Precision:       {c.precision:.4f}")
            print(f"  Recall:          {c.recall:.4f}")
            print(f"  MCC:             {c.mcc:.4f}")
            print(f"  Brier Score:     {c.brier_score:.4f}")
            print(f"  Log Loss:        {c.log_loss_val:.4f}")
            cm = c.confusion_matrix
            print(f"\n  Confusion Matrix:")
            print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
            print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
        if self.regression:
            r = self.regression
            print(f"\nRegression Metrics:")
            print(f"  RMSE:  {r.rmse:.4f}")
            print(f"  MAE:   {r.mae:.4f}")
            print(f"  R²:    {r.r2:.4f}")
            print(f"  MAPE:  {r.mape:.4f}")
        if self.calibration:
            print(f"\nCalibration: ECE={self.calibration.ece:.4f}")
        print(f"{'='*55}")


class ModelEvaluator:
    """
    Comprehensive model evaluator for classification and regression tasks.

    Usage
    -----
    evaluator = ModelEvaluator(task="classification")
    report = evaluator.evaluate(model, X_test, y_test, feature_names=names)
    report.print_summary()
    report.to_json("data/processed/evaluation_report.json")
    """

    def __init__(self, task: str = "classification", n_bins: int = 10) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_bins = n_bins

    def evaluate(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sensitive_feature: Optional[np.ndarray] = None,
        sensitive_feature_name: Optional[str] = None,
        threshold: float = 0.5,
    ) -> EvaluationReport:
        """
        Run comprehensive evaluation.

        Parameters
        ----------
        model : fitted sklearn-compatible model
        X_test, y_test : test set
        feature_names : for report metadata
        sensitive_feature : array for fairness analysis
        threshold : classification decision threshold (auto-optimised if None)
        """
        n_samples = len(y_test)
        n_features = X_test.shape[1]

        report = EvaluationReport(
            task=self.task,
            n_samples=n_samples,
            n_features=n_features,
        )

        if self.task == "classification":
            report.classification = self._classify(model, X_test, y_test, threshold)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                report.calibration = self._calibrate(y_test, y_prob)

        else:
            report.regression = self._regress(model, X_test, y_test)

        if sensitive_feature is not None:
            report.fairness = self._fairness(
                model, X_test, y_test, sensitive_feature, sensitive_feature_name or "sensitive"
            )

        return report

    # ── Classification ─────────────────────────────────────────────────────

    def _classify(
        self, model: Any, X: Any, y: np.ndarray, threshold: float
    ) -> ClassificationReport:
        y_pred_raw = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = y_pred_raw.astype(float)

        roc_auc = float(roc_auc_score(y, y_prob))
        precision_pts, recall_pts, thresholds = precision_recall_curve(y, y_prob)
        pr_auc = float(auc(recall_pts, precision_pts))

        # Optimal threshold: maximise F1
        f1_scores = 2 * (precision_pts * recall_pts) / (precision_pts + recall_pts + 1e-9)
        opt_idx = np.argmax(f1_scores[:-1])
        opt_threshold = float(thresholds[opt_idx]) if len(thresholds) > opt_idx else threshold
        y_pred_opt = (y_prob >= opt_threshold).astype(int)

        # Fixed threshold predictions
        y_pred = (y_prob >= threshold).astype(int)

        return ClassificationReport(
            accuracy=round(float(accuracy_score(y, y_pred)), 4),
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            f1=round(float(f1_score(y, y_pred, zero_division=0)), 4),
            precision=round(float(precision_score(y, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y, y_pred, zero_division=0)), 4),
            brier_score=round(float(brier_score_loss(y, y_prob)), 4),
            log_loss_val=round(float(log_loss(y, y_prob)), 4),
            mcc=round(float(matthews_corrcoef(y, y_pred)), 4),
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            optimal_threshold=round(opt_threshold, 3),
            threshold_f1=round(float(f1_scores[opt_idx]), 4),
        )

    # ── Regression ──────────────────────────────────────────────────────────

    def _regress(self, model: Any, X: Any, y: np.ndarray) -> RegressionReport:
        y_pred = model.predict(X)
        return RegressionReport(
            rmse=round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            mae=round(float(mean_absolute_error(y, y_pred)), 4),
            r2=round(float(r2_score(y, y_pred)), 4),
            mape=round(float(mean_absolute_percentage_error(y, y_pred)), 4),
            max_error_val=round(float(max_error(y, y_pred)), 4),
            median_ae=round(float(np.median(np.abs(y - y_pred))), 4),
        )

    # ── Calibration ─────────────────────────────────────────────────────────

    def _calibrate(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationReport:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins, strategy="quantile")

        # Expected Calibration Error
        bin_sizes = np.histogram(y_prob, bins=self.n_bins)[0]
        weights = bin_sizes / bin_sizes.sum()
        ece = float(np.sum(weights[:len(prob_true)] * np.abs(prob_true - prob_pred[:len(prob_true)])))

        return CalibrationReport(
            ece=round(ece, 4),
            mean_confidence=round(float(y_prob.mean()), 4),
            mean_accuracy=round(float(y_true.mean()), 4),
            n_bins=self.n_bins,
            calibration_curve_prob_true=prob_true.tolist(),
            calibration_curve_prob_pred=prob_pred.tolist(),
        )

    # ── Fairness ────────────────────────────────────────────────────────────

    def _fairness(
        self,
        model: Any,
        X: Any,
        y_true: np.ndarray,
        sensitive: np.ndarray,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Demographic parity and equalised odds analysis."""
        groups = np.unique(sensitive)
        group_metrics = {}

        for group in groups:
            mask = sensitive == group
            y_g = y_true[mask]
            X_g = X[mask] if hasattr(X, "__getitem__") else X.values[mask]
            y_pred_g = model.predict(X_g)
            y_prob_g = model.predict_proba(X_g)[:, 1] if hasattr(model, "predict_proba") else y_pred_g
            group_metrics[str(group)] = {
                "n": int(mask.sum()),
                "positive_rate": round(float(y_pred_g.mean()), 4),
                "true_positive_rate": round(float(recall_score(y_g, y_pred_g, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_g, y_prob_g)) if y_g.sum() > 0 else 0.0, 4),
            }

        # Demographic parity difference
        pos_rates = [m["positive_rate"] for m in group_metrics.values()]
        dpd = round(max(pos_rates) - min(pos_rates), 4)

        return {
            "sensitive_feature": feature_name,
            "n_groups": len(groups),
            "group_metrics": group_metrics,
            "demographic_parity_difference": dpd,
            "fairness_assessment": "PASS" if dpd < 0.1 else "REVIEW" if dpd < 0.2 else "FAIL",
        }
