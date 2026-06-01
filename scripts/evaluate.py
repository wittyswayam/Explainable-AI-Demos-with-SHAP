"""
scripts/evaluate.py
====================
DVC pipeline Stage 5 — Comprehensive model evaluation.

Reads:  data/processed/trained_model.pkl
        data/processed/X_test.parquet
        data/processed/y_test.parquet
        data/processed/feature_names.json

Writes: data/processed/evaluation_report.json   (full metrics)
        data/processed/metrics.json              (DVC-tracked metrics subset)
        data/plots/calibration_curve.png
        data/plots/roc_curve.png
        data/plots/precision_recall_curve.png

The metrics.json file is tracked by DVC (`dvc metrics show`) and
surfaced in `dvc params diff` comparisons across experiments.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging
from src.models.evaluator import ModelEvaluator

configure_logging()
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = Path("data/plots")


def load_artefacts() -> tuple:
    """Load trained model, test data, and feature names."""
    model_path = PROCESSED_DIR / "trained_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run `dvc repro train` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test_df = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")
    y_test = y_test_df["target"].values if "target" in y_test_df.columns else y_test_df.iloc[:, 0].values

    feature_names_path = PROCESSED_DIR / "feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path) as f:
            feature_names = json.load(f)
    else:
        feature_names = list(X_test.columns)

    logger.info("Loaded model: %s", type(model).__name__)
    logger.info("Test set: %s rows, positive rate=%.2f%%", len(X_test), y_test.mean() * 100)
    return model, X_test, y_test, feature_names


def plot_roc_curve(model: object, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """Generate and save ROC curve plot."""
    from sklearn.metrics import roc_curve, roc_auc_score

    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X_test.values)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#3D84F5", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3D84F5")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Bank Churn Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve: %s", path)


def plot_precision_recall_curve(model: object, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """Generate and save Precision-Recall curve plot."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X_test.values)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#FF4B4B", linewidth=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline (random) = {baseline:.3f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#FF4B4B")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve — Bank Churn Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = PLOTS_DIR / "precision_recall_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve: %s", path)


def plot_calibration_curve(model: object, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """Generate and save calibration reliability diagram."""
    from sklearn.calibration import calibration_curve

    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X_test.values)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Calibration Analysis", fontsize=13, fontweight="bold")

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes[0].plot(prob_pred, prob_true, "o-", color="#3D84F5", linewidth=2, markersize=6, label="Model")
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=10)
    axes[0].set_ylabel("Fraction of Positives", fontsize=10)
    axes[0].set_title("Reliability Diagram")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Probability distribution histogram
    axes[1].hist(y_prob[y_test == 0], bins=30, alpha=0.6, color="#3D84F5", label="Retained (y=0)", density=True)
    axes[1].hist(y_prob[y_test == 1], bins=30, alpha=0.6, color="#FF4B4B", label="Churned (y=1)", density=True)
    axes[1].set_xlabel("Predicted Churn Probability", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].set_title("Score Distribution by Class")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration curve: %s", path)


def write_dvc_metrics(report_dict: dict) -> None:
    """
    Write DVC-compatible metrics.json.

    DVC metrics are flat key-value pairs tracked via `dvc metrics show`
    and compared across experiments with `dvc metrics diff`.
    Only scalar values — no nested dicts, no lists.
    """
    clf = report_dict.get("classification", {})
    cal = report_dict.get("calibration", {})

    dvc_metrics = {
        "roc_auc": clf.get("roc_auc", 0.0),
        "pr_auc": clf.get("pr_auc", 0.0),
        "f1": clf.get("f1", 0.0),
        "precision": clf.get("precision", 0.0),
        "recall": clf.get("recall", 0.0),
        "accuracy": clf.get("accuracy", 0.0),
        "mcc": clf.get("mcc", 0.0),
        "brier_score": clf.get("brier_score", 0.0),
        "log_loss": clf.get("log_loss_val", 0.0),
        "ece": cal.get("ece", 0.0),
        "optimal_threshold": clf.get("optimal_threshold", 0.5),
        "n_test_samples": report_dict.get("n_samples", 0),
        "n_features": report_dict.get("n_features", 0),
    }

    metrics_path = PROCESSED_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dvc_metrics, f, indent=2)

    logger.info("DVC metrics written to %s", metrics_path)
    return dvc_metrics


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test, y_test, feature_names = load_artefacts()

    # Run comprehensive evaluation
    evaluator = ModelEvaluator(task="classification", n_bins=10)
    report = evaluator.evaluate(
        model=model,
        X_test=X_test.values,
        y_test=y_test,
        feature_names=feature_names,
    )

    # Print summary to console
    report.print_summary()

    # Save full evaluation report
    report.to_json(PROCESSED_DIR / "evaluation_report.json")
    logger.info("Evaluation report saved to %s", PROCESSED_DIR / "evaluation_report.json")

    # Save DVC-tracked metrics subset
    dvc_metrics = write_dvc_metrics(report.to_dict())

    # Generate plots
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall_curve(model, X_test, y_test)
    plot_calibration_curve(model, X_test, y_test)

    # Quality gate check (warn only — hard gate is in Airflow/Prefect)
    roc_auc = dvc_metrics["roc_auc"]
    ece = dvc_metrics["ece"]
    if roc_auc < 0.80:
        logger.warning("⚠️ ROC-AUC=%.4f is below the 0.80 production threshold", roc_auc)
    else:
        logger.info("✅ ROC-AUC=%.4f passes the 0.80 quality threshold", roc_auc)

    if ece > 0.05:
        logger.warning("⚠️ ECE=%.4f indicates poor calibration (threshold=0.05)", ece)
    else:
        logger.info("✅ ECE=%.4f — model is well-calibrated", ece)

    print("\n📊 DVC Metrics (trackable with `dvc metrics show`):")
    for key, value in sorted(dvc_metrics.items()):
        print(f"  {key:<30} {value}")


if __name__ == "__main__":
    main()
