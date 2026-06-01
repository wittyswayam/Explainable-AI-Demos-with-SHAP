"""
src/monitoring/metrics_collector.py
=====================================
Production Prometheus custom metric collectors for the XAI Platform.

Exposes metrics referenced in `docker/prometheus_alerts.yml`:
  - xai_drift_psi_score{feature}         — PSI per feature from latest drift scan
  - xai_model_roc_auc{model_id}          — Latest validation ROC-AUC per model
  - xai_model_versions_by_stage{stage}   — MLflow model registry counts
  - xai_explanation_duration_ms          — SHAP computation time histogram
  - xai_explanation_additivity_failures  — Additivity check failure counter
  - xai_cache_hit_total / xai_cache_miss_total — Redis cache effectiveness

Without this module the AlertManager rules reference non-existent metrics,
making the entire monitoring stack a no-op. This module wires everything together.

Integration:
    # In src/api/main.py — add to create_app():
    from src.monitoring.metrics_collector import XAIMetricsCollector
    collector = XAIMetricsCollector(mlflow_uri=settings.MLFLOW_TRACKING_URI)
    collector.register()

    # The /metrics endpoint (prometheus_client.generate_latest) will then
    # include all custom gauges alongside the standard httpx metrics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static metrics (updated inline during request processing)
# ---------------------------------------------------------------------------

# Explanation computation time (milliseconds) — histogram for p50/p95/p99
EXPLANATION_DURATION_MS = Histogram(
    "xai_explanation_duration_ms",
    "SHAP explanation computation time in milliseconds",
    ["model_id", "explainer_type", "scope"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

# Additivity failures counter
EXPLANATION_ADDITIVITY_FAILURES = Counter(
    "xai_explanation_additivity_failures_total",
    "Total SHAP explanations that failed the additivity property check",
    ["model_id", "explainer_type"],
)

# Cache counters
CACHE_HIT = Counter(
    "xai_cache_hit_total",
    "Total SHAP value cache hits",
    ["model_id"],
)
CACHE_MISS = Counter(
    "xai_cache_miss_total",
    "Total SHAP value cache misses",
    ["model_id"],
)

# LIME cross-validation agreement gauge (0=LOW, 1=MEDIUM, 2=HIGH)
LIME_AGREEMENT_GAUGE = Gauge(
    "xai_lime_shap_agreement",
    "LIME-SHAP feature ranking agreement level (0=LOW, 1=MEDIUM, 2=HIGH)",
    ["model_id"],
)

# Counterfactual generation success rate
COUNTERFACTUAL_SUCCESS = Counter(
    "xai_counterfactual_generated_total",
    "Total successfully generated counterfactual explanations",
    ["model_id", "desired_class"],
)
COUNTERFACTUAL_FAILED = Counter(
    "xai_counterfactual_failed_total",
    "Total counterfactual generation failures",
    ["model_id"],
)


# ---------------------------------------------------------------------------
# Helper functions for inline metric recording
# ---------------------------------------------------------------------------

def record_explanation(
    model_id: str,
    explainer_type: str,
    scope: str,
    duration_ms: float,
    additivity_passed: bool,
    cache_hit: bool,
) -> None:
    """
    Record explanation metrics inline during API request processing.

    Call this from the /explain endpoint after computing SHAP values.

    Parameters
    ----------
    model_id : registered model identifier
    explainer_type : "tree" | "linear" | "kernel" | "gradient"
    scope : "local" | "global"
    duration_ms : SHAP computation time in milliseconds
    additivity_passed : whether additivity check succeeded
    cache_hit : whether result was served from Redis cache
    """
    EXPLANATION_DURATION_MS.labels(
        model_id=model_id,
        explainer_type=explainer_type,
        scope=scope,
    ).observe(duration_ms)

    if not additivity_passed:
        EXPLANATION_ADDITIVITY_FAILURES.labels(
            model_id=model_id,
            explainer_type=explainer_type,
        ).inc()

    if cache_hit:
        CACHE_HIT.labels(model_id=model_id).inc()
    else:
        CACHE_MISS.labels(model_id=model_id).inc()


def record_lime_agreement(model_id: str, agreement: str) -> None:
    """Record LIME-SHAP agreement level as a gauge."""
    level_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    LIME_AGREEMENT_GAUGE.labels(model_id=model_id).set(level_map.get(agreement, -1))


def record_counterfactual(model_id: str, desired_class: int, success: bool) -> None:
    """Record counterfactual generation outcome."""
    if success:
        COUNTERFACTUAL_SUCCESS.labels(model_id=model_id, desired_class=str(desired_class)).inc()
    else:
        COUNTERFACTUAL_FAILED.labels(model_id=model_id).inc()


# ---------------------------------------------------------------------------
# Dynamic collectors (scraped on each Prometheus pull)
# ---------------------------------------------------------------------------

class DriftMetricsCollector:
    """
    Prometheus custom collector that reads the latest drift scan results
    and exposes PSI scores as gauges.

    Reads from `data/processed/drift_report.json` (written by DriftDetector
    or the Airflow/Prefect pipeline).

    Exposes:
        xai_drift_psi_score{feature="age"} 0.05
        xai_drift_psi_score{feature="balance"} 0.23
    """

    DRIFT_REPORT_PATH = Path("data/processed/drift_report.json")

    def __init__(self) -> None:
        self._lock = Lock()

    def collect(self):  # noqa: ANN201
        """Called by Prometheus client on each scrape."""
        psi_family = GaugeMetricFamily(
            "xai_drift_psi_score",
            "Population Stability Index per feature from latest drift scan",
            labels=["feature"],
        )

        try:
            if self.DRIFT_REPORT_PATH.exists():
                with self._lock:
                    with open(self.DRIFT_REPORT_PATH) as f:
                        report = json.load(f)

                psi_scores: Dict[str, float] = report.get("psi_scores", {})
                for feature, psi in psi_scores.items():
                    psi_family.add_metric([feature], float(psi))
            else:
                logger.debug("Drift report not found at %s", self.DRIFT_REPORT_PATH)
        except Exception as exc:
            logger.warning("DriftMetricsCollector error: %s", exc)

        yield psi_family


class ModelQualityCollector:
    """
    Prometheus custom collector exposing model performance metrics
    from the latest evaluation report and MLflow registry.

    Exposes:
        xai_model_roc_auc{model_id="churn-ensemble-v2"} 0.887
        xai_model_versions_by_stage{stage="Production"} 1
        xai_model_versions_by_stage{stage="Staging"} 2
    """

    EVAL_REPORT_PATH = Path("data/processed/evaluation_report.json")

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        model_name: str = "xai-churn-ensemble",
    ) -> None:
        self.mlflow_uri = mlflow_tracking_uri
        self.model_name = model_name
        self._lock = Lock()

    def collect(self):  # noqa: ANN201
        # ── ROC-AUC from evaluation report ────────────────────────────────
        roc_auc_family = GaugeMetricFamily(
            "xai_model_roc_auc",
            "Model validation ROC-AUC from latest evaluation",
            labels=["model_id"],
        )
        try:
            if self.EVAL_REPORT_PATH.exists():
                with self._lock:
                    with open(self.EVAL_REPORT_PATH) as f:
                        report = json.load(f)
                clf = report.get("classification", {})
                roc_auc = clf.get("roc_auc")
                if roc_auc is not None:
                    roc_auc_family.add_metric([self.model_name], float(roc_auc))
        except Exception as exc:
            logger.warning("ModelQualityCollector (roc_auc) error: %s", exc)

        yield roc_auc_family

        # ── MLflow model registry stage counts ────────────────────────────
        versions_family = GaugeMetricFamily(
            "mlflow_model_versions_by_stage",
            "Number of MLflow model versions by stage",
            labels=["stage"],
        )
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=self.mlflow_uri)
            stage_counts: Dict[str, int] = {}
            for mv in client.search_model_versions(f"name='{self.model_name}'"):
                stage = mv.current_stage
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            for stage, count in stage_counts.items():
                versions_family.add_metric([stage], float(count))
            # Always emit Production even if 0
            if "Production" not in stage_counts:
                versions_family.add_metric(["Production"], 0.0)
        except Exception as exc:
            logger.debug("ModelQualityCollector (mlflow) skipped: %s", exc)

        yield versions_family


class ExplanationQualityCollector:
    """
    Prometheus custom collector for SHAP explanation quality metrics.

    Reads from `data/processed/shap_importance.json` and exposes
    the top-feature importance values as gauges — useful for tracking
    feature ranking drift (if top feature changes, model behaviour changed).

    Exposes:
        xai_shap_global_importance{feature="age", model_id="..."} 0.183
    """

    SHAP_IMPORTANCE_PATH = Path("data/processed/shap_importance.json")
    TOP_N = 10

    def __init__(self, model_id: str = "xai-churn-ensemble") -> None:
        self.model_id = model_id
        self._lock = Lock()

    def collect(self):  # noqa: ANN201
        importance_family = GaugeMetricFamily(
            "xai_shap_global_importance",
            "Mean absolute SHAP value per feature (global importance)",
            labels=["model_id", "feature"],
        )
        try:
            if self.SHAP_IMPORTANCE_PATH.exists():
                with self._lock:
                    with open(self.SHAP_IMPORTANCE_PATH) as f:
                        importance: Dict[str, float] = json.load(f)
                for feature, value in list(importance.items())[: self.TOP_N]:
                    importance_family.add_metric([self.model_id, feature], float(value))
        except Exception as exc:
            logger.warning("ExplanationQualityCollector error: %s", exc)

        yield importance_family


# ---------------------------------------------------------------------------
# Collector registry manager
# ---------------------------------------------------------------------------

class XAIMetricsCollector:
    """
    Unified metrics collector that registers all custom Prometheus collectors.

    Usage (in src/api/main.py lifespan):
        collector = XAIMetricsCollector(
            mlflow_uri=settings.MLFLOW_TRACKING_URI,
            model_name="xai-churn-ensemble",
        )
        collector.register()

    After registration, `GET /metrics` will include:
      - xai_drift_psi_score{feature=...}
      - xai_model_roc_auc{model_id=...}
      - mlflow_model_versions_by_stage{stage=...}
      - xai_shap_global_importance{model_id=..., feature=...}
    """

    def __init__(
        self,
        mlflow_uri: str = "http://localhost:5000",
        model_name: str = "xai-churn-ensemble",
        registry: CollectorRegistry = REGISTRY,
    ) -> None:
        self.registry = registry
        self._collectors = [
            DriftMetricsCollector(),
            ModelQualityCollector(mlflow_tracking_uri=mlflow_uri, model_name=model_name),
            ExplanationQualityCollector(model_id=model_name),
        ]
        self._registered = False

    def register(self) -> None:
        """Register all custom collectors with the Prometheus registry."""
        if self._registered:
            logger.warning("XAIMetricsCollector already registered — skipping")
            return
        for collector in self._collectors:
            try:
                self.registry.register(collector)
                logger.info("Registered collector: %s", type(collector).__name__)
            except Exception as exc:
                logger.warning("Failed to register %s: %s", type(collector).__name__, exc)
        self._registered = True
        logger.info("XAIMetricsCollector registered: %d collectors", len(self._collectors))

    def unregister(self) -> None:
        """Unregister all collectors (call during application shutdown)."""
        for collector in self._collectors:
            try:
                self.registry.unregister(collector)
            except Exception:
                pass
        self._registered = False

    def write_drift_report(self, drift_report_dict: Dict[str, Any]) -> None:
        """
        Persist a drift report dict so DriftMetricsCollector can expose it.

        Call this after each DriftDetector.detect() run in the pipeline.

        Parameters
        ----------
        drift_report_dict : output of DriftReport.summary() merged with psi_scores
        """
        path = DriftMetricsCollector.DRIFT_REPORT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(drift_report_dict, f, indent=2)
        logger.info("Drift report written to %s", path)

    @staticmethod
    def health_check() -> Dict[str, Any]:
        """Return health status of the metrics collection system."""
        return {
            "drift_report_exists": DriftMetricsCollector.DRIFT_REPORT_PATH.exists(),
            "eval_report_exists": ModelQualityCollector.EVAL_REPORT_PATH.exists(),
            "shap_importance_exists": ExplanationQualityCollector.SHAP_IMPORTANCE_PATH.exists(),
        }
