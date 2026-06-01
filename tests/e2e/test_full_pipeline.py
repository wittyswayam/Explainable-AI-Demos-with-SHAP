"""
tests/e2e/test_full_pipeline.py
=================================
End-to-end tests: data ingestion → training → SHAP explanation → API response.
Uses synthetic data to avoid Kaggle dependency in CI.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.explainability.shap_engine import ExplainerConfig, SHAPEngine
from src.models.evaluator import ModelEvaluator
from src.monitoring.drift_detector import DriftDetector


# ---------------------------------------------------------------------------
# E2E: Training → Evaluation → Explanation
# ---------------------------------------------------------------------------

class TestFullMLPipeline:
    @pytest.fixture(scope="class")
    def pipeline_outputs(self):
        """Run a miniaturised full pipeline and return all outputs."""
        # Data
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        evaluator = ModelEvaluator(task="classification")
        eval_report = evaluator.evaluate(model, X_test.values, y_test.values)

        # SHAP
        engine = SHAPEngine(
            model=model,
            background_data=X_train,
            config=ExplainerConfig(background_sample_size=50),
        )

        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "eval_report": eval_report,
            "shap_engine": engine,
            "feature_names": list(X_train.columns),
        }

    def test_model_performance_acceptable(self, pipeline_outputs):
        report = pipeline_outputs["eval_report"]
        assert report.classification.roc_auc > 0.85, (
            f"ROC-AUC too low: {report.classification.roc_auc}"
        )

    def test_model_recall_above_threshold(self, pipeline_outputs):
        report = pipeline_outputs["eval_report"]
        assert report.classification.recall > 0.70

    @pytest.mark.asyncio
    async def test_shap_explains_test_set(self, pipeline_outputs):
        engine = pipeline_outputs["shap_engine"]
        X_test = pipeline_outputs["X_test"]
        result = await engine.explain(X_test.iloc[:10])
        assert result.shap_values.shape[0] == 10
        assert len(result.feature_names) == X_test.shape[1]

    @pytest.mark.asyncio
    async def test_shap_additivity(self, pipeline_outputs):
        engine = pipeline_outputs["shap_engine"]
        X_test = pipeline_outputs["X_test"]
        result = await engine.explain(X_test.iloc[:5])
        assert result.additivity_check_passed

    @pytest.mark.asyncio
    async def test_global_importance_all_features(self, pipeline_outputs):
        engine = pipeline_outputs["shap_engine"]
        X_test = pipeline_outputs["X_test"]
        result = await engine.explain(X_test)
        importance = result.global_feature_importance()
        assert len(importance) == X_test.shape[1]
        # All importances should be non-negative
        assert all(v >= 0 for v in importance.values())

    def test_drift_detection_no_drift_on_same_distribution(self, pipeline_outputs):
        X_train = pipeline_outputs["X_train"]
        X_test = pipeline_outputs["X_test"]
        detector = DriftDetector(X_train)
        report = detector.detect(X_test)
        assert report.n_features_tested > 0
        # Test set from same distribution should not trigger high-severity drift
        from src.monitoring.drift_detector import DriftSeverity
        assert report.overall_severity in {DriftSeverity.NONE, DriftSeverity.LOW}

    def test_evaluation_report_serialisable(self, pipeline_outputs):
        report = pipeline_outputs["eval_report"]
        d = report.to_dict()
        # Must be JSON serialisable
        json_str = json.dumps(d, default=str)
        restored = json.loads(json_str)
        assert "classification" in restored

    @pytest.mark.asyncio
    async def test_counterfactual_generation(self, pipeline_outputs):
        from src.explainability.counterfactual import CounterfactualConfig, CounterfactualEngine

        model = pipeline_outputs["model"]
        X_train = pipeline_outputs["X_train"]
        X_test = pipeline_outputs["X_test"]
        feature_names = pipeline_outputs["feature_names"]

        engine = CounterfactualEngine(
            model=model,
            training_data=X_train.values,
            feature_names=feature_names,
            config=CounterfactualConfig(n_counterfactuals=1, max_iterations=50, tolerance=0.5),
        )

        instance = X_test.values[0]
        cfs = engine.generate(instance, desired_class=0)
        # Should return at least 0 (may not find CF in 50 iterations)
        assert isinstance(cfs, list)
        if cfs:
            assert cfs[0].distance >= 0

    @pytest.mark.asyncio
    async def test_lime_cross_validation(self, pipeline_outputs):
        from src.explainability.lime_engine import LIMEEngine

        model = pipeline_outputs["model"]
        X_train = pipeline_outputs["X_train"]
        X_test = pipeline_outputs["X_test"]
        feature_names = pipeline_outputs["feature_names"]

        lime_engine = LIMEEngine(
            training_data=X_train.values,
            feature_names=feature_names,
            mode="classification",
        )

        result = lime_engine.explain(
            predict_fn=model.predict_proba,
            instance=X_test.values[0],
            num_samples=500,
            num_features=10,
        )
        assert isinstance(result.feature_weights, dict)
        assert result.computation_time_ms > 0
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_shap_lime_comparison(self, pipeline_outputs):
        from src.explainability.lime_engine import LIMEEngine

        model = pipeline_outputs["model"]
        X_train = pipeline_outputs["X_train"]
        X_test = pipeline_outputs["X_test"]
        feature_names = pipeline_outputs["feature_names"]

        # Get SHAP importance
        shap_engine = pipeline_outputs["shap_engine"]
        shap_result = await shap_engine.explain(X_test.iloc[:20])
        shap_imp = shap_result.global_feature_importance()

        # Get LIME result
        lime_engine = LIMEEngine(
            training_data=X_train.values,
            feature_names=feature_names,
            mode="classification",
        )
        lime_result = lime_engine.explain(
            model.predict_proba, X_test.values[0], num_samples=500
        )

        comparison = lime_engine.compare_with_shap(lime_result, shap_imp, top_n=5)

        assert "overlap_pct" in comparison
        assert "agreement" in comparison
        assert comparison["agreement"] in {"HIGH", "MEDIUM", "LOW"}
        assert 0 <= comparison["overlap_count"] <= 5
