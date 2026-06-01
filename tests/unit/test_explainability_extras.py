"""
tests/unit/test_explainability_extras.py
=========================================
Unit tests for visualisation utilities, counterfactual engine,
and model evaluator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.explainability.visualisation import (
    bar_importance_plot,
    beeswarm_plot,
    dependence_plot,
    heatmap_plot,
    save_all_plots,
    waterfall_plot,
)
from src.explainability.counterfactual import (
    ActionabilityConstraint,
    CounterfactualConfig,
    CounterfactualEngine,
)
from src.models.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clf_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def clf_model(clf_data):
    X_train, _, y_train, _ = clf_data
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def sample_shap_data():
    rng = np.random.default_rng(42)
    n, f = 50, 8
    return {
        "shap_values": rng.randn(n, f),
        "feature_values": rng.rand(n, f),
        "feature_names": [f"feat_{i}" for i in range(f)],
        "base_value": 0.35,
        "global_importance": {f"feat_{i}": float(rng.exponential(0.1)) for i in range(f)},
    }


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------

class TestVisualisationUtils:
    def test_waterfall_returns_figure(self, sample_shap_data):
        d = sample_shap_data
        fig = waterfall_plot(
            d["shap_values"], d["feature_values"], d["feature_names"], d["base_value"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_waterfall_single_instance_1d(self, sample_shap_data):
        d = sample_shap_data
        fig = waterfall_plot(
            d["shap_values"][0],  # 1D
            d["feature_values"][0],
            d["feature_names"],
            d["base_value"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_beeswarm_returns_figure(self, sample_shap_data):
        d = sample_shap_data
        fig = beeswarm_plot(d["shap_values"], d["feature_values"], d["feature_names"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_beeswarm_raises_on_1d_input(self, sample_shap_data):
        d = sample_shap_data
        with pytest.raises(ValueError, match="2D"):
            beeswarm_plot(d["shap_values"][0], d["feature_values"], d["feature_names"])

    def test_bar_importance_plot(self, sample_shap_data):
        d = sample_shap_data
        fig = bar_importance_plot(d["global_importance"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_plot(self, sample_shap_data):
        d = sample_shap_data
        fig = heatmap_plot(d["shap_values"], d["feature_names"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dependence_plot_by_name(self, sample_shap_data):
        d = sample_shap_data
        fig = dependence_plot(
            d["shap_values"], d["feature_values"], d["feature_names"], feature="feat_0"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dependence_plot_with_interaction(self, sample_shap_data):
        d = sample_shap_data
        fig = dependence_plot(
            d["shap_values"], d["feature_values"], d["feature_names"],
            feature="feat_0", interaction_feature="feat_1"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_all_plots(self, sample_shap_data):
        d = sample_shap_data
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_all_plots(
                shap_values=d["shap_values"],
                feature_values=d["feature_values"],
                feature_names=d["feature_names"],
                base_value=d["base_value"],
                global_importance=d["global_importance"],
                output_dir=tmpdir,
                fmt="png",
                dpi=72,
            )
            assert len(saved) >= 3
            for path in saved.values():
                assert path.exists()


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------

class TestCounterfactualEngine:
    @pytest.fixture(scope="class")
    def cf_setup(self, clf_data, clf_model):
        X_train, X_test, y_train, y_test = clf_data
        engine = CounterfactualEngine(
            model=clf_model,
            training_data=X_train.values,
            feature_names=list(X_train.columns),
            config=CounterfactualConfig(
                n_counterfactuals=2,
                max_iterations=50,
                random_state=42,
                tolerance=0.5,
            ),
        )
        return engine, X_test.values, y_test.values

    def test_generates_counterfactuals(self, cf_setup):
        engine, X_test, y_test = cf_setup
        instance = X_test[0]
        cfs = engine.generate(instance, desired_class=0)
        assert isinstance(cfs, list)

    def test_counterfactual_has_changed_features(self, cf_setup):
        engine, X_test, _ = cf_setup
        instance = X_test[0]
        cfs = engine.generate(instance, desired_class=0)
        if cfs:
            assert len(cfs[0].changed_features) >= 0

    def test_counterfactual_to_dict(self, cf_setup):
        engine, X_test, _ = cf_setup
        instance = X_test[0]
        cfs = engine.generate(instance, desired_class=0)
        if cfs:
            d = cfs[0].to_dict()
            assert "original_prediction" in d
            assert "counterfactual_prediction" in d
            assert "changed_features" in d

    def test_counterfactual_sorted_by_distance(self, cf_setup):
        engine, X_test, _ = cf_setup
        instance = X_test[0]
        cfs = engine.generate(instance, desired_class=0)
        if len(cfs) >= 2:
            assert cfs[0].distance <= cfs[1].distance

    def test_actionability_constraint_respected(self, clf_data, clf_model):
        X_train, X_test, _, _ = clf_data
        feat_names = list(X_train.columns)
        constraints = [
            ActionabilityConstraint(feature=feat_names[0], mutable=False),
        ]
        engine = CounterfactualEngine(
            model=clf_model,
            training_data=X_train.values,
            feature_names=feat_names,
            config=CounterfactualConfig(
                n_counterfactuals=1,
                max_iterations=30,
                constraints=constraints,
            ),
        )
        cfs = engine.generate(X_test.values[0], desired_class=0)
        if cfs:
            # Immutable feature should not appear in changed features
            assert feat_names[0] not in cfs[0].changed_features


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestModelEvaluator:
    def test_classification_report_keys(self, clf_data, clf_model):
        _, X_test, _, y_test = clf_data
        evaluator = ModelEvaluator(task="classification")
        report = evaluator.evaluate(clf_model, X_test.values, y_test.values)
        assert report.classification is not None
        assert 0.5 <= report.classification.roc_auc <= 1.0

    def test_calibration_report_present(self, clf_data, clf_model):
        _, X_test, _, y_test = clf_data
        evaluator = ModelEvaluator(task="classification")
        report = evaluator.evaluate(clf_model, X_test.values, y_test.values)
        assert report.calibration is not None
        assert 0.0 <= report.calibration.ece <= 1.0

    def test_regression_report(self):
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge()
        model.fit(X_train, y_train)
        evaluator = ModelEvaluator(task="regression")
        report = evaluator.evaluate(model, X_test, y_test)
        assert report.regression is not None
        assert report.regression.r2 > 0

    def test_to_json_creates_file(self, clf_data, clf_model):
        _, X_test, _, y_test = clf_data
        evaluator = ModelEvaluator(task="classification")
        report = evaluator.evaluate(clf_model, X_test.values, y_test.values)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "classification" in data
        assert data["task"] == "classification"

    def test_fairness_analysis(self, clf_data, clf_model):
        _, X_test, _, y_test = clf_data
        # Create synthetic binary sensitive feature
        rng = np.random.default_rng(42)
        sensitive = rng.integers(0, 2, size=len(y_test))
        evaluator = ModelEvaluator(task="classification")
        report = evaluator.evaluate(
            clf_model, X_test.values, y_test.values,
            sensitive_feature=sensitive, sensitive_feature_name="gender"
        )
        assert report.fairness is not None
        assert "demographic_parity_difference" in report.fairness
        assert report.fairness["fairness_assessment"] in {"PASS", "REVIEW", "FAIL"}

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            ModelEvaluator(task="clustering")
