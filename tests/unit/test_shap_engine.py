"""
Unit Tests — SHAP Engine
=========================
Tests for ExplainerFactory, SHAPEngine, and ExplanationResult.
Uses lightweight sklearn models to avoid heavy dependency loading in CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from src.explainability.shap_engine import (
    ExplainerConfig,
    ExplainerFactory,
    ExplainerType,
    ExplanationResult,
    ExplanationScope,
    SHAPEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def diabetes_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def clf_model(iris_data):
    X_train, _, y_train, _ = iris_data
    model = GradientBoostingClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def reg_model(diabetes_data):
    X_train, _, y_train, _ = diabetes_data
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# ExplainerFactory tests
# ---------------------------------------------------------------------------
class TestExplainerFactory:
    def test_resolves_tree_type_for_gbm(self, clf_model):
        resolved = ExplainerFactory._resolve_type(clf_model, ExplainerType.AUTO)
        assert resolved == ExplainerType.TREE

    def test_resolves_linear_type_for_ridge(self, reg_model):
        resolved = ExplainerFactory._resolve_type(reg_model, ExplainerType.AUTO)
        assert resolved == ExplainerType.LINEAR

    def test_explicit_type_overrides_auto(self, clf_model):
        resolved = ExplainerFactory._resolve_type(clf_model, ExplainerType.KERNEL)
        assert resolved == ExplainerType.KERNEL

    def test_background_sampling(self):
        X = pd.DataFrame(np.random.rand(1000, 5))
        sampled = ExplainerFactory._prepare_background(X, n_samples=100)
        assert len(sampled) == 100

    def test_background_no_sampling_when_small(self):
        X = pd.DataFrame(np.random.rand(50, 5))
        sampled = ExplainerFactory._prepare_background(X, n_samples=100)
        assert len(sampled) == 50

    def test_predict_fn_for_classifier(self, clf_model):
        fn = ExplainerFactory._get_predict_fn(clf_model)
        X = np.random.rand(5, 4)
        result = fn(X)
        assert result.shape == (5,)

    def test_create_tree_explainer(self, clf_model, iris_data):
        X_train, _, _, _ = iris_data
        config = ExplainerConfig(explainer_type=ExplainerType.AUTO)
        explainer, resolved_type = ExplainerFactory.create(clf_model, X_train, config)
        assert resolved_type == ExplainerType.TREE
        assert explainer is not None

    def test_create_linear_explainer(self, reg_model, diabetes_data):
        X_train, _, _, _ = diabetes_data
        config = ExplainerConfig(explainer_type=ExplainerType.AUTO)
        explainer, resolved_type = ExplainerFactory.create(reg_model, X_train, config)
        assert resolved_type == ExplainerType.LINEAR
        assert explainer is not None


# ---------------------------------------------------------------------------
# SHAPEngine tests
# ---------------------------------------------------------------------------
class TestSHAPEngine:
    @pytest.mark.asyncio
    async def test_explain_returns_result(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        engine = SHAPEngine(model=clf_model, background_data=X_train)
        result = await engine.explain(X_test.iloc[:3])
        assert isinstance(result, ExplanationResult)
        assert result.shap_values is not None
        assert len(result.feature_names) == X_test.shape[1]

    @pytest.mark.asyncio
    async def test_computation_time_recorded(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        engine = SHAPEngine(model=clf_model, background_data=X_train)
        result = await engine.explain(X_test.iloc[:2])
        assert result.computation_time_ms > 0

    @pytest.mark.asyncio
    async def test_global_importance_has_all_features(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        engine = SHAPEngine(model=clf_model, background_data=X_train)
        result = await engine.explain(X_test)
        imp = result.global_feature_importance()
        assert len(imp) == X_test.shape[1]

    @pytest.mark.asyncio
    async def test_global_importance_sorted_descending(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        engine = SHAPEngine(model=clf_model, background_data=X_train)
        result = await engine.explain(X_test)
        imp = result.global_feature_importance()
        values = list(imp.values())
        assert values == sorted(values, reverse=True)

    @pytest.mark.asyncio
    async def test_explain_with_custom_feature_names(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        custom_names = [f"custom_{i}" for i in range(X_test.shape[1])]
        engine = SHAPEngine(model=clf_model, background_data=X_train)
        result = await engine.explain(X_test.iloc[:2], feature_names=custom_names)
        assert result.feature_names == custom_names

    @pytest.mark.asyncio
    async def test_with_cache(self, clf_model, iris_data):
        X_train, X_test, _, _ = iris_data
        mock_cache = MagicMock()
        mock_cache.get = asyncio.coroutine(lambda k: None)
        mock_cache.set = asyncio.coroutine(lambda k, v, ttl=None: True)
        engine = SHAPEngine(model=clf_model, background_data=X_train, cache=mock_cache)
        result = await engine.explain(X_test.iloc[:2])
        assert result is not None


# ---------------------------------------------------------------------------
# ExplanationResult tests
# ---------------------------------------------------------------------------
class TestExplanationResult:
    def _make_result(self, n: int = 5, f: int = 4) -> ExplanationResult:
        return ExplanationResult(
            shap_values=np.random.randn(n, f),
            base_values=np.zeros(n),
            feature_names=[f"feat_{i}" for i in range(f)],
            feature_values=np.random.rand(n, f),
            explainer_type=ExplainerType.TREE,
            model_output_type="raw",
            computation_time_ms=42.0,
            additivity_check_passed=True,
            expected_value=0.5,
        )

    def test_to_dict_is_json_serialisable(self):
        import json
        result = self._make_result()
        d = result.to_dict()
        json.dumps(d)  # should not raise

    def test_global_feature_importance_shape(self):
        result = self._make_result(n=10, f=6)
        imp = result.global_feature_importance()
        assert len(imp) == 6

    def test_global_importance_non_negative(self):
        result = self._make_result()
        imp = result.global_feature_importance()
        assert all(v >= 0 for v in imp.values())

    def test_to_dict_contains_required_keys(self):
        result = self._make_result()
        d = result.to_dict()
        required = {"shap_values", "base_values", "feature_names", "feature_values",
                    "explainer_type", "computation_time_ms", "additivity_check_passed"}
        assert required.issubset(d.keys())
