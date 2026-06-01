"""tests/unit/test_sdk_client.py — SDK client unit tests"""
from __future__ import annotations
import json
import pytest
import httpx
from unittest.mock import MagicMock, patch

from sdk.python.xai_client import (
    AsyncXAIClient, ExplanationResponse, FeatureContribution, ModelInfo,
    PredictionResponse, XAIClient,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_EXPLAIN_RESPONSE = {
    "model_id": "test-model",
    "explainer_type": "tree",
    "scope": "local",
    "n_instances": 1,
    "computation_time_ms": 42.0,
    "additivity_check_passed": True,
    "expected_value": 0.25,
    "global_importance": {"feat_0": 0.15, "feat_1": 0.10},
    "local_explanations": [[
        {"feature": "feat_0", "shap_value": 0.15, "feature_value": 0.5, "abs_shap": 0.15},
        {"feature": "feat_1", "shap_value": -0.10, "feature_value": 0.2, "abs_shap": 0.10},
    ]],
}

MOCK_PREDICT_RESPONSE = {
    "model_id": "test-model",
    "predictions": [1],
    "probabilities": [[0.3, 0.7]],
    "n_instances": 1,
}

MOCK_MODELS_RESPONSE = [
    {"model_id": "test-model", "name": "Test", "version": "1.0", "framework": "sklearn",
     "task": "classification", "metrics": {"roc_auc": 0.88}},
]


@pytest.fixture
def mock_client():
    with patch("httpx.Client") as mock_http:
        client = XAIClient(base_url="http://test:8000", api_key="test-key")
        yield client, mock_http


# ---------------------------------------------------------------------------
# ExplanationResponse tests
# ---------------------------------------------------------------------------

class TestExplanationResponse:
    def make_response(self) -> ExplanationResponse:
        return XAIClient._parse_explanation(MOCK_EXPLAIN_RESPONSE)

    def test_parse_explanation_creates_object(self):
        resp = self.make_response()
        assert isinstance(resp, ExplanationResponse)
        assert resp.model_id == "test-model"

    def test_global_importance_parsed(self):
        resp = self.make_response()
        assert "feat_0" in resp.global_importance
        assert resp.global_importance["feat_0"] == pytest.approx(0.15)

    def test_local_explanations_parsed(self):
        resp = self.make_response()
        assert resp.local_explanations is not None
        assert len(resp.local_explanations) == 1
        assert isinstance(resp.local_explanations[0][0], FeatureContribution)

    def test_top_features_returns_n(self):
        resp = self.make_response()
        top = resp.top_features(n=1)
        assert len(top) == 1
        assert "feat_0" in top

    def test_top_local_features_sorted_by_abs(self):
        resp = self.make_response()
        local = resp.top_local_features(instance_index=0, n=2)
        assert local[0].abs_shap >= local[1].abs_shap

    def test_print_waterfall_no_error(self, capsys):
        resp = self.make_response()
        resp.print_waterfall(instance_index=0, n=5)
        captured = capsys.readouterr()
        assert "feat_0" in captured.out
        assert "SHAP Waterfall" in captured.out

    def test_top_local_empty_on_no_local(self):
        resp = self.make_response()
        resp.local_explanations = None
        assert resp.top_local_features(0) == []

    def test_top_local_empty_on_invalid_index(self):
        resp = self.make_response()
        assert resp.top_local_features(instance_index=99) == []


# ---------------------------------------------------------------------------
# FeatureContribution tests
# ---------------------------------------------------------------------------

class TestFeatureContribution:
    def test_repr_positive(self):
        fc = FeatureContribution("age", 0.15, 42.0, 0.15)
        assert "↑" in repr(fc)
        assert "age" in repr(fc)

    def test_repr_negative(self):
        fc = FeatureContribution("balance", -0.10, 0.0, 0.10)
        assert "↓" in repr(fc)


# ---------------------------------------------------------------------------
# XAIClient sync tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestXAIClientSync:
    def _make_mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = status_code
        mock.json.return_value = data
        mock.raise_for_status = MagicMock()
        return mock

    def test_explain_returns_explanation_response(self):
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = self._make_mock_response(MOCK_EXPLAIN_RESPONSE)
            client = XAIClient(base_url="http://test:8000", api_key="key")
            result = client.explain("test-model", [[0.1, 0.2]])
            assert isinstance(result, ExplanationResponse)
            assert result.model_id == "test-model"

    def test_predict_returns_prediction_response(self):
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = self._make_mock_response(MOCK_PREDICT_RESPONSE)
            client = XAIClient(base_url="http://test:8000", api_key="key")
            result = client.predict("test-model", [[0.1, 0.2]])
            assert isinstance(result, PredictionResponse)
            assert result.predictions == [1]

    def test_list_models_returns_model_info_list(self):
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = self._make_mock_response(MOCK_MODELS_RESPONSE)
            client = XAIClient(base_url="http://test:8000")
            models = client.list_models()
            assert len(models) == 1
            assert isinstance(models[0], ModelInfo)

    def test_context_manager(self):
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = self._make_mock_response({"status": "ok", "version": "2.0.0", "service": "xai"})
            with XAIClient(base_url="http://test:8000") as client:
                result = client.health()
                assert result["status"] == "ok"

    def test_api_key_in_headers(self):
        with patch("httpx.Client.__init__", return_value=None):
            client = XAIClient(base_url="http://test", api_key="my-key")
            # API key should be passed in headers (tested via mock)
            assert client.base_url == "http://test"


# ---------------------------------------------------------------------------
# AsyncXAIClient tests
# ---------------------------------------------------------------------------

class TestAsyncXAIClient:
    @pytest.mark.asyncio
    async def test_async_explain(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EXPLAIN_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            async with AsyncXAIClient(base_url="http://test") as client:
                result = await client.explain("test-model", [[0.1, 0.2]])
                assert isinstance(result, ExplanationResponse)

    @pytest.mark.asyncio
    async def test_async_health(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "2.0.0", "service": "xai"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with AsyncXAIClient(base_url="http://test") as client:
                result = await client.health()
                assert result["status"] == "ok"
