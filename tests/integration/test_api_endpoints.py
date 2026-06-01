"""
Integration Tests — FastAPI Endpoints
======================================
Uses httpx AsyncClient to test the full request/response cycle.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_liveness(self, client):
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["alive"] is True

    @pytest.mark.asyncio
    async def test_readiness(self, client):
        resp = await client.get("/health/ready")
        assert resp.status_code == 200


class TestExplainEndpoints:
    @pytest.mark.asyncio
    async def test_explain_local(self, client):
        payload = {
            "model_id": "churn-ensemble-v2",
            "instances": [[0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.3]],
            "explainer_type": "auto",
            "scope": "local",
            "top_features": 5,
        }
        resp = await client.post("/api/v1/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "churn-ensemble-v2"
        assert "global_importance" in data
        assert data["n_instances"] == 1

    @pytest.mark.asyncio
    async def test_explain_global_scope(self, client):
        payload = {
            "model_id": "churn-ensemble-v2",
            "instances": [[0.1] * 8] * 10,
            "scope": "global",
            "explainer_type": "auto",
        }
        resp = await client.post("/api/v1/explain", json=payload)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_explain_invalid_scope_returns_422(self, client):
        payload = {
            "model_id": "test",
            "instances": [[0.1, 0.2]],
            "scope": "invalid_scope",
        }
        resp = await client.post("/api/v1/explain", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_explain_batch_async(self, client):
        payload = {
            "model_id": "test",
            "instances": [[0.1, 0.2, 0.3]] * 50,
            "scope": "local",
        }
        resp = await client.post("/api/v1/explain/batch", json=payload)
        assert resp.status_code == 202
        assert "job_id" in resp.json()


class TestPredictEndpoints:
    @pytest.mark.asyncio
    async def test_predict_returns_predictions(self, client):
        payload = {
            "model_id": "churn-ensemble-v2",
            "instances": [[0.5] * 10, [0.2] * 10],
            "return_probabilities": True,
        }
        resp = await client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert data["n_instances"] == 2
        assert data["probabilities"] is not None


class TestModelsEndpoints:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) >= 1

    @pytest.mark.asyncio
    async def test_get_model_by_id(self, client):
        resp = await client.get("/api/v1/models/churn-ensemble-v2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "churn-ensemble-v2"
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_model_returns_404(self, client):
        resp = await client.get("/api/v1/models/does-not-exist")
        assert resp.status_code == 404
