"""
sdk/python/xai_client.py
=========================
Official Python SDK for the Explainable AI Platform.

Provides a high-level, type-safe client with:
- Sync and async interfaces
- Automatic retry with exponential backoff
- JWT and API-key authentication
- Response caching (optional)
- Rich explanation objects with visualisation helpers

Installation:
    pip install xai-platform-sdk  # once published
    # or use directly from this repo

Usage:
    from sdk.python.xai_client import XAIClient

    client = XAIClient(base_url="http://localhost:8000", api_key="your-key")

    result = client.explain(
        model_id="churn-ensemble-v2",
        instances=[[0.5, 0.3, 42, 0.0, 2, 1, 1, 0, 1, 75000]],
    )
    print(result.top_features(n=5))
    result.plot_waterfall(instance_index=0)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


# ---------------------------------------------------------------------------
# Response objects
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    feature: str
    shap_value: float
    feature_value: float
    abs_shap: float

    def __repr__(self) -> str:
        direction = "↑" if self.shap_value > 0 else "↓"
        return f"{direction} {self.feature}: {self.shap_value:+.4f} (value={self.feature_value:.3g})"


@dataclass
class ExplanationResponse:
    model_id: str
    explainer_type: str
    scope: str
    n_instances: int
    computation_time_ms: float
    additivity_check_passed: bool
    expected_value: float
    global_importance: Dict[str, float]
    local_explanations: Optional[List[List[FeatureContribution]]] = None
    raw: Optional[Dict] = None

    def top_features(self, n: int = 10) -> Dict[str, float]:
        """Return top-n features by global SHAP importance."""
        return dict(list(self.global_importance.items())[:n])

    def top_local_features(self, instance_index: int = 0, n: int = 10) -> List[FeatureContribution]:
        """Return top-n features for a specific instance sorted by |SHAP|."""
        if not self.local_explanations or instance_index >= len(self.local_explanations):
            return []
        contribs = self.local_explanations[instance_index]
        return sorted(contribs, key=lambda x: x.abs_shap, reverse=True)[:n]

    def print_waterfall(self, instance_index: int = 0, n: int = 10) -> None:
        """Print a text-based waterfall chart for an instance."""
        features = self.top_local_features(instance_index, n)
        if not features:
            print("No local explanations available.")
            return

        max_abs = max(f.abs_shap for f in features) or 1.0
        bar_width = 30

        print(f"\n{'='*60}")
        print(f"SHAP Waterfall — Instance {instance_index} | Model: {self.model_id}")
        print(f"Expected value (base): {self.expected_value:.4f}")
        print(f"{'='*60}")

        cumulative = self.expected_value
        for fc in features:
            bar_len = int(abs(fc.shap_value) / max_abs * bar_width)
            bar = ("█" * bar_len) if fc.shap_value > 0 else ("░" * bar_len)
            direction = "+" if fc.shap_value > 0 else "-"
            cumulative += fc.shap_value
            print(f"  {fc.feature:<25} {direction}{abs(fc.shap_value):.4f}  {bar}")

        print(f"{'─'*60}")
        print(f"  {'Prediction':25} {cumulative:.4f}")
        print()

    def plot_waterfall(self, instance_index: int = 0, max_display: int = 12) -> None:
        """Generate matplotlib waterfall plot (requires matplotlib)."""
        try:
            import sys
            sys.path.insert(0, ".")
            from src.explainability.visualisation import waterfall_plot
            import numpy as np

            contribs = self.local_explanations[instance_index] if self.local_explanations else []
            if not contribs:
                print("No local explanations for plotting.")
                return

            feat_names = [c.feature for c in contribs]
            sv = np.array([c.shap_value for c in contribs])
            fv = np.array([c.feature_value for c in contribs])

            fig = waterfall_plot(sv, fv, feat_names, self.expected_value, max_display=max_display)
            fig.show() if hasattr(fig, "show") else None
            return fig
        except ImportError as e:
            print(f"Plotting requires matplotlib: {e}")


@dataclass
class PredictionResponse:
    model_id: str
    predictions: List[Any]
    probabilities: Optional[List[List[float]]]
    n_instances: int


@dataclass
class ModelInfo:
    model_id: str
    name: str
    version: str
    framework: str
    task: str
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _with_retry(
    fn: Any,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    retryable_status: tuple = (429, 500, 502, 503, 504),
) -> Any:
    """Execute a callable with exponential backoff retry."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = fn()
            if response.status_code in retryable_status and attempt < max_retries:
                wait = backoff_factor * (2 ** attempt)
                logger.warning(
                    "HTTP %d — retrying in %.1fs (attempt %d/%d)",
                    response.status_code, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                continue
            return response
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_factor * (2 ** attempt)
                logger.warning("Request failed (%s) — retrying in %.1fs", exc, wait)
                time.sleep(wait)
    raise ConnectionError(f"Request failed after {max_retries} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------

class XAIClient:
    """
    Synchronous Python SDK client for the XAI Platform API.

    Parameters
    ----------
    base_url : API base URL (e.g., "http://localhost:8000")
    api_key : static API key (X-API-Key header)
    jwt_token : Bearer JWT token (alternative to api_key)
    timeout : request timeout in seconds
    max_retries : number of retry attempts on transient failures
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        elif jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        logger.info("XAIClient connected: %s", base_url)

    def explain(
        self,
        model_id: str,
        instances: List[List[float]],
        feature_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
        scope: str = "local",
        include_lime: bool = False,
        top_features: int = 10,
    ) -> ExplanationResponse:
        """
        Generate SHAP explanations for one or more instances.

        Parameters
        ----------
        model_id : registered model identifier
        instances : list of feature value lists
        feature_names : optional feature name override
        explainer_type : "auto" | "tree" | "linear" | "kernel"
        scope : "local" (per-instance) | "global" (dataset-level)
        include_lime : also compute LIME cross-validation
        top_features : number of features to return

        Returns
        -------
        ExplanationResponse
        """
        payload = {
            "model_id": model_id,
            "instances": instances,
            "explainer_type": explainer_type,
            "scope": scope,
            "include_lime": include_lime,
            "top_features": top_features,
        }
        if feature_names:
            payload["feature_names"] = feature_names

        response = _with_retry(
            lambda: self._client.post("/api/v1/explain", json=payload),
            max_retries=self.max_retries,
        )
        response.raise_for_status()
        return self._parse_explanation(response.json())

    def predict(
        self,
        model_id: str,
        instances: List[List[float]],
        return_probabilities: bool = True,
    ) -> PredictionResponse:
        """Run inference against a registered model."""
        response = _with_retry(
            lambda: self._client.post("/api/v1/predict", json={
                "model_id": model_id,
                "instances": instances,
                "return_probabilities": return_probabilities,
            }),
            max_retries=self.max_retries,
        )
        response.raise_for_status()
        data = response.json()
        return PredictionResponse(
            model_id=data["model_id"],
            predictions=data["predictions"],
            probabilities=data.get("probabilities"),
            n_instances=data["n_instances"],
        )

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        response = _with_retry(
            lambda: self._client.get("/api/v1/models"),
            max_retries=self.max_retries,
        )
        response.raise_for_status()
        return [ModelInfo(**m) for m in response.json()]

    def get_model(self, model_id: str) -> ModelInfo:
        """Get metadata for a specific model."""
        response = _with_retry(
            lambda: self._client.get(f"/api/v1/models/{model_id}"),
            max_retries=self.max_retries,
        )
        response.raise_for_status()
        return ModelInfo(**response.json())

    def health(self) -> Dict[str, str]:
        """Check API health."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "XAIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def _parse_explanation(data: Dict) -> ExplanationResponse:
        local_explanations = None
        if data.get("local_explanations"):
            local_explanations = [
                [FeatureContribution(**fc) for fc in instance]
                for instance in data["local_explanations"]
            ]
        return ExplanationResponse(
            model_id=data["model_id"],
            explainer_type=data["explainer_type"],
            scope=data["scope"],
            n_instances=data["n_instances"],
            computation_time_ms=data["computation_time_ms"],
            additivity_check_passed=data["additivity_check_passed"],
            expected_value=data["expected_value"],
            global_importance=data["global_importance"],
            local_explanations=local_explanations,
            raw=data,
        )


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class AsyncXAIClient:
    """
    Async Python SDK client for the XAI Platform API.

    Identical interface to XAIClient but uses httpx.AsyncClient.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        elif jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    async def explain(
        self,
        model_id: str,
        instances: List[List[float]],
        feature_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
        scope: str = "local",
    ) -> ExplanationResponse:
        payload = {
            "model_id": model_id,
            "instances": instances,
            "explainer_type": explainer_type,
            "scope": scope,
        }
        if feature_names:
            payload["feature_names"] = feature_names
        response = await self._client.post("/api/v1/explain", json=payload)
        response.raise_for_status()
        return XAIClient._parse_explanation(response.json())

    async def predict(
        self, model_id: str, instances: List[List[float]]
    ) -> PredictionResponse:
        response = await self._client.post("/api/v1/predict", json={
            "model_id": model_id, "instances": instances, "return_probabilities": True
        })
        response.raise_for_status()
        data = response.json()
        return PredictionResponse(**data)

    async def health(self) -> Dict[str, str]:
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncXAIClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
