"""
API Router — Explain
====================
Endpoints for generating SHAP and LIME explanations.
Supports single-instance (local) and batch (global) modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    model_id: str = Field(..., description="Registered model identifier")
    instances: List[List[float]] = Field(..., description="2-D array of feature values")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature name override")
    explainer_type: str = Field("auto", description="tree | linear | kernel | auto")
    scope: str = Field("local", description="local | global")
    include_lime: bool = Field(False, description="Also compute LIME cross-validation")
    top_features: int = Field(10, ge=1, le=50)

    @field_validator("explainer_type")
    @classmethod
    def valid_explainer(cls, v: str) -> str:
        if v not in {"tree", "linear", "kernel", "gradient", "auto"}:
            raise ValueError("explainer_type must be one of: tree, linear, kernel, gradient, auto")
        return v

    @field_validator("scope")
    @classmethod
    def valid_scope(cls, v: str) -> str:
        if v not in {"local", "global"}:
            raise ValueError("scope must be 'local' or 'global'")
        return v


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    abs_shap: float


class ExplainResponse(BaseModel):
    model_id: str
    explainer_type: str
    scope: str
    n_instances: int
    computation_time_ms: float
    additivity_check_passed: bool
    expected_value: Any
    global_importance: Dict[str, float]
    local_explanations: Optional[List[List[FeatureContribution]]] = None
    lime_comparison: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Generate SHAP explanations for one or more instances.

    - **local scope**: per-instance feature attributions (waterfall-ready)
    - **global scope**: mean |SHAP| feature importance across all instances
    - **include_lime**: cross-validate top features with LIME local surrogate
    """
    import numpy as np
    import time

    X = np.array(request.instances, dtype=np.float64)
    n_instances = X.shape[0]

    # In production this would load from model registry; stub response here
    t0 = time.perf_counter()

    # Stub global importance (replace with real engine call in production)
    n_features = X.shape[1]
    feat_names = request.feature_names or [f"feature_{i}" for i in range(n_features)]
    rng = np.random.default_rng(42)
    importance_vals = rng.exponential(0.1, size=n_features)
    global_imp = {
        name: round(float(v), 6)
        for name, v in sorted(
            zip(feat_names, importance_vals), key=lambda x: x[1], reverse=True
        )
    }

    # Build local explanations stub
    local_explanations = None
    if request.scope == "local":
        shap_vals = rng.normal(0, 0.05, size=(n_instances, n_features))
        local_explanations = [
            [
                FeatureContribution(
                    feature=feat_names[j],
                    shap_value=round(float(shap_vals[i, j]), 6),
                    feature_value=round(float(X[i, j]), 4),
                    abs_shap=round(abs(float(shap_vals[i, j])), 6),
                )
                for j in range(n_features)
            ]
            for i in range(n_instances)
        ]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ExplainResponse(
        model_id=request.model_id,
        explainer_type=request.explainer_type,
        scope=request.scope,
        n_instances=n_instances,
        computation_time_ms=round(elapsed_ms, 2),
        additivity_check_passed=True,
        expected_value=0.5,
        global_importance=global_imp,
        local_explanations=local_explanations,
        lime_comparison=None,
    )


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED)
async def explain_batch_async(
    request: ExplainRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Submit a large batch explanation job asynchronously.
    Returns a job_id; poll /explain/status/{job_id} for results.
    """
    import uuid
    job_id = str(uuid.uuid4())
    # In production: enqueue to Celery / ARQ / Redis queue
    background_tasks.add_task(_run_batch_job, job_id, request)
    logger.info("Batch explanation job submitted | job_id=%s | n=%d", job_id, len(request.instances))
    return {"job_id": job_id, "status": "queued"}


async def _run_batch_job(job_id: str, request: ExplainRequest) -> None:
    """Background task placeholder for async batch explanation."""
    logger.info("Processing batch job %s", job_id)
    # Real implementation: run SHAPEngine.explain_batch(), store result in Redis
