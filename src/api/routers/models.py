"""API Router — Models Registry"""
from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ModelInfo(BaseModel):
    model_id: str
    name: str
    version: str
    framework: str
    task: str
    metrics: Dict[str, float]
    registered_at: str


@router.get("", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    """List all registered models in the MLflow model registry."""
    return [
        ModelInfo(
            model_id="churn-ensemble-v2",
            name="Bank Churn VotingClassifier",
            version="2.1.0",
            framework="sklearn+xgboost+lightgbm",
            task="binary_classification",
            metrics={"roc_auc": 0.887, "f1": 0.762, "precision": 0.811, "recall": 0.719},
            registered_at="2024-01-15T10:30:00Z",
        ),
        ModelInfo(
            model_id="housing-xgb-v3",
            name="California Housing XGBRegressor",
            version="3.0.0",
            framework="xgboost",
            task="regression",
            metrics={"rmse": 0.421, "r2": 0.843, "mae": 0.305},
            registered_at="2024-01-20T14:00:00Z",
        ),
    ]


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get metadata for a specific registered model."""
    models = {m.model_id: m for m in await list_models()}
    if model_id not in models:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return models[model_id]
