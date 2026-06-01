"""
Model Registry
==============
Thin wrapper around MLflow model registry providing:
- Model registration and versioning
- Stage transitions (Staging → Production → Archived)
- Model loading with caching
- Metadata management
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages the full model lifecycle via MLflow.

    Stages: None → Staging → Production → Archived
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._cache: Dict[str, Any] = {}
        logger.info("ModelRegistry connected: %s", tracking_uri)

    def register(
        self,
        run_id: str,
        artifact_path: str,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a logged model and return its version."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        self._client.update_model_version(
            name=name,
            version=mv.version,
            description=description,
        )
        if tags:
            for k, v in tags.items():
                self._client.set_model_version_tag(name, mv.version, k, v)
        logger.info("Registered model '%s' version %s", name, mv.version)
        return mv.version

    def promote(self, name: str, version: str, stage: str = "Production") -> None:
        """Transition a model version to the given stage."""
        self._client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=True
        )
        logger.info("Promoted '%s' v%s → %s", name, version, stage)

    def load(self, name: str, stage: str = "Production") -> Any:
        """Load a model from the registry, with in-process caching."""
        cache_key = f"{name}@{stage}"
        if cache_key in self._cache:
            logger.debug("Model cache HIT: %s", cache_key)
            return self._cache[cache_key]
        model_uri = f"models:/{name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        self._cache[cache_key] = model
        logger.info("Loaded model: %s", cache_key)
        return model

    def list_versions(self, name: str) -> List[Dict[str, str]]:
        """Return all versions for a registered model."""
        versions = self._client.search_model_versions(f"name='{name}'")
        return [
            {
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "created_at": mv.creation_timestamp,
                "description": mv.description or "",
            }
            for mv in versions
        ]

    def get_latest_production(self, name: str) -> Optional[Dict[str, str]]:
        """Return the latest Production-stage version metadata."""
        versions = self._client.get_latest_versions(name, stages=["Production"])
        if not versions:
            return None
        mv = versions[0]
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "model_uri": f"models:/{name}/Production",
        }
