"""
LIME Explainability Engine
==========================
Wraps the LIME library for tabular, text, and image explanations.
Used alongside SHAP to cross-validate feature attributions and
provide practitioner confidence in explanation stability.

Reference: Ribeiro et al., "Why Should I Trust You?", KDD 2016.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


@dataclass
class LIMEResult:
    """Container for LIME explanation output."""

    feature_weights: Dict[str, float]     # feature_name → local weight
    intercept: float
    local_pred: float
    score: float                          # local fidelity R²
    computation_time_ms: float
    instance_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_weights": self.feature_weights,
            "intercept": round(self.intercept, 6),
            "local_pred": round(self.local_pred, 6),
            "fidelity_score": round(self.score, 4),
            "computation_time_ms": round(self.computation_time_ms, 2),
        }

    def top_features(self, n: int = 10) -> Dict[str, float]:
        """Return top-n features by absolute weight."""
        return dict(
            sorted(self.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
        )


class LIMEEngine:
    """
    Unified LIME engine for tabular classification and regression.

    Key properties vs SHAP:
    - Locally faithful linear surrogate (not globally consistent)
    - Does NOT satisfy Shapley axioms (missingness, consistency)
    - Faster for single-instance explanations on non-tree models
    - Useful as a cross-validation signal for SHAP attributions

    Usage
    -----
    engine = LIMEEngine(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="classification",
        categorical_features=[4, 5, 6],
    )
    result = engine.explain(model.predict_proba, instance=X_test.iloc[0].values)
    print(result.top_features())
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        mode: str = "classification",
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        discretize_continuous: bool = True,
        kernel_width: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.categorical_features = categorical_features or []
        self.class_names = class_names

        self._explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode=mode,
            categorical_features=self.categorical_features,
            class_names=class_names,
            discretize_continuous=discretize_continuous,
            kernel_width=kernel_width,
            random_state=random_state,
        )
        logger.info(
            "LIMEEngine initialised | mode=%s | features=%d | categorical=%d",
            mode,
            len(feature_names),
            len(self.categorical_features),
        )

    def explain(
        self,
        predict_fn: Callable,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: int = 1,
        label_index: int = 1,
    ) -> LIMEResult:
        """
        Generate LIME explanation for a single instance.

        Parameters
        ----------
        predict_fn : model.predict_proba or model.predict callable
        instance : 1-D array of feature values
        num_features : max features in the local linear model
        num_samples : neighbourhood samples for linear fit
        label_index : class index for classification (ignored for regression)

        Returns
        -------
        LIMEResult
        """
        t0 = time.perf_counter()

        if self.mode == "classification":
            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels,
            )
            weights = dict(explanation.as_list(label=label_index))
            intercept = explanation.intercept[label_index]
            local_pred = explanation.local_pred[label_index]
            score = explanation.score[label_index]
        else:
            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )
            weights = dict(explanation.as_list())
            intercept = explanation.intercept[1] if hasattr(explanation.intercept, "__getitem__") else float(explanation.intercept)
            local_pred = float(explanation.predicted_value)
            score = float(explanation.score)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "LIME explanation | mode=%s | fidelity=%.4f | time_ms=%.1f",
            self.mode,
            score,
            elapsed_ms,
        )

        return LIMEResult(
            feature_weights=weights,
            intercept=float(intercept),
            local_pred=float(local_pred),
            score=float(score),
            computation_time_ms=elapsed_ms,
        )

    def explain_batch(
        self,
        predict_fn: Callable,
        instances: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> List[LIMEResult]:
        """Explain multiple instances sequentially (LIME is CPU-bound per sample)."""
        return [
            self.explain(predict_fn, instance, num_features=num_features, num_samples=num_samples)
            for instance in instances
        ]

    def compare_with_shap(
        self,
        lime_result: LIMEResult,
        shap_importance: Dict[str, float],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare top-N feature rankings between LIME and SHAP.

        Returns a dict with rank correlation and feature overlap percentage.
        """
        lime_top = list(lime_result.top_features(n=top_n).keys())
        shap_top = list(dict(
            sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        ).keys())

        overlap = len(set(lime_top) & set(shap_top))
        overlap_pct = overlap / top_n * 100

        # Spearman rank correlation on shared features
        shared = list(set(lime_top) & set(shap_top))
        if len(shared) >= 2:
            lime_ranks = [lime_top.index(f) for f in shared]
            shap_ranks = [shap_top.index(f) for f in shared]
            n = len(shared)
            d_sq = sum((r1 - r2) ** 2 for r1, r2 in zip(lime_ranks, shap_ranks))
            rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
        else:
            rho = None

        return {
            "lime_top_features": lime_top,
            "shap_top_features": shap_top,
            "overlap_count": overlap,
            "overlap_pct": round(overlap_pct, 1),
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "agreement": "HIGH" if overlap_pct >= 80 else "MEDIUM" if overlap_pct >= 50 else "LOW",
        }
