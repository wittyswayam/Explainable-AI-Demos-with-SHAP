"""
src/explainability/counterfactual.py
======================================
Counterfactual Explanation Engine.

Answers: "What is the minimum change to input features that would
flip the model's prediction to the desired outcome?"

Algorithm: Gradient-free optimisation via random restarts + projection
onto valid feature ranges. Supports actionability constraints
(e.g., "age cannot decrease", "geography cannot change").

Reference: Mothilal et al., "Explaining Machine Learning Classifiers
through Diverse Counterfactual Explanations", FAT* 2020.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ActionabilityConstraint:
    """Defines which features can be changed and how."""
    feature: str
    mutable: bool = True
    increasing_only: bool = False    # e.g., age, work experience
    decreasing_only: bool = False    # rarely used
    categorical_values: Optional[List[Any]] = None  # allowed values for cat features


@dataclass
class Counterfactual:
    """A single counterfactual explanation."""
    original_instance: np.ndarray
    counterfactual_instance: np.ndarray
    feature_names: List[str]
    original_prediction: float
    counterfactual_prediction: float
    desired_class: int
    distance: float                  # L2 normalised distance
    changed_features: Dict[str, Tuple[float, float]]  # feature → (orig, cf)
    feasibility_score: float         # 0–1, lower = more actionable
    proximity_score: float           # 0–1, lower distance = higher proximity

    def to_dict(self) -> Dict:
        return {
            "original_prediction": round(self.original_prediction, 4),
            "counterfactual_prediction": round(self.counterfactual_prediction, 4),
            "desired_class": self.desired_class,
            "distance": round(self.distance, 4),
            "n_changes": len(self.changed_features),
            "changed_features": {
                k: {"original": round(v[0], 4), "counterfactual": round(v[1], 4), "delta": round(v[1] - v[0], 4)}
                for k, v in self.changed_features.items()
            },
            "feasibility_score": round(self.feasibility_score, 4),
            "proximity_score": round(self.proximity_score, 4),
        }

    def summary(self) -> str:
        lines = [
            f"Counterfactual Explanation",
            f"  Original prediction:       {self.original_prediction:.3f}",
            f"  Counterfactual prediction: {self.counterfactual_prediction:.3f}",
            f"  Distance (L2 normalised):  {self.distance:.4f}",
            f"  Features changed ({len(self.changed_features)}):",
        ]
        for feat, (orig, cf) in sorted(
            self.changed_features.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True
        ):
            direction = "↑" if cf > orig else "↓"
            lines.append(f"    {direction} {feat}: {orig:.3g} → {cf:.3g}")
        return "\n".join(lines)


@dataclass
class CounterfactualConfig:
    n_counterfactuals: int = 3           # number of diverse CFs to generate
    max_iterations: int = 1000
    step_size: float = 0.01
    proximity_weight: float = 0.5        # weight for distance penalty
    diversity_weight: float = 0.3        # weight for diversity among CFs
    feasibility_weight: float = 0.2      # weight for actionability
    tolerance: float = 0.5              # desired class probability threshold
    random_state: int = 42
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = field(default=None)
    constraints: List[ActionabilityConstraint] = field(default_factory=list)


class CounterfactualEngine:
    """
    Generates actionable counterfactual explanations for tabular classifiers.

    Usage
    -----
    engine = CounterfactualEngine(
        model=model,
        training_data=X_train,
        feature_names=feature_names,
        config=CounterfactualConfig(n_counterfactuals=3),
    )
    cfs = engine.generate(instance=X_test.iloc[0].values, desired_class=0)
    for cf in cfs:
        print(cf.summary())
    """

    def __init__(
        self,
        model: Any,
        training_data: np.ndarray,
        feature_names: List[str],
        config: Optional[CounterfactualConfig] = None,
    ) -> None:
        self.model = model
        self.training_data = training_data if isinstance(training_data, np.ndarray) else training_data.values
        self.feature_names = feature_names
        self.config = config or CounterfactualConfig()

        # Compute feature ranges from training data
        self._feat_min = self.training_data.min(axis=0)
        self._feat_max = self.training_data.max(axis=0)
        self._feat_range = self._feat_max - self._feat_min + 1e-9

        # Build constraint maps
        self._mutable: np.ndarray = np.ones(len(feature_names), dtype=bool)
        self._increasing_only: np.ndarray = np.zeros(len(feature_names), dtype=bool)
        for constraint in self.config.constraints:
            if constraint.feature in feature_names:
                idx = feature_names.index(constraint.feature)
                self._mutable[idx] = constraint.mutable
                self._increasing_only[idx] = constraint.increasing_only

        logger.info(
            "CounterfactualEngine initialised | features=%d | mutable=%d | n_cf=%d",
            len(feature_names),
            int(self._mutable.sum()),
            config.n_counterfactuals if config else 3,
        )

    def generate(
        self,
        instance: np.ndarray,
        desired_class: int = 0,
        predict_fn: Optional[Callable] = None,
    ) -> List[Counterfactual]:
        """
        Generate diverse counterfactual explanations.

        Parameters
        ----------
        instance : 1-D feature array
        desired_class : target class (typically the opposite of current prediction)
        predict_fn : optional custom predict_proba function

        Returns
        -------
        List[Counterfactual] sorted by distance (closest first)
        """
        t0 = time.perf_counter()
        predict = predict_fn or self._get_predict_fn()

        orig_prob = predict(instance.reshape(1, -1))[0]
        if hasattr(orig_prob, "__len__"):
            orig_pred = float(orig_prob[1]) if len(orig_prob) > 1 else float(orig_prob[0])
        else:
            orig_pred = float(orig_prob)

        rng = np.random.default_rng(self.config.random_state)
        counterfactuals = []
        generated_cfs = []

        for trial in range(self.config.n_counterfactuals * 10):
            if len(counterfactuals) >= self.config.n_counterfactuals:
                break

            # Initialise from training data point or random perturbation
            if trial % 3 == 0:
                idx = rng.integers(0, len(self.training_data))
                candidate = self.training_data[idx].copy().astype(float)
            else:
                candidate = instance.copy().astype(float) + rng.normal(0, 0.1, size=len(instance))

            # Optimise via gradient-free perturbation
            candidate = self._optimise(candidate, instance, desired_class, predict, rng)

            if candidate is None:
                continue

            # Check desired outcome
            cf_prob = predict(candidate.reshape(1, -1))[0]
            if hasattr(cf_prob, "__len__"):
                cf_pred = float(cf_prob[desired_class]) if len(cf_prob) > desired_class else float(cf_prob[0])
            else:
                cf_pred = float(cf_prob)

            if cf_pred < self.config.tolerance:
                continue

            # Check diversity vs already-found CFs
            if generated_cfs and self._too_similar(candidate, generated_cfs):
                continue

            generated_cfs.append(candidate)
            cf = self._build_counterfactual(instance, candidate, orig_pred, cf_pred, desired_class)
            counterfactuals.append(cf)

        counterfactuals.sort(key=lambda x: x.distance)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Generated %d/%d counterfactuals | desired_class=%d | time_ms=%.1f",
            len(counterfactuals),
            self.config.n_counterfactuals,
            desired_class,
            elapsed,
        )
        return counterfactuals

    def _optimise(
        self,
        candidate: np.ndarray,
        original: np.ndarray,
        desired_class: int,
        predict_fn: Callable,
        rng: np.random.Generator,
        n_steps: int = 200,
    ) -> Optional[np.ndarray]:
        """Random walk with projection onto feasible region."""
        best = candidate.copy()
        best_prob = self._desired_prob(predict_fn, best, desired_class)

        for _ in range(n_steps):
            perturbation = rng.normal(0, self.config.step_size, size=len(candidate))
            perturbation[~self._mutable] = 0.0

            trial = best + perturbation

            # Apply actionability constraints
            mask_inc = self._increasing_only
            trial[mask_inc] = np.maximum(trial[mask_inc], original[mask_inc])

            # Clip to feature ranges
            trial = np.clip(trial, self._feat_min, self._feat_max)

            prob = self._desired_prob(predict_fn, trial, desired_class)

            # Accept if improves desired class probability
            if prob > best_prob:
                best = trial
                best_prob = prob

            if best_prob >= self.config.tolerance:
                return best

        return best if best_prob >= self.config.tolerance * 0.8 else None

    def _desired_prob(self, predict_fn: Callable, x: np.ndarray, desired_class: int) -> float:
        prob = predict_fn(x.reshape(1, -1))[0]
        if hasattr(prob, "__len__"):
            return float(prob[desired_class]) if len(prob) > desired_class else float(prob[0])
        return float(prob) if desired_class == 1 else 1.0 - float(prob)

    def _too_similar(self, candidate: np.ndarray, existing: List[np.ndarray], threshold: float = 0.05) -> bool:
        for ex in existing:
            dist = np.linalg.norm((candidate - ex) / self._feat_range)
            if dist < threshold:
                return True
        return False

    def _build_counterfactual(
        self,
        original: np.ndarray,
        cf: np.ndarray,
        orig_pred: float,
        cf_pred: float,
        desired_class: int,
    ) -> Counterfactual:
        # Normalised L2 distance
        dist = float(np.linalg.norm((cf - original) / self._feat_range))

        # Changed features
        changed = {}
        for i, name in enumerate(self.feature_names):
            if abs(cf[i] - original[i]) > 1e-6:
                changed[name] = (float(original[i]), float(cf[i]))

        # Feasibility: proportion of immutable features changed (should be 0)
        immutable_changed = sum(
            1 for i, name in enumerate(self.feature_names)
            if not self._mutable[i] and abs(cf[i] - original[i]) > 1e-6
        )
        feasibility = 1.0 - immutable_changed / max(1, len(self.feature_names))

        return Counterfactual(
            original_instance=original,
            counterfactual_instance=cf,
            feature_names=self.feature_names,
            original_prediction=orig_pred,
            counterfactual_prediction=cf_pred,
            desired_class=desired_class,
            distance=dist,
            changed_features=changed,
            feasibility_score=feasibility,
            proximity_score=1.0 - min(dist, 1.0),
        )

    def _get_predict_fn(self) -> Callable:
        if hasattr(self.model, "predict_proba"):
            return lambda X: self.model.predict_proba(X)
        return lambda X: self.model.predict(X).reshape(-1, 1)
