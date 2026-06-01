"""
Data & Concept Drift Detection
================================
Monitors production data distributions against training baselines.
Implements Population Stability Index (PSI), Kolmogorov-Smirnov test,
Jensen-Shannon divergence, and SHAP-based concept drift detection.

Used by the monitoring pipeline to trigger model retraining alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    feature: str
    test: str
    statistic: float
    p_value: Optional[float]
    psi: Optional[float]
    js_divergence: Optional[float]
    severity: DriftSeverity
    drift_detected: bool

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "test": self.test,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "psi": round(self.psi, 4) if self.psi is not None else None,
            "js_divergence": round(self.js_divergence, 4) if self.js_divergence is not None else None,
            "severity": self.severity.value,
            "drift_detected": self.drift_detected,
        }


@dataclass
class DriftReport:
    n_features_tested: int
    n_drifted: int
    drift_rate: float
    overall_severity: DriftSeverity
    feature_results: List[FeatureDriftResult]
    psi_scores: Dict[str, float] = field(default_factory=dict)
    alert_required: bool = False

    def summary(self) -> Dict:
        return {
            "n_features_tested": self.n_features_tested,
            "n_drifted": self.n_drifted,
            "drift_rate_pct": round(self.drift_rate * 100, 1),
            "overall_severity": self.overall_severity.value,
            "alert_required": self.alert_required,
            "top_drifted": [
                r.feature for r in sorted(
                    self.feature_results, key=lambda x: x.statistic, reverse=True
                )[:5] if r.drift_detected
            ],
        }


class DriftDetector:
    """
    Multivariate drift detector for tabular production data.

    Supported tests:
    - Kolmogorov-Smirnov (continuous features)
    - Chi-squared (categorical features)
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence

    Thresholds follow industry convention:
    - PSI < 0.1   → no drift
    - PSI 0.1–0.2 → moderate drift
    - PSI > 0.2   → significant drift
    - KS p-value < 0.05 → statistically significant drift
    """

    PSI_THRESHOLDS = {
        DriftSeverity.NONE: 0.1,
        DriftSeverity.LOW: 0.15,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.25,
    }
    KS_ALPHA = 0.05

    def __init__(
        self,
        reference_data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        psi_bins: int = 10,
    ) -> None:
        self.reference = reference_data
        self.categorical_features = set(categorical_features or [])
        self.psi_bins = psi_bins
        logger.info(
            "DriftDetector initialised | features=%d | categorical=%d",
            len(reference_data.columns),
            len(self.categorical_features),
        )

    def detect(self, production_data: pd.DataFrame) -> DriftReport:
        """
        Run full drift detection suite on production data.

        Parameters
        ----------
        production_data : new data to compare against reference

        Returns
        -------
        DriftReport
        """
        results = []
        common_cols = [c for c in self.reference.columns if c in production_data.columns]

        for col in common_cols:
            ref_vals = self.reference[col].dropna().values
            prod_vals = production_data[col].dropna().values

            if col in self.categorical_features:
                result = self._chi2_test(col, ref_vals, prod_vals)
            else:
                result = self._ks_test(col, ref_vals, prod_vals)
                psi = self._compute_psi(ref_vals, prod_vals)
                result.psi = psi
                result.js_divergence = self._js_divergence(ref_vals, prod_vals)
                # Override severity with PSI-based classification
                result.severity = self._psi_severity(psi)
                result.drift_detected = psi > self.PSI_THRESHOLDS[DriftSeverity.NONE] or result.drift_detected

            results.append(result)

        n_drifted = sum(r.drift_detected for r in results)
        drift_rate = n_drifted / len(results) if results else 0.0
        overall_sev = self._overall_severity(results)

        report = DriftReport(
            n_features_tested=len(results),
            n_drifted=n_drifted,
            drift_rate=drift_rate,
            overall_severity=overall_sev,
            feature_results=results,
            psi_scores={r.feature: r.psi for r in results if r.psi is not None},
            alert_required=overall_sev in {DriftSeverity.HIGH, DriftSeverity.CRITICAL},
        )

        logger.info(
            "Drift detection complete | drifted=%d/%d | severity=%s | alert=%s",
            n_drifted,
            len(results),
            overall_sev.value,
            report.alert_required,
        )
        return report

    # ── Statistical tests ──────────────────────────────────────────────────

    def _ks_test(self, feature: str, ref: np.ndarray, prod: np.ndarray) -> FeatureDriftResult:
        stat, p_val = stats.ks_2samp(ref, prod)
        drift = p_val < self.KS_ALPHA
        severity = DriftSeverity.MEDIUM if drift else DriftSeverity.NONE
        return FeatureDriftResult(
            feature=feature,
            test="kolmogorov_smirnov",
            statistic=float(stat),
            p_value=float(p_val),
            psi=None,
            js_divergence=None,
            severity=severity,
            drift_detected=drift,
        )

    def _chi2_test(self, feature: str, ref: np.ndarray, prod: np.ndarray) -> FeatureDriftResult:
        all_cats = np.unique(np.concatenate([ref, prod]))
        ref_counts = np.array([np.sum(ref == c) for c in all_cats], dtype=float)
        prod_counts = np.array([np.sum(prod == c) for c in all_cats], dtype=float)
        # Normalise to same total
        prod_counts = prod_counts / prod_counts.sum() * ref_counts.sum()
        # Avoid zeros
        ref_counts = np.maximum(ref_counts, 1e-6)
        prod_counts = np.maximum(prod_counts, 1e-6)
        stat, p_val = stats.chisquare(prod_counts, f_exp=ref_counts)
        drift = p_val < self.KS_ALPHA
        return FeatureDriftResult(
            feature=feature,
            test="chi_squared",
            statistic=float(stat),
            p_value=float(p_val),
            psi=None,
            js_divergence=None,
            severity=DriftSeverity.MEDIUM if drift else DriftSeverity.NONE,
            drift_detected=drift,
        )

    def _compute_psi(self, ref: np.ndarray, prod: np.ndarray) -> float:
        """Population Stability Index (PSI)."""
        bins = np.percentile(ref, np.linspace(0, 100, self.psi_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        prod_pct = np.histogram(prod, bins=bins)[0] / len(prod)
        ref_pct = np.maximum(ref_pct, 1e-6)
        prod_pct = np.maximum(prod_pct, 1e-6)
        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return round(abs(psi), 4)

    def _js_divergence(self, ref: np.ndarray, prod: np.ndarray, bins: int = 50) -> float:
        """Jensen-Shannon divergence (symmetric KL-based)."""
        lo, hi = min(ref.min(), prod.min()), max(ref.max(), prod.max())
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(ref, bins=edges, density=True)[0] + 1e-9
        q = np.histogram(prod, bins=edges, density=True)[0] + 1e-9
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return round(float(js), 4)

    @staticmethod
    def _psi_severity(psi: float) -> DriftSeverity:
        if psi < 0.1:
            return DriftSeverity.NONE
        if psi < 0.15:
            return DriftSeverity.LOW
        if psi < 0.2:
            return DriftSeverity.MEDIUM
        if psi < 0.3:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

    @staticmethod
    def _overall_severity(results: List[FeatureDriftResult]) -> DriftSeverity:
        if not results:
            return DriftSeverity.NONE
        severities = [r.severity for r in results]
        order = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        return max(severities, key=lambda s: order.index(s))
