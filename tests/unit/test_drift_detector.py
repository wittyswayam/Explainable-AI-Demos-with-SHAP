"""Unit Tests — Drift Detector"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, DriftSeverity


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.normal(40, 10, 1000),
        "income": rng.exponential(50000, 1000),
        "score": rng.uniform(300, 850, 1000),
        "gender": rng.choice(["M", "F"], 1000),
    })


@pytest.fixture
def no_drift_df(reference_df):
    """Production data from same distribution."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age": rng.normal(40, 10, 500),
        "income": rng.exponential(50000, 500),
        "score": rng.uniform(300, 850, 500),
        "gender": rng.choice(["M", "F"], 500),
    })


@pytest.fixture
def high_drift_df():
    """Production data from very different distribution."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "age": rng.normal(70, 5, 500),        # shifted mean
        "income": rng.normal(200000, 10000, 500),  # totally different
        "score": rng.uniform(100, 300, 500),   # lower range
        "gender": rng.choice(["M", "F"], 500),
    })


class TestDriftDetector:
    def test_no_drift_detected_on_same_distribution(self, reference_df, no_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(no_drift_df)
        assert report.overall_severity in {DriftSeverity.NONE, DriftSeverity.LOW}

    def test_high_drift_detected_on_shifted_distribution(self, reference_df, high_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(high_drift_df)
        assert report.n_drifted > 0
        assert report.overall_severity in {DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL}

    def test_report_structure(self, reference_df, no_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(no_drift_df)
        assert report.n_features_tested == 4
        assert 0.0 <= report.drift_rate <= 1.0
        assert len(report.feature_results) == 4

    def test_psi_computed_for_continuous(self, reference_df, high_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(high_drift_df)
        continuous = [r for r in report.feature_results if r.feature != "gender"]
        assert all(r.psi is not None for r in continuous)

    def test_chi2_used_for_categorical(self, reference_df, no_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(no_drift_df)
        cat_results = [r for r in report.feature_results if r.feature == "gender"]
        assert cat_results[0].test == "chi_squared"

    def test_ks_used_for_continuous(self, reference_df, no_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(no_drift_df)
        cont_results = [r for r in report.feature_results if r.feature != "gender"]
        assert all(r.test == "kolmogorov_smirnov" for r in cont_results)

    def test_alert_required_on_high_severity(self, reference_df, high_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(high_drift_df)
        if report.overall_severity in {DriftSeverity.HIGH, DriftSeverity.CRITICAL}:
            assert report.alert_required

    def test_summary_dict_has_expected_keys(self, reference_df, no_drift_df):
        detector = DriftDetector(reference_df, categorical_features=["gender"])
        report = detector.detect(no_drift_df)
        summary = report.summary()
        for key in ("n_features_tested", "n_drifted", "drift_rate_pct", "overall_severity", "alert_required"):
            assert key in summary

    def test_psi_severity_thresholds(self):
        assert DriftDetector._psi_severity(0.05) == DriftSeverity.NONE
        assert DriftDetector._psi_severity(0.12) == DriftSeverity.LOW
        assert DriftDetector._psi_severity(0.18) == DriftSeverity.MEDIUM
        assert DriftDetector._psi_severity(0.22) == DriftSeverity.HIGH
        assert DriftDetector._psi_severity(0.50) == DriftSeverity.CRITICAL
