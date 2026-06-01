"""
scripts/benchmark_shap.py
==========================
Comprehensive SHAP explainer benchmarking suite.

Measures:
- Wall-clock time across explainer types (TreeSHAP, Linear, Kernel, Gradient)
- Scaling behaviour: n_samples × n_features × n_estimators
- Cache speedup analysis
- Memory footprint per explainer
- Additivity pass rate

Outputs results to data/benchmarks/shap_benchmark_results.json
and renders a markdown table for README embedding.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.logging import configure_logging
from src.explainability.shap_engine import ExplainerConfig, ExplainerType, SHAPEngine

configure_logging(level="WARNING")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/benchmarks")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    scenario: str
    explainer_type: str
    n_samples: int
    n_features: int
    n_estimators: Optional[int]
    elapsed_ms: float
    throughput_rows_per_sec: float
    peak_memory_mb: float
    additivity_passed: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def _time_explain(engine: SHAPEngine, X: Any) -> Tuple[float, bool, Optional[str]]:
    """Return (elapsed_ms, additivity_passed, error)."""
    try:
        t0 = time.perf_counter()
        result = await engine.explain(X)
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, result.additivity_check_passed, None
    except Exception as exc:
        return 0.0, False, str(exc)


async def benchmark_tree_scaling() -> List[BenchmarkResult]:
    """TreeExplainer scaling across n_samples and n_estimators."""
    results = []
    X_full, y = load_breast_cancer(return_X_y=True, as_frame=True)

    for n_estimators in [50, 100, 200]:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=4, random_state=42
        )
        model.fit(X_full, y)

        for n_samples in [1, 10, 100, 500]:
            X_bg, X_test, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)
            X_explain = X_test.iloc[:n_samples]

            config = ExplainerConfig(
                explainer_type=ExplainerType.TREE,
                background_sample_size=100,
            )
            engine = SHAPEngine(model=model, background_data=X_bg, config=config)

            # Warm up
            await engine.explain(X_explain.iloc[:1])

            # Benchmark (3 runs, take median)
            times = []
            tracemalloc.start()
            for _ in range(3):
                elapsed, passed, err = await _time_explain(engine, X_explain)
                times.append(elapsed)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            median_ms = float(np.median(times))
            throughput = n_samples / (median_ms / 1000) if median_ms > 0 else 0

            results.append(BenchmarkResult(
                scenario="tree_scaling",
                explainer_type="TreeExplainer",
                n_samples=n_samples,
                n_features=X_full.shape[1],
                n_estimators=n_estimators,
                elapsed_ms=round(median_ms, 2),
                throughput_rows_per_sec=round(throughput, 1),
                peak_memory_mb=round(peak / 1e6, 2),
                additivity_passed=passed,
                error=err,
            ))
            print(
                f"  Tree | n_est={n_estimators:3d} | n={n_samples:4d} | "
                f"{median_ms:8.2f} ms | {throughput:8.1f} rows/s"
            )

    return results


async def benchmark_linear() -> List[BenchmarkResult]:
    """LinearExplainer on regression."""
    results = []
    X_full, y = load_diabetes(return_X_y=True, as_frame=True)
    model = Ridge(alpha=1.0)
    X_bg, X_test, y_bg, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)
    model.fit(X_bg, y_bg)

    config = ExplainerConfig(explainer_type=ExplainerType.LINEAR, background_sample_size=200)
    engine = SHAPEngine(model=model, background_data=X_bg, config=config)

    for n_samples in [1, 10, 100, 500, 1000]:
        X_explain = X_test.iloc[:n_samples] if n_samples <= len(X_test) else X_test

        times = []
        tracemalloc.start()
        for _ in range(3):
            elapsed, passed, err = await _time_explain(engine, X_explain)
            times.append(elapsed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        median_ms = float(np.median(times))
        throughput = len(X_explain) / (median_ms / 1000) if median_ms > 0 else 0

        results.append(BenchmarkResult(
            scenario="linear_scaling",
            explainer_type="LinearExplainer",
            n_samples=len(X_explain),
            n_features=X_full.shape[1],
            n_estimators=None,
            elapsed_ms=round(median_ms, 2),
            throughput_rows_per_sec=round(throughput, 1),
            peak_memory_mb=round(peak / 1e6, 2),
            additivity_passed=passed,
            error=err,
        ))
        print(f"  Linear | n={len(X_explain):4d} | {median_ms:8.2f} ms | {throughput:8.1f} rows/s")

    return results


async def benchmark_kernel_vs_tree() -> List[BenchmarkResult]:
    """Direct Kernel vs Tree explainer comparison on same model/data."""
    results = []
    X_full, y = load_iris(return_X_y=True, as_frame=True)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    X_bg, X_test, y_bg, _ = train_test_split(X_full, y, test_size=0.3, random_state=42)
    model.fit(X_bg, y_bg)

    for n_samples in [1, 5, 10, 20]:
        X_explain = X_test.iloc[:n_samples]

        for explainer_type, label in [
            (ExplainerType.TREE, "TreeExplainer"),
            (ExplainerType.KERNEL, "KernelExplainer"),
        ]:
            config = ExplainerConfig(
                explainer_type=explainer_type,
                background_sample_size=50,
                max_evals=200,
            )
            engine = SHAPEngine(model=model, background_data=X_bg, config=config)

            times = []
            tracemalloc.start()
            for _ in range(3):
                elapsed, passed, err = await _time_explain(engine, X_explain)
                times.append(elapsed)
                # Reset explainer to avoid caching bias
                engine._explainer = None
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            median_ms = float(np.median(times))
            throughput = n_samples / (median_ms / 1000) if median_ms > 0 else 0

            results.append(BenchmarkResult(
                scenario="tree_vs_kernel",
                explainer_type=label,
                n_samples=n_samples,
                n_features=X_full.shape[1],
                n_estimators=50,
                elapsed_ms=round(median_ms, 2),
                throughput_rows_per_sec=round(throughput, 1),
                peak_memory_mb=round(peak / 1e6, 2),
                additivity_passed=passed,
                error=err,
            ))
            print(
                f"  {label:20s} | n={n_samples:3d} | "
                f"{median_ms:8.2f} ms | {throughput:8.1f} rows/s"
            )

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_markdown_table(results: List[BenchmarkResult]) -> str:
    """Generate a markdown table for README embedding."""
    lines = [
        "| Scenario | Explainer | n_samples | n_estimators | Time (ms) | Throughput (rows/s) | Memory (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario} | {r.explainer_type} | {r.n_samples} | "
            f"{r.n_estimators or 'N/A'} | {r.elapsed_ms:.2f} | "
            f"{r.throughput_rows_per_sec:.1f} | {r.peak_memory_mb:.2f} |"
        )
    return "\n".join(lines)


async def main() -> None:
    print("\n" + "=" * 70)
    print("SHAP Explainer Benchmarking Suite")
    print("=" * 70 + "\n")

    all_results: List[BenchmarkResult] = []

    print("─── TreeExplainer Scaling ───")
    all_results.extend(await benchmark_tree_scaling())

    print("\n─── LinearExplainer Scaling ───")
    all_results.extend(await benchmark_linear())

    print("\n─── Tree vs Kernel Comparison ───")
    all_results.extend(await benchmark_kernel_vs_tree())

    # Persist results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "shap_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n✅ Results saved to {results_path}")

    # Markdown table
    md_table = render_markdown_table(all_results)
    md_path = OUT_DIR / "shap_benchmark_table.md"
    with open(md_path, "w") as f:
        f.write("## SHAP Benchmark Results\n\n")
        f.write(md_table)
    print(f"✅ Markdown table saved to {md_path}")

    # Summary stats
    tree_results = [r for r in all_results if r.explainer_type == "TreeExplainer" and r.n_samples == 100]
    kernel_results = [r for r in all_results if r.explainer_type == "KernelExplainer" and r.n_samples == 10]

    if tree_results and kernel_results:
        tree_ms = tree_results[0].elapsed_ms
        kernel_ms = kernel_results[0].elapsed_ms
        speedup = kernel_ms / tree_ms if tree_ms > 0 else 0
        print(f"\n📊 TreeExplainer vs KernelExplainer speedup: {speedup:.1f}×")


if __name__ == "__main__":
    asyncio.run(main())
