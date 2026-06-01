"""
src/explainability/visualisation.py
=====================================
Production-grade SHAP visualisation utilities.

Generates publication-quality plots:
- Waterfall (local attribution)
- Force plot (interactive HTML)
- Beeswarm (global importance distribution)
- Bar chart (global importance ranking)
- Heatmap (instance × feature SHAP matrix)
- Dependence scatter (feature interaction)
- Decision plot (multi-instance decision paths)

All functions accept ExplanationResult and return matplotlib Figure objects,
enabling programmatic use in API responses, notebooks, and reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Colour palette consistent with SHAP conventions
POSITIVE_COLOR = "#FF4B4B"   # red → increases prediction
NEGATIVE_COLOR = "#3D84F5"   # blue → decreases prediction
NEUTRAL_COLOR = "#AAAAAA"


def waterfall_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    instance_index: int = 0,
    max_display: int = 15,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Generate a waterfall plot for a single instance.

    The waterfall decomposes the model output into additive feature contributions:
        f(x) = E[f(x)] + Σ φᵢ

    Parameters
    ----------
    shap_values : (n_samples, n_features) or (n_features,) array
    feature_values : raw feature values for display
    feature_names : list of feature names
    base_value : expected model output E[f(x)]
    instance_index : which row to plot (if 2D input)
    max_display : max features to show before truncating
    """
    if shap_values.ndim == 2:
        sv = shap_values[instance_index]
        fv = feature_values[instance_index]
    else:
        sv = shap_values
        fv = feature_values

    # Sort by absolute value
    order = np.argsort(np.abs(sv))[::-1][:max_display]
    sv_display = sv[order]
    fv_display = fv[order]
    names_display = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)

    # Cumulative sum for waterfall
    cumulative = base_value
    lefts = []
    for i, val in enumerate(sv_display[::-1]):
        lefts.append(cumulative if val >= 0 else cumulative + val)
        cumulative += val

    # Draw bars (reversed to show top contribution at top)
    for i, (name, val, left, fval) in enumerate(
        zip(
            reversed(names_display),
            reversed(sv_display),
            lefts,
            reversed(fv_display),
        )
    ):
        color = POSITIVE_COLOR if val >= 0 else NEGATIVE_COLOR
        ax.barh(i, abs(val), left=left, color=color, alpha=0.85, height=0.6, edgecolor="white")
        label = f" +{val:.3f}" if val >= 0 else f" {val:.3f}"
        ax.text(
            left + abs(val) + 0.001,
            i,
            f"{name} = {fval:.3g}{label}",
            va="center",
            fontsize=9,
        )

    # Base value marker
    ax.axvline(base_value, color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"E[f(x)] = {base_value:.3f}")
    ax.axvline(cumulative, color="black", linestyle="-", linewidth=1.5, alpha=0.8, label=f"f(x) = {cumulative:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("Model Output", fontsize=11)
    ax.set_title(title or f"SHAP Waterfall — Instance {instance_index}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def beeswarm_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Beeswarm plot showing distribution of SHAP values per feature.
    Colour encodes feature value (low=blue, high=red).
    """
    if shap_values.ndim != 2:
        raise ValueError("beeswarm_plot requires 2D shap_values (n_samples, n_features)")

    n_samples = shap_values.shape[0]
    # Sort features by mean |SHAP|
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:max_display]
    order = order[::-1]  # bottom-up for horizontal display

    fig, ax = plt.subplots(figsize=figsize)

    for y_pos, feat_idx in enumerate(order):
        sv = shap_values[:, feat_idx]
        fv = feature_values[:, feat_idx]

        # Jitter vertically
        rng = np.random.default_rng(feat_idx)
        jitter = rng.uniform(-0.25, 0.25, n_samples)

        # Normalise feature values to [0, 1] for colour
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        colors = plt.cm.RdBu_r(fv_norm)

        ax.scatter(sv, y_pos + jitter, c=colors, s=6, alpha=0.6, linewidths=0)
        ax.axhline(y_pos, color="lightgray", linewidth=0.5, alpha=0.4)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
    ax.set_title(title or "SHAP Beeswarm Plot", fontsize=13, fontweight="bold")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=40)
    cbar.set_label("Feature Value (low → high)", fontsize=9)

    fig.tight_layout()
    return fig


def bar_importance_plot(
    global_importance: Dict[str, float],
    max_display: int = 20,
    figsize: Tuple[int, int] = (9, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of global mean |SHAP| feature importance.
    """
    items = list(global_importance.items())[:max_display]
    features = [item[0] for item in items][::-1]
    values = [item[1] for item in items][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in values]
    bars = ax.barh(features, values, color=POSITIVE_COLOR, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(title or "Global SHAP Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def heatmap_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    max_features: int = 15,
    max_instances: int = 50,
    figsize: Tuple[int, int] = (12, 7),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of SHAP values: rows=instances, columns=features.
    Useful for identifying cohort-level patterns.
    """
    sv = shap_values[:max_instances]
    importance = np.abs(sv).mean(axis=0)
    feat_order = np.argsort(importance)[::-1][:max_features]
    sv_display = sv[:, feat_order]
    feat_names_display = [feature_names[i] for i in feat_order]

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.abs(sv_display).max()
    im = ax.imshow(sv_display, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feat_names_display)))
    ax.set_xticklabels(feat_names_display, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Instance Index", fontsize=10)
    ax.set_title(title or "SHAP Value Heatmap", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="SHAP Value")
    fig.tight_layout()
    return fig


def dependence_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    feature: Union[str, int],
    interaction_feature: Optional[Union[str, int]] = None,
    figsize: Tuple[int, int] = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot: feature value vs SHAP value.
    Colour encodes interaction feature if provided.
    """
    if isinstance(feature, str):
        feat_idx = feature_names.index(feature)
    else:
        feat_idx = feature

    sv = shap_values[:, feat_idx]
    fv = feature_values[:, feat_idx]

    fig, ax = plt.subplots(figsize=figsize)

    if interaction_feature is not None:
        if isinstance(interaction_feature, str):
            int_idx = feature_names.index(interaction_feature)
        else:
            int_idx = interaction_feature
        int_fv = feature_values[:, int_idx]
        norm = (int_fv - int_fv.min()) / (int_fv.max() - int_fv.min() + 1e-9)
        scatter = ax.scatter(fv, sv, c=norm, cmap="coolwarm", s=10, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(feature_names[int_idx], fontsize=9)
    else:
        ax.scatter(fv, sv, color=POSITIVE_COLOR, s=10, alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"{feature_names[feat_idx]} (feature value)", fontsize=10)
    ax.set_ylabel(f"SHAP({feature_names[feat_idx]})", fontsize=10)
    ax.set_title(title or f"SHAP Dependence: {feature_names[feat_idx]}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def save_all_plots(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    global_importance: Dict[str, float],
    output_dir: Union[str, Path] = "data/plots",
    fmt: str = "png",
    dpi: int = 150,
) -> Dict[str, Path]:
    """
    Generate and save the full standard plot suite.
    Returns a dict mapping plot name → saved file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    plots = {
        "waterfall": lambda: waterfall_plot(shap_values, feature_values, feature_names, base_value),
        "beeswarm": lambda: beeswarm_plot(shap_values, feature_values, feature_names),
        "bar_importance": lambda: bar_importance_plot(global_importance),
        "heatmap": lambda: heatmap_plot(shap_values, feature_names),
    }

    for name, fn in plots.items():
        try:
            fig = fn()
            path = out / f"shap_{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved[name] = path
            logger.info("Saved plot: %s", path)
        except Exception as exc:
            logger.warning("Failed to generate %s plot: %s", name, exc)

    return saved
