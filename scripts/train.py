"""
train.py — End-to-End Training CLI
====================================
Orchestrates data loading, feature engineering, Optuna HPO,
MLflow tracking, and SHAP post-training explainability.

Usage:
    python scripts/train.py --config configs/training.yaml
    python scripts/train.py --config configs/training.yaml --set training.optuna_trials=50
    python scripts/train.py --config configs/training.yaml --dry-run
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import click
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import configure_logging
from src.pipelines.training_pipeline import TrainingConfig, TrainingPipeline

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: tuple[str, ...]) -> Dict[str, Any]:
    """Load YAML config and apply CLI --set overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for override in overrides:
        if "=" not in override:
            raise click.BadParameter(f"--set must be key=value, got: {override}")
        key_path, value = override.split("=", 1)
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to cast to numeric
        try:
            node[keys[-1]] = int(value)
        except ValueError:
            try:
                node[keys[-1]] = float(value)
            except ValueError:
                node[keys[-1]] = value

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_churn_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load Bank Customer Churn dataset via kagglehub or local cache."""
    local_path = Path("data/raw/churn_dataset.csv")

    if local_path.exists():
        logger.info("Loading from local cache: %s", local_path)
        return pd.read_csv(local_path)

    try:
        import kagglehub
        path = kagglehub.dataset_download(cfg["dataset"]["kaggle_path"])
        csv_files = list(Path(path).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV found in kagglehub download")
        df = pd.read_csv(csv_files[0])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        logger.info("Downloaded and cached: %s", local_path)
        return df
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise


def preprocess_churn(df: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and prepare the churn dataset."""
    feat_cfg = cfg.get("features", {})
    target = feat_cfg.get("target_col", "exited")
    drop_cols = feat_cfg.get("drop_cols", [])

    # Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop identifiers
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in DataFrame. Columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(cfg: Dict[str, Any]) -> ColumnTransformer:
    """Build sklearn ColumnTransformer from config."""
    feat_cfg = cfg.get("features", {})
    num_cols = feat_cfg.get("numerical_cols", [])
    cat_cols = feat_cfg.get("categorical_cols", [])
    pre_cfg = cfg.get("preprocessing", {})

    from sklearn import set_config
    set_config(transform_output="pandas")

    return ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(
                handle_unknown=pre_cfg.get("handle_unknown", "ignore"),
                sparse_output=False,
            ), cat_cols),
        ],
        remainder="passthrough",
    )


# ---------------------------------------------------------------------------
# Model builders (Optuna-compatible)
# ---------------------------------------------------------------------------

def churn_model_builder_optuna(trial: Any) -> Pipeline:
    """Optuna objective: suggest hyperparameters and return unfitted pipeline."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int("rf__n_estimators", 50, 300, step=50),
        max_depth=trial.suggest_int("rf__max_depth", 3, 10),
        min_samples_leaf=trial.suggest_int("rf__min_samples_leaf", 1, 10),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=trial.suggest_int("xgb__n_estimators", 50, 300, step=50),
        max_depth=trial.suggest_int("xgb__max_depth", 3, 8),
        learning_rate=trial.suggest_float("xgb__learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("xgb__subsample", 0.6, 1.0),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm__n_estimators", 50, 300, step=50),
        num_leaves=trial.suggest_int("lgbm__num_leaves", 20, 100),
        learning_rate=trial.suggest_float("lgbm__learning_rate", 0.01, 0.3, log=True),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        n_jobs=-1,
    )
    return voting


def churn_model_builder_fixed(params: Dict[str, Any]) -> Any:
    """Build a VotingClassifier from explicit best params dict."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    rf = RandomForestClassifier(
        n_estimators=params.get("rf__n_estimators", 100),
        max_depth=params.get("rf__max_depth", 6),
        min_samples_leaf=params.get("rf__min_samples_leaf", 2),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=params.get("xgb__n_estimators", 100),
        max_depth=params.get("xgb__max_depth", 5),
        learning_rate=params.get("xgb__learning_rate", 0.1),
        subsample=params.get("xgb__subsample", 0.8),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=params.get("lgbm__n_estimators", 100),
        num_leaves=params.get("lgbm__num_leaves", 50),
        learning_rate=params.get("lgbm__learning_rate", 0.1),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def display_results(result: Any) -> None:
    """Render training results as rich console tables."""
    console.print("\n[bold green]✅ Training Complete[/bold green]\n")

    # Metrics table
    table = Table(title="📊 Model Performance", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")

    all_metrics = set(result.train_metrics) | set(result.val_metrics)
    for metric in sorted(all_metrics):
        train_v = f"{result.train_metrics.get(metric, 'N/A'):.4f}"
        val_v = f"{result.val_metrics.get(metric, 'N/A'):.4f}"
        table.add_row(metric.upper(), train_v, val_v)

    console.print(table)

    # SHAP importance
    if result.shap_importance:
        simp_table = Table(title="🧠 SHAP Global Importance (Top 10)", header_style="bold magenta")
        simp_table.add_column("Feature", style="dim")
        simp_table.add_column("Mean |SHAP|", justify="right")

        for feat, val in list(result.shap_importance.items())[:10]:
            simp_table.add_row(feat, f"{val:.6f}")

        console.print(simp_table)

    # Summary
    console.print(f"\n[bold]MLflow Run ID:[/bold] {result.mlflow_run_id}")
    console.print(f"[bold]CV Score:[/bold] {result.cv_score:.4f}")
    console.print(f"[bold]Training Time:[/bold] {result.training_time_seconds:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "-c", default="configs/training.yaml", help="Path to training config YAML")
@click.option("--data-config", default="configs/data.yaml", help="Path to data config YAML")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE", help="Override config values")
@click.option("--dry-run", is_flag=True, help="Validate config and data; skip training")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    config: str,
    data_config: str,
    overrides: tuple,
    dry_run: bool,
    log_level: str,
) -> None:
    """Train and register a model with full MLflow tracking and SHAP explainability."""
    configure_logging(level=log_level)

    # Load configs
    train_cfg = load_config(config, overrides)
    data_cfg = load_config(data_config, ())

    console.print(f"\n[bold blue]🚀 XAI Platform Training Pipeline[/bold blue]")
    console.print(f"Config: {config} | Task: {train_cfg['experiment']['name']}\n")

    # Load data
    with console.status("[bold]Loading dataset..."):
        df = load_churn_data(data_cfg)
        X, y = preprocess_churn(df, data_cfg)

    console.print(f"✅ Dataset: {X.shape[0]:,} rows × {X.shape[1]} features | Target: {y.mean():.2%} positive\n")

    if dry_run:
        console.print("[yellow]Dry run — skipping training[/yellow]")
        return

    # Build preprocessor and split
    preprocessor = build_preprocessor(data_cfg)
    X_processed = preprocessor.fit_transform(X)
    feature_names = list(X_processed.columns) if hasattr(X_processed, "columns") else [f"f{i}" for i in range(X_processed.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y.values,
        test_size=train_cfg["training"]["test_size"],
        stratify=y,
        random_state=train_cfg["training"]["random_state"],
    )

    console.print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Build TrainingConfig
    pipeline_config = TrainingConfig(
        experiment_name=train_cfg["experiment"]["name"],
        task=train_cfg["training"]["task"],
        cv_folds=train_cfg["training"]["cv_folds"],
        test_size=train_cfg["training"]["test_size"],
        random_state=train_cfg["training"]["random_state"],
        optuna_trials=train_cfg["training"]["optuna_trials"],
        optuna_timeout_seconds=train_cfg["training"]["optuna_timeout_seconds"],
        metric=train_cfg["training"]["metric"],
        mlflow_tracking_uri=train_cfg["mlflow"]["tracking_uri"],
        model_registry_name=train_cfg["mlflow"].get("model_registry_name"),
        shap_sample_size=train_cfg["shap"]["background_sample_size"],
    )

    # Model builder adapter (Optuna trial OR fixed params)
    def model_builder(trial: Any = None, params: Optional[Dict] = None) -> Any:
        if trial is not None:
            return churn_model_builder_optuna(trial)
        return churn_model_builder_fixed(params or {})

    pipeline = TrainingPipeline(config=pipeline_config)

    with console.status("[bold]Training... (this may take several minutes)[/bold]"):
        result = pipeline.run(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            model_builder=model_builder,
            feature_names=feature_names,
        )

    display_results(result)


if __name__ == "__main__":
    main()
