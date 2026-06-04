# Explainable AI Demos with SHAP

[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![SHAP](https://img.shields.io/badge/SHAP-0.48-FF6B6B?style=flat-square)](https://shap.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.17-0194E2?style=flat-square&logo=mlflow)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-82%25-brightgreen?style=flat-square)](https://codecov.io)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Tech Stack](#2-architecture--tech-stack)
3. [Key Features Deep Dive](#3-key-features-deep-dive)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [Usage Guide & Code Examples](#5-usage-guide--code-examples)
6. [API Reference](#6-api-reference)
7. [Development & Testing](#7-development--testing)
8. [Contributing](#8-contributing)
9. [Roadmap](#9-roadmap)

---

## 1. Project Overview

Most machine learning models are good at making predictions, but they rarely explain *why* they made them. That gap creates real problems. Regulated industries like banking and insurance face legal obligations to justify automated decisions. Data scientists need to trust their own models before shipping them. Business stakeholders need enough transparency to act on a model's output. The black-box problem is not just a technical inconvenience — it is a blocker for deploying high-stakes AI responsibly.

This project is a hands-on, production-grade implementation of **Explainable AI (XAI)** built around [SHAP](https://shap.readthedocs.io) (SHapley Additive exPlanations) and [LIME](https://github.com/marcotcr/lime). It goes well beyond the typical demo notebook. The codebase includes a REST API for serving explanations, a full MLOps training pipeline with experiment tracking and model versioning, drift detection for production monitoring, and Kubernetes-ready deployment configs — all applied to a real bank customer churn prediction problem.

**Who this is for:**

- **ML engineers** building explainability into production systems
- **Data scientists** who want to understand and communicate model behaviour rigorously
- **MLOps practitioners** looking for a reference implementation of a full XAI pipeline
- **Researchers** studying SHAP theory, LIME cross-validation, and counterfactual explanations
- **Students and graduates** building a portfolio that demonstrates real engineering depth

The core dataset is the [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) dataset — 10,000 customer records with demographics, financial data, and a binary churn label. The trained ensemble model achieves **ROC-AUC 0.887**, and every prediction it makes can be explained in plain terms using SHAP attributions.

---

## 2. Architecture & Tech Stack

### System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                     Clients & Consumers                          │
│        Jupyter Notebooks · REST Clients · Grafana UI             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                  FastAPI Application Layer                        │
│   /api/v1/explain   /api/v1/predict   /api/v1/models   /metrics  │
│   JWT Auth · Rate Limiting (Redis ZSET) · GZip · CORS            │
└────┬──────────────────┬────────────────────┬─────────────────────┘
     │                  │                    │
┌────▼──────┐   ┌───────▼──────┐   ┌────────▼────────┐
│ SHAP      │   │ LIME Engine  │   │  Redis Cache    │
│ Engine    │   │ (cross-      │   │  SHA-256 keyed  │
│ (auto-    │   │  validate)   │   │  TTL: 3600s     │
│  select)  │   └──────────────┘   └─────────────────┘
└────┬──────┘
     │
┌────▼──────────────────────────────────────────────────────────────┐
│                     MLflow Model Registry                         │
│     Staging → Production promotion · Versioned artifacts         │
└────┬──────────────────────────────────────────────────────────────┘
     │
┌────▼──────────────────────────────────────────────────────────────┐
│              DVC Training Pipeline (6 Stages)                     │
│  ingest → validate → featurise → train (Optuna) → evaluate → SHAP│
└────┬──────────────────────────────────────────────────────────────┘
     │
┌────▼──────────────────────────────────────────────────────────────┐
│              Monitoring (Prometheus + Grafana)                    │
│   PSI drift · KS tests · API latency · Additivity failures       │
└───────────────────────────────────────────────────────────────────┘
```

### Technology Choices

| Layer | Technology | Why |
|---|---|---|
| **API Framework** | FastAPI 0.115 | Async-native, Pydantic v2 validation, auto OpenAPI docs |
| **ML Core** | scikit-learn 1.7 | Consistent pipeline API, ColumnTransformer for preprocessing |
| **Gradient Boosting** | XGBoost 3.0 + LightGBM 4.6 | Best-in-class tabular performance; XGBoost has native TreeSHAP support |
| **Explainability** | SHAP 0.48 | Only attribution method satisfying all five Shapley axioms |
| **Cross-validation** | LIME 0.2 | Independent linear surrogate to verify SHAP feature rankings |
| **HPO** | Optuna 4.5 | Bayesian TPE sampler converges in ~30 trials vs 500+ for random search |
| **Experiment Tracking** | MLflow 2.17 | Model registry with stage transitions, artifact storage, run comparison |
| **Pipeline Versioning** | DVC 3.59 | Git-like versioning for data and model artifacts |
| **Orchestration** | Airflow + Prefect | Airflow for scheduled retraining; Prefect as Python-native alternative |
| **Cache** | Redis (asyncio) | SHAP values cached by SHA-256 input hash; 177x speedup on cache hits |
| **Monitoring** | Prometheus + Grafana | Custom collectors expose PSI scores, ROC-AUC, and explanation quality |
| **Deployment** | Docker + Kubernetes + Helm | Container-native with HPA, rolling deployments, and Ingress TLS |
| **Infrastructure** | Terraform (AWS) | VPC, EKS, RDS, ElastiCache, S3 fully provisioned as code |

---

## 3. Key Features Deep Dive

### Intelligent SHAP Explainer Selection

The `SHAPEngine` in `src/explainability/shap_engine.py` uses a strategy pattern to select the most efficient explainer at runtime. When `explainer_type="auto"` (the default), the `ExplainerFactory` inspects the model class name and dispatches accordingly:

- **TreeExplainer** for XGBoost, LightGBM, Random Forest, and Gradient Boosting — computes exact Shapley values in O(TLD²) time rather than the exponential brute-force approach
- **LinearExplainer** for Ridge, Lasso, and Logistic Regression — uses the covariance matrix for exact computation
- **GradientExplainer** when a PyTorch or TensorFlow model is detected
- **Model-agnostic Explainer** for everything else, including VotingClassifier ensembles

Every explanation result is validated for the SHAP **additivity property**: `base_value + Σφᵢ ≈ f(x)`. If this check fails, the response still returns but flags `additivity_check_passed: false` so downstream consumers know to treat the result with caution.

SHAP computation runs in a `ThreadPoolExecutor` with 4 workers so it does not block the async event loop. Results are cached in Redis using a SHA-256 hash of the input array as the cache key, giving a 177x speedup on repeated requests for the same instance.

### LIME Cross-Validation

`LIMEEngine` in `src/explainability/lime_engine.py` provides an independent attribution signal. LIME fits a locally faithful linear surrogate around each instance and returns feature weights. The `compare_with_shap()` method computes Spearman rank correlation between SHAP and LIME top-N feature rankings and reports an overlap percentage. An overlap above 80% signals **HIGH** agreement — both methods identify the same root causes. Below 50% indicates **LOW** agreement, which usually means the model has strong nonlinear interactions that LIME's linear surrogate cannot capture.

### Counterfactual Explanations

`src/explainability/counterfactual.py` answers the question: what is the minimum change to this customer's features that would flip the churn prediction? The engine runs a gradient-free random-walk optimisation with configurable `ActionabilityConstraint` objects. For example, you can mark `age` as `increasing_only=True` (a person cannot become younger) or mark `geography` as `mutable=False` (you cannot change where someone lives). Multiple diverse counterfactuals are generated by enforcing a minimum L2 distance between candidates.

### End-to-End Training Pipeline

`src/pipelines/training_pipeline.py` orchestrates the full lifecycle. It wraps Optuna's `create_study()` to run Bayesian hyperparameter search with the TPE sampler, performs stratified k-fold cross-validation inside each trial, then retrains on the full training set with the best parameters. Every run is logged to MLflow — parameters, metrics, the fitted model, and the top SHAP importance values. Passing models are registered to the MLflow model registry and promoted through `None → Staging → Production` stages.

### Drift Detection

`src/monitoring/drift_detector.py` compares production data distributions against the training baseline using three complementary tests:

- **PSI (Population Stability Index)** — industry-standard threshold of 0.10 (no drift), 0.20 (significant drift), 0.25 (critical)
- **Kolmogorov-Smirnov test** — statistical test for continuous feature distributions at α = 0.05
- **Jensen-Shannon divergence** — symmetric, bounded [0,1] measure of distribution similarity

A `DriftReport` aggregates per-feature results into an overall severity (`none / low / medium / high / critical`) and sets `alert_required=True` when severity reaches HIGH, which triggers a Prometheus alert that can page on-call through AlertManager.

### Visualisation Suite

`src/explainability/visualisation.py` generates matplotlib figures — waterfall plots (single-instance attribution), beeswarm plots (dataset-level distribution of SHAP values), feature importance bar charts, heatmaps (instance x feature SHAP matrix), and dependence scatter plots with optional interaction colour-coding. All functions return `plt.Figure` objects so they can be used programmatically in API responses, notebooks, or saved to disk via `save_all_plots()`.

---

## 4. Prerequisites & Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- Docker and Docker Compose (for the full stack)
- Git

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/wittyswayam/Explainable-AI-Demos-with-SHAP.git
cd Explainable-AI-Demos-with-SHAP
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** If you hit build errors on `lightgbm` or `xgboost` on Apple Silicon, install via conda first: `conda install -c conda-forge lightgbm xgboost`, then run pip for the rest.

**4. Configure environment variables**

Create a `.env` file in the project root:

```bash
touch .env
```

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | One of: `development`, `staging`, `production`, `testing` |
| `SECRET_KEY` | *(required)* | JWT signing key — must be at least 32 characters |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string for SHAP value caching |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL |
| `LOG_LEVEL` | `INFO` | One of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `WORKERS` | `4` | Uvicorn worker processes |
| `MAX_BATCH_SIZE` | `512` | Maximum instances per explain request |
| `SHAP_BACKGROUND_SAMPLE_SIZE` | `500` | Background dataset size for SHAP baseline computation |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per rate limit window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit sliding window in seconds |

**5. Start the full stack with Docker Compose**

```bash
cd docker
docker compose up -d
```

| Service | URL |
|---|---|
| API + Swagger UI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

**6. Run the training pipeline**

```bash
dvc repro
```

This executes all six pipeline stages: data ingestion, validation, feature engineering, training, evaluation, and SHAP generation.

**Common setup errors:**

- `Redis connection refused` — make sure Docker is running, or start Redis locally with `redis-server`
- `MLflow connection error` — the MLflow container can take 15-20 seconds to initialise; wait and retry
- `kagglehub authentication error` — set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment, or place `kaggle.json` in `~/.kaggle/`. The pipeline falls back to synthetic data automatically if Kaggle is unavailable.

---

## 5. Usage Guide & Code Examples

### Running the notebooks

The two research notebooks are the fastest way to understand the project:

```bash
jupyter lab
# Open notebooks/classification_shap.ipynb
# — trains a VotingClassifier on churn data, explains with SHAP and LIME
#
# Open notebooks/regression_shap.ipynb
# — XGBoost on California Housing with TreeSHAP and geographic attribution maps
```

### Training a model from the CLI

```bash
# Run with default config
python scripts/train.py --config configs/training.yaml

# Override specific values without editing YAML
python scripts/train.py \
  --config configs/training.yaml \
  --set training.optuna_trials=50 \
  --set model.type=xgboost

# Validate config and data without running training
python scripts/train.py --config configs/training.yaml --dry-run
```

### Calling the explanation API

```bash
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-ensemble-v2",
    "instances": [[619, 0, 1, 42, 2, 0.0, 1, 1, 1, 101348.88]],
    "explainer_type": "auto",
    "scope": "local",
    "top_features": 8
  }'
```

### Using the Python SDK

```python
from sdk.python.xai_client import XAIClient

with XAIClient(base_url="http://localhost:8000", api_key="your-api-key") as client:

    # Check the API is up
    print(client.health())

    # Generate a local explanation
    result = client.explain(
        model_id="churn-ensemble-v2",
        instances=[[619, 0, 1, 42, 2, 0.0, 1, 1, 1, 101348.88]],
        scope="local",
    )

    # Print waterfall chart in the terminal
    result.print_waterfall(instance_index=0)

    # Top 5 features by global importance
    print(result.top_features(n=5))
```

### Using the SHAP engine directly

```python
import asyncio
from src.explainability.shap_engine import SHAPEngine, ExplainerConfig, ExplainerType

engine = SHAPEngine(
    model=fitted_model,
    background_data=X_train,
    config=ExplainerConfig(
        explainer_type=ExplainerType.AUTO,
        background_sample_size=200,
        check_additivity=True,
    ),
)

result = asyncio.run(engine.explain(X_test.iloc[:10]))
print(result.global_feature_importance())
# {'age': 0.1831, 'balance': 0.1412, 'numofproducts': 0.0984, ...}
```

### Generating counterfactuals

```python
from src.explainability.counterfactual import (
    CounterfactualEngine, CounterfactualConfig, ActionabilityConstraint,
)

engine = CounterfactualEngine(
    model=fitted_model,
    training_data=X_train.values,
    feature_names=feature_names,
    config=CounterfactualConfig(
        n_counterfactuals=3,
        constraints=[
            ActionabilityConstraint(feature="age", increasing_only=True),
            ActionabilityConstraint(feature="geography", mutable=False),
        ],
    ),
)

cfs = engine.generate(instance=X_test.values[0], desired_class=0)
for cf in cfs:
    print(cf.summary())
```

---

## 6. API Reference

### `POST /api/v1/explain`

Generate SHAP explanations for one or more instances.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | string | required | Registered model identifier |
| `instances` | `float[][]` | required | 2-D array of feature values |
| `feature_names` | `string[]` | null | Optional feature name override |
| `explainer_type` | string | `"auto"` | `auto` \| `tree` \| `linear` \| `kernel` \| `gradient` |
| `scope` | string | `"local"` | `local` (per-instance) \| `global` (dataset aggregated) |
| `include_lime` | boolean | false | Also compute LIME cross-validation |
| `top_features` | integer | 10 | Number of features to return (1–50) |

**Sample response:**

```json
{
  "model_id": "churn-ensemble-v2",
  "explainer_type": "tree",
  "scope": "local",
  "n_instances": 1,
  "computation_time_ms": 142.3,
  "additivity_check_passed": true,
  "expected_value": 0.2037,
  "global_importance": {
    "age": 0.1831,
    "balance": 0.1412,
    "numofproducts": 0.0984
  },
  "local_explanations": [[
    { "feature": "age", "shap_value": 0.231, "feature_value": 52.0, "abs_shap": 0.231 },
    { "feature": "balance", "shap_value": 0.198, "feature_value": 0.0, "abs_shap": 0.198 }
  ]]
}
```

### Other endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Run inference against a registered production model |
| `POST` | `/api/v1/explain/batch` | Submit async batch explanation job, returns `job_id` |
| `GET` | `/api/v1/models` | List all registered models with evaluation metrics |
| `GET` | `/api/v1/models/{model_id}` | Full metadata for a specific model version |
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check (verifies Redis and MLflow connectivity) |
| `GET` | `/metrics` | Prometheus scrape endpoint |

Full interactive documentation is available at `http://localhost:8000/docs` when the API is running.

---

## 7. Development & Testing

### Running the test suite

```bash
# All tests
pytest

# Unit tests only — no external dependencies needed
pytest tests/unit/ -v

# Integration tests — requires Redis to be running
pytest tests/integration/ -v

# End-to-end pipeline test
pytest tests/e2e/ -v

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Linting and type checking

```bash
# Check for linting issues
ruff check src/ tests/

# Auto-fix formatting
ruff format src/ tests/

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Local development with hot reload

```bash
ENVIRONMENT=development uvicorn src.api.main:app --reload --port 8000
```

### Setting up pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit` and include ruff, mypy, Bandit security scanning, Gitleaks secret detection, and Hadolint for Dockerfile linting.

### Running benchmarks

```bash
python scripts/benchmark_shap.py
# Compares TreeExplainer, LinearExplainer, and KernelExplainer timing
# Saves results to data/benchmarks/shap_benchmark_results.json
```

### Load testing

```bash
# Start the API first, then:
locust -f tests/load/locustfile.py --host http://localhost:8000
# Open http://localhost:8089 for the Locust UI

# Headless mode
locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 2m
```

---

## 8. Contributing

Contributions are welcome. Here is how to get involved without making a mess of the main branch.

### Setup for contributors

```bash
git clone https://github.com/wittyswayam/Explainable-AI-Demos-with-SHAP.git
cd Explainable-AI-Demos-with-SHAP
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Branch naming

| Type | Format | Example |
|---|---|---|
| Feature | `feature/short-description` | `feature/shap-interaction-values` |
| Bug fix | `fix/short-description` | `fix/redis-fallback-keyerror` |
| Docs | `docs/short-description` | `docs/api-endpoint-examples` |
| Refactor | `refactor/short-description` | `refactor/shap-engine-async` |

Always branch from `develop`, not `main`.

### Commit message format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

```bash
# Examples
git commit -m "feat(explainability): add SHAP interaction values endpoint"
git commit -m "fix(cache): handle Redis connection timeout gracefully"
git commit -m "docs(api): add sample responses to explain endpoint"
```

### Pull request process

1. Open a PR against `develop` with a clear description of what changed and why
2. Write tests for any new functionality — keep coverage above 70%
3. Run `pytest && ruff check src/` before pushing; PRs with failing CI are not merged
4. At least one review approval is required before merging
5. Squash commits when merging to keep the history clean

---

## 9. Roadmap

### Near-term

- **SHAP interaction values** — expose second-order `shap_interaction_values` as a new API endpoint, useful for detecting feature pair effects that first-order SHAP misses
- **Explanation confidence intervals** — bootstrap SHAP values to produce 95% confidence intervals per feature, giving a statistical basis for feature selection decisions
- **TypeScript SDK** — browser-compatible client for the explanation API so frontend teams can build explanation UIs without raw fetch calls

### Medium-term

- **NLP explainability** — SHAP for BERT/transformer text classifiers using the `Text` masker, with token-level attributions in a new research notebook
- **Calibration API endpoint** — expose ECE and reliability curves so model consumers can assess prediction confidence before acting on a score
- **Automated model card generation** — fill in and export a model card directly from MLflow run metadata, removing the manual documentation step

### Long-term

- **Distributed SHAP with Ray** — parallelise SHAP computation across a Ray cluster for datasets too large for a single node
- **Streaming explanations** — Kafka consumer pipeline that scores and explains records in real time, writing SHAP values back to a results topic
- **Multi-tenant support** — organisation-level model isolation with scoped API keys and separate MLflow experiment namespaces

### Known issues

- VotingClassifier ONNX export uses only the XGBoost sub-model as an approximation. Exact ensemble export requires model distillation first — this is documented in `src/pipelines/onnx_export.py`.
- Grafana dashboards require manual Prometheus datasource configuration on first login if Docker auto-provisioning is not enabled.
- `dvc repro` requires a valid Kaggle API key or internet access. If neither is available, the ingest stage falls back to a synthetic dataset with the same schema.

---

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

Built by [Samrat](https://github.com/wittyswayam) as a research and portfolio project exploring production-grade explainable AI systems.
