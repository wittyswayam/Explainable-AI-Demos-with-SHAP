# Changelog

All notable changes to the Explainable AI Demos with SHAP platform are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2024-01-15

### Added — Platform Architecture
- **FastAPI REST API** (`src/api/main.py`) with production-grade middleware stack
  - GZip compression, CORS, structured exception handling
  - Prometheus metrics at `/metrics` (request count, latency histograms, explanation counters)
  - JWT + API Key dual authentication (`src/api/auth.py`)
  - Redis sliding window rate limiter (`src/api/rate_limit.py`)
- **Async SHAP Engine** (`src/explainability/shap_engine.py`)
  - Auto-selects TreeExplainer, LinearExplainer, GradientExplainer, or KernelExplainer
  - Thread-pool async execution (non-blocking event loop)
  - SHA-256 input hashing for Redis cache key generation
  - Additivity verification on all explanation outputs
- **LIME Cross-Validation Engine** (`src/explainability/lime_engine.py`)
  - Spearman rank correlation comparison with SHAP rankings
  - Top-N feature overlap reporting
  - Categorical feature support with explicit index mapping
- **Counterfactual Explanation Engine** (`src/explainability/counterfactual.py`)
  - Gradient-free random-walk optimisation with actionability constraints
  - Diversity enforcement across generated counterfactuals
  - Feasibility scoring for immutability constraint adherence
- **Visualisation Module** (`src/explainability/visualisation.py`)
  - Waterfall, beeswarm, bar importance, heatmap, dependence scatter plots
  - `save_all_plots()` batch export function
- **Training Pipeline** (`src/pipelines/training_pipeline.py`)
  - Optuna Bayesian HPO with TPE sampler
  - MLflow experiment tracking (params, metrics, model, SHAP importances)
  - Post-training SHAP global importance computation
- **ONNX Export Pipeline** (`src/pipelines/onnx_export.py`)
  - sklearn, XGBoost, LightGBM conversion
  - Output validation (sklearn vs ONNX Runtime comparison)
  - Latency benchmarking (sklearn vs ONNX speedup)
  - `ONNXPredictor` production inference class
- **Drift Detection** (`src/monitoring/drift_detector.py`)
  - Population Stability Index (PSI) with 10-bin quantile bucketing
  - Kolmogorov-Smirnov test for continuous features
  - Chi-squared test for categorical features
  - Jensen-Shannon divergence
  - Severity classification (None/Low/Medium/High/Critical)
- **Model Registry** (`src/models/registry.py`)
  - MLflow model registration, versioning, stage transitions
  - In-process model caching for production inference
- **Model Evaluator** (`src/models/evaluator.py`)
  - Classification: ROC-AUC, PR-AUC, F1, Precision, Recall, MCC, Brier, Log Loss, ECE
  - Regression: RMSE, MAE, R², MAPE, Max Error, Median AE
  - Calibration: reliability curves, Expected Calibration Error
  - Fairness: Demographic Parity Difference, Equalised Odds by sensitive group
- **Redis Cache** (`src/utils/cache.py`)
  - Async Redis client with connection pooling and graceful fallback
- **Python SDK** (`sdk/python/xai_client.py`)
  - Synchronous `XAIClient` and async `AsyncXAIClient`
  - Typed response objects with waterfall print utility
  - Exponential backoff retry on transient failures

### Added — MLOps
- **Airflow DAG** (`dags/xai_training_pipeline.py`)
  - 8-stage pipeline: ingest → validate → featurise → train → evaluate → SHAP → drift → register
  - Quality gates: ROC-AUC > 0.80, PSI < 0.20
  - XCom-based inter-task data sharing
- **DVC Pipeline** (`dvc.yaml`) — 6 reproducible stages
- **Training CLI** (`scripts/train.py`) — Click-based with YAML config + `--set` overrides
- **Data Ingestion Script** (`scripts/ingest_data.py`) — Kaggle download with synthetic fallback
- **SHAP Generation Script** (`scripts/generate_shap_explanations.py`) — DVC Stage 6
- **SHAP Benchmark Suite** (`scripts/benchmark_shap.py`)
  - TreeExplainer, LinearExplainer, Kernel vs Tree comparison
  - Memory footprint measurement, markdown table output

### Added — Infrastructure
- **Dockerfile** (multi-stage, non-root user, read-only FS)
- **Docker Compose** (API + Redis + PostgreSQL + MLflow + Prometheus + Grafana)
- **Kubernetes manifests** (`k8s/deployment.yaml`) — Deployment, Service, HPA, Ingress, ConfigMap
- **Helm chart** (`k8s/helm/xai-platform/`) — Chart.yaml, values.yaml, deployment template
- **Terraform** (`terraform/main.tf`, `variables.tf`) — VPC, EKS, ECR, S3, ElastiCache, RDS, security groups
- **Prometheus config** (`docker/prometheus.yml`)
- **Prometheus AlertManager rules** (`docker/prometheus_alerts.yml`)
- **Grafana operations dashboard** (`docker/grafana/dashboards/xai_operations.json`)

### Added — Testing
- Unit tests: SHAPEngine, DriftDetector, visualisation, counterfactual, evaluator, SDK client
- Integration tests: all FastAPI endpoints (httpx AsyncClient)
- E2E tests: full pipeline (training → evaluation → SHAP → LIME → counterfactual → drift)
- Load tests: Locust suite with ExplainUser, PredictUser, HealthCheckUser
- conftest.py with session-scoped event loop

### Added — Documentation
- Model Card (`docs/model_card.md`) — Follows Mitchell et al. FAccT 2019 format
- Data Card (`docs/data_card.md`) — Follows Gebru et al. Datasheets format
- MkDocs site (`mkdocs.yml`, `docs/index.md`) — Material theme, math support, mermaid diagrams
- 900+ line README with ASCII banner, Mermaid diagrams, benchmarks, cost estimates

### Added — Notebooks
- `classification_shap.ipynb` — VotingClassifier ensemble, EDA, Optuna, local+global SHAP, LIME
- `regression_shap.ipynb` — XGBRegressor, geographic analysis, dependence plots, TreeSHAP benchmark

### Added — CI/CD
- GitHub Actions workflow (`.github/workflows/ci.yml`)
  - Lint (Ruff), type check (mypy), unit tests (3 Python versions), integration tests
  - Docker build (multi-arch), Trivy security scan, staging/production deploy

---

## [1.5.0] — 2023-09-15

### Added
- XGBoost sub-model to ensemble (RF → RF+XGB); ROC-AUC: 0.853 → 0.871
- SHAP TreeExplainer integration (replaced basic feature_importances_)
- MLflow experiment tracking (basic params and metrics)

### Changed
- `classification_shap.ipynb` updated with XGBoost SHAP analysis

---

## [1.0.0] — 2023-06-01

### Added
- Initial release with 4 Jupyter notebooks
- Bank Customer Churn dataset via Kaggle
- Random Forest baseline model
- Basic SHAP waterfall and bar plots
- requirements.txt, pyproject.toml

---

## Upcoming (v2.1.0)

- [ ] Prefect workflow orchestration (alternative to Airflow)
- [ ] TypeScript SDK for browser-based explanation apps
- [ ] SHAP interaction values (`shap_interaction_values` endpoint)
- [ ] Calibration-aware explanation (uncertainty intervals on SHAP values)
- [ ] Multi-tenant API with organisation-level model isolation
- [ ] Automated model card generation from MLflow metadata
- [ ] Explanation audit log (immutable, compliance-grade)
