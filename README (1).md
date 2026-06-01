# Explainable AI Demos with SHAP

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-82%25-brightgreen?style=flat-square)](https://codecov.io/)

Most ML models are good at making predictions. The problem is they rarely tell you *why*. That matters a lot when you're in a regulated industry, presenting results to stakeholders, or just trying to trust your own model enough to ship it.

This project is a hands-on demonstration of how to make ML models explainable using SHAP and LIME — built on top of a real bank customer churn dataset. It goes beyond the usual notebook demos and includes a proper REST API, a training pipeline, drift monitoring, and deployment configs you can actually use.

---

## What's inside

The core idea is simple: train a model, then explain every prediction it makes — both globally (what features matter overall) and locally (why did the model score *this specific customer* the way it did).

The stack built around that idea includes:

- **SHAP explanations** with auto-selection between TreeExplainer, LinearExplainer, and model-agnostic approaches
- **LIME cross-validation** to independently verify whether SHAP's feature rankings make sense
- **Counterfactual explanations** — "what would need to change for this prediction to flip?"
- **FastAPI service** to serve explanations over HTTP with caching, auth, and rate limiting
- **Optuna hyperparameter search** baked into the training pipeline
- **MLflow** for experiment tracking and model versioning
- **DVC** for reproducible data and training pipelines
- **Drift detection** using PSI and KS tests so you know when to retrain
- **Docker + Kubernetes configs** if you want to deploy this somewhere real

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                               │
│          Notebooks · REST Consumers · Grafana                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     FastAPI Service                          │
│   /explain   /predict   /models   /health   /metrics        │
└──────┬──────────────┬──────────────────────────┬────────────┘
       │              │                          │
┌──────▼──────┐ ┌─────▼──────┐         ┌────────▼────────┐
│ SHAP Engine │ │LIME Engine │         │  Redis Cache    │
│ (auto-select│ │(cross-     │         │  (SHAP values)  │
│  explainer) │ │ validate)  │         └─────────────────┘
└──────┬──────┘ └────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                     MLflow Registry                          │
│     Model versions · Experiment runs · Artifacts            │
└──────┬──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                  Training Pipeline (DVC)                     │
│  ingest → validate → featurise → train → evaluate → explain │
│                    (Optuna HPO inside)                       │
└──────┬──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│               Monitoring (Prometheus + Grafana)              │
│       Drift detection · API metrics · Alert rules           │
└─────────────────────────────────────────────────────────────┘
```

---

## Getting started

Clone the repo and install dependencies:

```bash
git clone https://github.com/wittyswayam/Explainable-AI-Demos-with-SHAP.git
cd Explainable-AI-Demos-with-SHAP
pip install -r requirements.txt
```

To start everything locally with Docker:

```bash
cd docker
docker compose up -d
```

That gives you the API at `http://localhost:8000`, MLflow at `http://localhost:5000`, and Grafana at `http://localhost:3000`.

To run the full training pipeline:

```bash
dvc repro
```

To call the explanation API directly:

```bash
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"model_id": "churn-ensemble-v2", "instances": [[619, 0, 1, 42, 2, 0.0, 1, 1, 1, 101348]], "scope": "local"}'
```

---

## Notebooks

There are two research notebooks worth starting with:

- **`classification_shap.ipynb`** — trains a VotingClassifier on the churn dataset, explains predictions with SHAP and LIME, and shows why the model-agnostic explainer is required for ensemble models
- **`regression_shap.ipynb`** — uses TreeSHAP on XGBoost for California Housing, with geographic SHAP plots that show where income matters most for house prices

---

## Project layout

```
src/
  api/          → FastAPI app, routers, auth, rate limiting
  explainability/ → SHAP engine, LIME engine, counterfactuals
  models/       → MLflow registry wrapper, evaluator
  monitoring/   → drift detection, Prometheus collectors
  pipelines/    → training pipeline, ONNX export
scripts/        → CLI tools for each DVC stage
notebooks/      → Jupyter research notebooks
tests/          → unit, integration, e2e, load tests
docker/         → Dockerfile, Compose, Grafana dashboards
k8s/            → Kubernetes manifests, Helm chart
terraform/      → AWS infrastructure (EKS, RDS, ElastiCache)
```

---

## License

MIT — do whatever you want with it.
