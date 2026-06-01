"""
tests/load/locustfile.py
=========================
Locust load testing suite for the XAI Platform API.

Run:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 10 --run-time 2m

Scenarios:
    - ExplainUser: heavy explanation endpoint users (70%)
    - PredictUser: lightweight prediction users (20%)
    - HealthCheckUser: monitoring probes (10%)
"""

from __future__ import annotations

import json
import random

from locust import HttpUser, between, task, TaskSet, events


# ---------------------------------------------------------------------------
# Payload generators
# ---------------------------------------------------------------------------

def make_explain_payload(n_instances: int = 1, n_features: int = 10) -> dict:
    return {
        "model_id": "churn-ensemble-v2",
        "instances": [
            [random.uniform(0.0, 1.0) for _ in range(n_features)]
            for _ in range(n_instances)
        ],
        "explainer_type": "auto",
        "scope": random.choice(["local", "global"]),
        "include_lime": False,
        "top_features": 10,
    }


def make_predict_payload(n_instances: int = 1, n_features: int = 10) -> dict:
    return {
        "model_id": "churn-ensemble-v2",
        "instances": [
            [random.uniform(0.0, 1.0) for _ in range(n_features)]
            for _ in range(n_instances)
        ],
        "return_probabilities": True,
    }


# ---------------------------------------------------------------------------
# Task sets
# ---------------------------------------------------------------------------

class ExplainTasks(TaskSet):
    @task(5)
    def explain_single(self) -> None:
        """Single-instance local explanation — most common pattern."""
        with self.client.post(
            "/api/v1/explain",
            json=make_explain_payload(n_instances=1),
            catch_response=True,
            name="/api/v1/explain [single]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
            else:
                data = resp.json()
                if "global_importance" not in data:
                    resp.failure("Missing global_importance in response")

    @task(2)
    def explain_batch_small(self) -> None:
        """Small batch explanation (10 instances)."""
        with self.client.post(
            "/api/v1/explain",
            json=make_explain_payload(n_instances=10),
            catch_response=True,
            name="/api/v1/explain [batch-10]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(1)
    def explain_global_scope(self) -> None:
        """Global scope explanation."""
        payload = make_explain_payload(n_instances=50)
        payload["scope"] = "global"
        with self.client.post(
            "/api/v1/explain",
            json=payload,
            catch_response=True,
            name="/api/v1/explain [global-50]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(1)
    def submit_async_batch(self) -> None:
        """Async batch job submission."""
        with self.client.post(
            "/api/v1/explain/batch",
            json=make_explain_payload(n_instances=100),
            catch_response=True,
            name="/api/v1/explain/batch [async]",
        ) as resp:
            if resp.status_code not in (200, 202):
                resp.failure(f"Expected 202, got {resp.status_code}")


class PredictTasks(TaskSet):
    @task(3)
    def predict_single(self) -> None:
        with self.client.post(
            "/api/v1/predict",
            json=make_predict_payload(n_instances=1),
            catch_response=True,
            name="/api/v1/predict [single]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(1)
    def list_models(self) -> None:
        self.client.get("/api/v1/models", name="/api/v1/models [list]")

    @task(1)
    def get_model(self) -> None:
        self.client.get("/api/v1/models/churn-ensemble-v2", name="/api/v1/models [get]")


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------

class ExplainUser(HttpUser):
    """Heavy explanation API users — data scientists, audit systems."""
    tasks = [ExplainTasks]
    weight = 7
    wait_time = between(0.5, 2.0)


class PredictUser(HttpUser):
    """Lightweight prediction API users — application backends."""
    tasks = [PredictTasks]
    weight = 2
    wait_time = between(0.1, 0.5)


class HealthCheckUser(HttpUser):
    """Monitoring probes — high frequency, lightweight."""
    weight = 1
    wait_time = between(5, 15)

    @task
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task
    def metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")


# ---------------------------------------------------------------------------
# Event hooks for custom reporting
# ---------------------------------------------------------------------------

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:  # type: ignore
    stats = environment.stats
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    for name, entry in stats.entries.items():
        if entry.num_requests > 0:
            print(
                f"{name[1]:45} | "
                f"RPS: {entry.current_rps:6.1f} | "
                f"p50: {entry.get_response_time_percentile(0.50):5.0f}ms | "
                f"p95: {entry.get_response_time_percentile(0.95):5.0f}ms | "
                f"p99: {entry.get_response_time_percentile(0.99):5.0f}ms | "
                f"Fail: {entry.num_failures}"
            )
