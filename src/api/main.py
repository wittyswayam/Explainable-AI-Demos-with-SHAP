"""
Explainable AI Platform — FastAPI Application
=============================================
Production-grade REST API for SHAP-based model explanations.
Supports sync/async inference, batch processing, JWT auth,
Redis caching, rate limiting, and Prometheus metrics.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.routers import explain, health, models, predict
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.utils.cache import RedisCache

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "xai_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "xai_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
EXPLANATION_COUNT = Counter(
    "xai_explanations_total",
    "Total SHAP explanations generated",
    ["model_type", "explainer_type"],
)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Application lifespan: initialise resources on startup, teardown on shutdown."""
    configure_logging(level=settings.LOG_LEVEL)
    logger.info("🚀 Starting Explainable AI Platform v%s", settings.APP_VERSION)

    # Initialise Redis connection pool
    app.state.cache = RedisCache(url=settings.REDIS_URL)
    await app.state.cache.connect()
    logger.info("✅ Redis cache connected")

    yield  # ── application runs ──

    # Graceful shutdown
    await app.state.cache.disconnect()
    logger.info("👋 Explainable AI Platform shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Application factory — returns a configured FastAPI instance."""
    app = FastAPI(
        title="Explainable AI Platform",
        description=(
            "Enterprise-grade REST API for SHAP-powered model explanations. "
            "Supports tree ensembles, neural networks, and model-agnostic explainers."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Instrumentation middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        return response

    # ── Global exception handler ────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])
    app.include_router(explain.router, prefix="/api/v1/explain", tags=["Explain"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])

    # ── Prometheus metrics endpoint ──────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
    )
