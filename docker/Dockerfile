# ─── Stage 1: dependency builder ─────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (Docker layer cache)
COPY pyproject.toml requirements.txt ./

# Install dependencies into a prefix for clean copying
RUN pip install --upgrade pip wheel && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="Explainable AI Platform" \
      org.opencontainers.image.description="SHAP-powered model explanation REST API" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY pyproject.toml ./

# Non-root user for security
RUN addgroup --system xai && adduser --system --ingroup xai xai
USER xai

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
