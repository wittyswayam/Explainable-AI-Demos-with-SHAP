"""
Structured logging configuration.
Outputs JSON-formatted logs in production, human-readable in development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure root logger with structured or plain formatting."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if HAS_STRUCTLOG and fmt == "json":
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            stream=sys.stdout,
        )

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


class RequestContext:
    """Context manager to attach per-request metadata to log records."""

    def __init__(self, request_id: str, user_id: str | None = None) -> None:
        self.fields: Dict[str, Any] = {"request_id": request_id}
        if user_id:
            self.fields["user_id"] = user_id

    def __enter__(self) -> "RequestContext":
        if HAS_STRUCTLOG:
            structlog.contextvars.bind_contextvars(**self.fields)
        return self

    def __exit__(self, *_: Any) -> None:
        if HAS_STRUCTLOG:
            structlog.contextvars.clear_contextvars()
