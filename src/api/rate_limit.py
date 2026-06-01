"""
src/api/rate_limit.py
======================
Redis-backed sliding window rate limiter middleware.

Algorithm: Token bucket via Redis sorted-set.
Each request logs a timestamp; count timestamps in [now - window, now].
If count > limit → 429 Too Many Requests.

Falls back to in-memory dict when Redis is unavailable (single-node only).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class InMemorySlidingWindow:
    """Fallback rate limiter using an in-memory list of timestamps."""

    def __init__(self) -> None:
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds
        # Evict stale entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        count = len(self._requests[key])
        if count >= limit:
            return False, 0
        self._requests[key].append(now)
        return True, limit - count - 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter middleware.

    Config:
        requests_per_window: max requests allowed per window
        window_seconds: window duration
        exclude_paths: paths exempt from rate limiting (/health, /metrics)

    Response headers:
        X-RateLimit-Limit: max allowed
        X-RateLimit-Remaining: remaining in window
        X-RateLimit-Reset: seconds until window resets
    """

    EXCLUDE_PATHS = {"/health", "/health/live", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}

    def __init__(
        self,
        app: Any,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        key_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(app)
        self.limit = requests_per_window
        self.window = window_seconds
        self.key_func = key_func or self._default_key
        self._fallback = InMemorySlidingWindow()
        self._redis: Optional[Any] = None

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path in self.EXCLUDE_PATHS:
            return await call_next(request)

        key = self.key_func(request)
        allowed, remaining = await self._check_rate_limit(key)

        if not allowed:
            logger.warning("Rate limit exceeded: key=%s path=%s", key[:20], request.url.path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": self.limit,
                    "window_seconds": self.window,
                    "retry_after": self.window,
                },
                headers={
                    "Retry-After": str(self.window),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(self.window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self.window)
        return response

    async def _check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """Try Redis first, fall back to in-memory."""
        try:
            redis = await self._get_redis()
            if redis:
                return await self._redis_check(redis, key)
        except Exception as exc:
            logger.debug("Redis rate limit check failed, using fallback: %s", exc)
        return self._fallback.is_allowed(key, self.limit, self.window)

    async def _redis_check(self, redis: Any, key: str) -> Tuple[bool, int]:
        """Sliding window via Redis ZSET."""
        now = time.time()
        cutoff = now - self.window
        pipe = redis.pipeline()
        rate_key = f"rl:{key}"
        pipe.zremrangebyscore(rate_key, 0, cutoff)
        pipe.zadd(rate_key, {str(now): now})
        pipe.zcard(rate_key)
        pipe.expire(rate_key, self.window + 1)
        results = await pipe.execute()
        count = results[2]
        allowed = count <= self.limit
        remaining = max(0, self.limit - count)
        return allowed, remaining

    async def _get_redis(self) -> Optional[Any]:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                from src.core.config import get_settings
                settings = get_settings()
                self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    @staticmethod
    def _default_key(request: Request) -> str:
        """Rate limit by client IP + path prefix."""
        client_ip = request.client.host if request.client else "unknown"
        # Bucket all /api/v1/explain variants together
        path = request.url.path.split("/")[1:3]
        return f"{client_ip}:{'_'.join(path)}"
