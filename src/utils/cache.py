"""
Redis Cache Utility
===================
Async Redis client wrapper with JSON serialisation, TTL support,
and graceful fallback when Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class RedisCache:
    """Async Redis cache with JSON serialisation and connection pooling."""

    def __init__(self, url: str = "redis://localhost:6379/0", default_ttl: int = 3600) -> None:
        self.url = url
        self.default_ttl = default_ttl
        self._client: Optional[Any] = None

    async def connect(self) -> None:
        if not HAS_REDIS:
            logger.warning("redis-py not installed — cache disabled")
            return
        try:
            self._client = aioredis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            logger.info("Redis connected: %s", self.url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — running without cache", exc)
            self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as exc:
            logger.debug("Cache GET error: %s", exc)
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if not self._client:
            return False
        try:
            await self._client.set(key, value, ex=ttl or self.default_ttl)
            return True
        except Exception as exc:
            logger.debug("Cache SET error: %s", exc)
            return False

    async def delete(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as exc:
            logger.debug("Cache DELETE error: %s", exc)
            return False

    async def exists(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False
