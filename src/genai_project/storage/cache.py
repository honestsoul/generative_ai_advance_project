"""
Caching utilities for LLM responses and embeddings.

Supports Redis and SQLite backends.
"""

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any

from genai_project.core.errors import CacheError
from genai_project.core.logging import get_logger
from genai_project.core.settings import settings

logger = get_logger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()


class SQLiteCache(BaseCache):
    """SQLite-based cache for local development."""

    def __init__(self, db_path: str = ".cache/cache.db") -> None:
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")

    async def get(self, key: str) -> Any | None:
        """Get value from SQLite cache."""
        import time

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                value, expires_at = row
                if expires_at and time.time() > expires_at:
                    await self.delete(key)
                    return None

                return json.loads(value)

        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """Set value in SQLite cache."""
        import time

        try:
            expires_at = time.time() + ttl.total_seconds() if ttl else None
            value_json = json.dumps(value)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, value_json, expires_at),
                )

        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            raise CacheError(f"Failed to set cache: {e}")

    async def delete(self, key: str) -> None:
        """Delete value from SQLite cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))

    async def clear(self) -> None:
        """Clear all cached values."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
        except Exception as e:
            logger.error("Cache clear error", error=str(e))


class RedisCache(BaseCache):
    """Redis-based cache for production."""

    def __init__(self, url: str | None = None) -> None:
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL (defaults to settings)
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis is required. Install with: pip install redis")

        self.url = url or settings.redis_url
        if not self.url:
            raise CacheError("Redis URL not configured")

        self.client = redis.from_url(self.url)

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """Set value in Redis cache."""
        try:
            value_json = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, value_json)
            else:
                await self.client.set(key, value_json)
        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            raise CacheError(f"Failed to set cache: {e}")

    async def delete(self, key: str) -> None:
        """Delete value from Redis cache."""
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))

    async def clear(self) -> None:
        """Clear all cached values."""
        try:
            await self.client.flushdb()
        except Exception as e:
            logger.error("Redis clear error", error=str(e))


def get_cache() -> BaseCache:
    """
    Get the appropriate cache instance based on configuration.

    Returns SQLite for development, Redis for production.
    """
    if settings.redis_url:
        return RedisCache()
    return SQLiteCache()
