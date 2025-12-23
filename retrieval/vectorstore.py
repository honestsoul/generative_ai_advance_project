"""
Vector store implementations for RAG.

Supports multiple backends: in-memory, PostgreSQL (pgvector), etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from genai_project.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development/testing."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    async def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the store."""
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            self._documents[doc_id] = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            }

        logger.info("Added documents", count=len(texts))
        return ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using cosine similarity."""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        results = []

        for doc_id, doc in self._documents.items():
            # Apply filter
            if filter:
                match = all(
                    doc["metadata"].get(k) == v for k, v in filter.items()
                )
                if not match:
                    continue

            score = cosine_similarity(query_embedding, doc["embedding"])
            results.append(
                SearchResult(
                    id=doc_id,
                    content=doc["text"],
                    score=score,
                    metadata=doc["metadata"],
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        for doc_id in ids:
            self._documents.pop(doc_id, None)


class PGVectorStore(BaseVectorStore):
    """PostgreSQL vector store using pgvector extension."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "documents",
        embedding_dimension: int = 1536,
    ) -> None:
        """
        Initialize pgvector store.

        Requires: pip install asyncpg pgvector
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.connection_string)

            # Create table if not exists
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        embedding vector({self.embedding_dimension}),
                        metadata JSONB
                    )
                    """
                )
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    """
                )

        return self._pool

    async def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to PostgreSQL."""
        import json
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    doc_id,
                    text,
                    str(embedding),
                    json.dumps(metadata),
                )

        logger.info("Added documents to pgvector", count=len(texts))
        return ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using pgvector."""
        pool = await self._get_pool()

        filter_clause = ""
        if filter:
            conditions = [f"metadata->>'{k}' = '{v}'" for k, v in filter.items()]
            filter_clause = "WHERE " + " AND ".join(conditions)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) as score
                FROM {self.table_name}
                {filter_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )

        import json

        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                score=float(row["score"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete documents from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY($1)",
                ids,
            )


def get_vectorstore(
    backend: str = "memory",
    **kwargs,
) -> BaseVectorStore:
    """Get a vector store by backend name."""
    backends = {
        "memory": InMemoryVectorStore,
        "pgvector": PGVectorStore,
    }

    if backend not in backends:
        raise ValueError(f"Unknown vector store backend: {backend}")

    return backends[backend](**kwargs)
