"""
Reranking utilities for improving retrieval quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from genai_project.core.logging import get_logger
from genai_project.retrieval.vectorstore import SearchResult

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result after reranking."""

    id: str
    content: str
    original_score: float
    rerank_score: float
    metadata: dict | None = None


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank search results."""
        pass


class CohereReranker(BaseReranker):
    """Cohere reranker implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        """
        Initialize Cohere reranker.

        Requires: pip install cohere
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is required. Install with: pip install cohere")

        from genai_project.core.settings import settings

        self.api_key = api_key or (
            settings.cohere_api_key.get_secret_value()
            if settings.cohere_api_key
            else None
        )
        self.model = model

        if not self.api_key:
            raise ValueError("Cohere API key required for reranking")

        self.client = cohere.AsyncClient(api_key=self.api_key)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using Cohere."""
        if not results:
            return []

        documents = [r.content for r in results]
        top_k = top_k or len(results)

        try:
            response = await self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
            )

            reranked = []
            for item in response.results:
                original = results[item.index]
                reranked.append(
                    RerankResult(
                        id=original.id,
                        content=original.content,
                        original_score=original.score,
                        rerank_score=item.relevance_score,
                        metadata=original.metadata,
                    )
                )

            logger.debug("Reranked results", count=len(reranked))
            return reranked

        except Exception as e:
            logger.error("Reranking failed", error=str(e))
            # Fallback to original order
            return [
                RerankResult(
                    id=r.id,
                    content=r.content,
                    original_score=r.score,
                    rerank_score=r.score,
                    metadata=r.metadata,
                )
                for r in results[:top_k]
            ]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """
        Initialize cross-encoder reranker.

        Requires: pip install sentence-transformers
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using cross-encoder."""
        import asyncio

        if not results:
            return []

        top_k = top_k or len(results)

        # Create pairs for scoring
        pairs = [(query, r.content) for r in results]

        # Run synchronous model in executor
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs),
        )

        # Combine with original results
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = [
            RerankResult(
                id=r.id,
                content=r.content,
                original_score=r.score,
                rerank_score=float(score),
                metadata=r.metadata,
            )
            for r, score in scored[:top_k]
        ]

        logger.debug("Reranked results", count=len(reranked))
        return reranked


def get_reranker(
    backend: str = "cohere",
    **kwargs,
) -> BaseReranker:
    """Get a reranker by backend name."""
    backends = {
        "cohere": CohereReranker,
        "cross-encoder": CrossEncoderReranker,
    }

    if backend not in backends:
        raise ValueError(f"Unknown reranker backend: {backend}")

    return backends[backend](**kwargs)
