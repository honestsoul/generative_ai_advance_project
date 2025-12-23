"""
Base class for embedding providers.

Defines the interface that all embedding provider implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embeddings: list[list[float]]
    model: str
    total_tokens: int = 0


class BaseEmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    provider_name: str = "base"

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed
            model: Optional model override

        Returns:
            EmbeddingResponse with embedding vectors
        """
        pass

    async def embed_single(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Optional model override

        Returns:
            Embedding vector
        """
        response = await self.embed([text], model)
        return response.embeddings[0]
