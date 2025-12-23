"""Embedding provider implementations."""

from genai_project.providers.embeddings.base import (
    BaseEmbeddingClient,
    EmbeddingResponse,
)
from genai_project.providers.embeddings.openai_embed import OpenAIEmbeddingClient

__all__ = [
    "BaseEmbeddingClient",
    "EmbeddingResponse",
    "OpenAIEmbeddingClient",
]

# Optional imports
try:
    from genai_project.providers.embeddings.cohere_embed import CohereEmbeddingClient

    __all__.append("CohereEmbeddingClient")
except ImportError:
    pass
