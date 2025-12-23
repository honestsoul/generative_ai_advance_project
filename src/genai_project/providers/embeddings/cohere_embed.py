"""
Cohere embedding client implementation.

Requires: pip install cohere
"""

from genai_project.core.errors import AuthenticationError
from genai_project.core.logging import get_logger
from genai_project.core.settings import settings
from genai_project.core.telemetry import traced
from genai_project.providers.embeddings.base import BaseEmbeddingClient, EmbeddingResponse

logger = get_logger(__name__)


class CohereEmbeddingClient(BaseEmbeddingClient):
    """Cohere embedding client."""

    provider_name = "cohere"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "embed-english-v3.0",
    ) -> None:
        """
        Initialize Cohere embedding client.

        Args:
            api_key: Cohere API key (defaults to settings)
            default_model: Default embedding model
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required. Install with: pip install cohere"
            )

        self.api_key = api_key or (
            settings.cohere_api_key.get_secret_value()
            if settings.cohere_api_key
            else None
        )
        self.default_model = default_model

        if not self.api_key:
            raise AuthenticationError(self.provider_name)

        self.client = cohere.AsyncClient(api_key=self.api_key)

    @traced("cohere_embed")
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: str = "search_document",
    ) -> EmbeddingResponse:
        """
        Generate embeddings using Cohere API.

        Args:
            texts: List of texts to embed
            model: Optional model override
            input_type: Type of input (search_document, search_query, classification, clustering)
        """
        model = model or self.default_model

        try:
            response = await self.client.embed(
                texts=texts,
                model=model,
                input_type=input_type,
            )

            logger.debug(
                "Generated embeddings",
                model=model,
                num_texts=len(texts),
            )

            return EmbeddingResponse(
                embeddings=response.embeddings,
                model=model,
                total_tokens=0,  # Cohere doesn't return token count
            )

        except Exception as e:
            logger.error("Cohere embedding error", error=str(e), model=model)
            raise
