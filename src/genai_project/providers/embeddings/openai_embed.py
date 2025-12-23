"""
OpenAI embedding client implementation.
"""

from openai import AsyncOpenAI

from genai_project.core.errors import AuthenticationError
from genai_project.core.logging import get_logger
from genai_project.core.settings import settings
from genai_project.core.telemetry import traced
from genai_project.providers.embeddings.base import BaseEmbeddingClient, EmbeddingResponse

logger = get_logger(__name__)


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding client."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "text-embedding-3-small",
    ) -> None:
        """
        Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key (defaults to settings)
            default_model: Default embedding model
        """
        self.api_key = api_key or (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )
        self.default_model = default_model

        if not self.api_key:
            raise AuthenticationError(self.provider_name)

        self.client = AsyncOpenAI(api_key=self.api_key)

    @traced("openai_embed")
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings using OpenAI API."""
        model = model or self.default_model

        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            total_tokens = response.usage.total_tokens if response.usage else 0

            logger.debug(
                "Generated embeddings",
                model=model,
                num_texts=len(texts),
                total_tokens=total_tokens,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error("OpenAI embedding error", error=str(e), model=model)
            raise
