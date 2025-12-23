"""
OpenAI LLM client implementation.
"""

import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from genai_project.core.errors import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelNotFoundError,
    RateLimitError,
)
from genai_project.core.logging import get_logger
from genai_project.core.settings import settings
from genai_project.core.telemetry import record_llm_metrics, traced
from genai_project.providers.llm.base import (
    BaseLLMClient,
    GenerateConfig,
    GenerateResponse,
    Message,
    Usage,
)

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for text generation."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to settings)
            organization: OpenAI organization ID (defaults to settings)
            default_model: Default model to use (defaults to settings)
        """
        self.api_key = api_key or (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )
        self.organization = organization or settings.openai_org_id
        self.default_model = default_model or settings.openai_default_model

        if not self.api_key:
            raise AuthenticationError(self.provider_name)

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    @traced("openai_generate")
    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> GenerateResponse:
        """Generate completion using OpenAI API."""
        config = config or GenerateConfig()
        model = config.model or self.default_model

        start_time = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "presence_penalty": config.presence_penalty,
                "frequency_penalty": config.frequency_penalty,
            }

            if config.stop:
                kwargs["stop"] = config.stop
            if config.tools:
                kwargs["tools"] = config.tools
            if config.tool_choice:
                kwargs["tool_choice"] = config.tool_choice
            if config.response_format:
                kwargs["response_format"] = config.response_format

            response = await self.client.chat.completions.create(**kwargs)

        except Exception as e:
            self._handle_error(e, model)
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        choice = response.choices[0]

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        record_llm_metrics(
            provider=self.provider_name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )

        return GenerateResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=[tc.model_dump() for tc in choice.message.tool_calls]
            if choice.message.tool_calls
            else None,
            raw_response=response,
        )

    @traced("openai_generate_stream")
    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        config = config or GenerateConfig()
        model = config.model or self.default_model

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "stream": True,
            }

            if config.stop:
                kwargs["stop"] = config.stop

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e, model)
            raise

    def _handle_error(self, error: Exception, model: str) -> None:
        """Convert OpenAI errors to our custom exceptions."""
        import openai

        error_message = str(error)

        if isinstance(error, openai.RateLimitError):
            raise RateLimitError(self.provider_name)
        elif isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(self.provider_name)
        elif isinstance(error, openai.NotFoundError):
            raise ModelNotFoundError(self.provider_name, model)
        elif "context_length_exceeded" in error_message.lower():
            raise ContextLengthExceededError(
                provider=self.provider_name,
                model=model,
                max_tokens=0,  # Would need to parse from error
                requested_tokens=0,
            )

        logger.error(
            "OpenAI API error",
            error=error_message,
            model=model,
        )
