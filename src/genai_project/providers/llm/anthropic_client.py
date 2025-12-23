"""
Anthropic Claude LLM client implementation.
"""

import time
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
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
    MessageRole,
    Usage,
)

logger = get_logger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client for text generation."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to settings)
            default_model: Default model to use (defaults to settings)
        """
        self.api_key = api_key or (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        self.default_model = default_model or settings.anthropic_default_model

        if not self.api_key:
            raise AuthenticationError(self.provider_name)

        self.client = AsyncAnthropic(api_key=self.api_key)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Anthropic uses a separate system parameter, so we extract it.

        Returns:
            Tuple of (system_prompt, converted_messages)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                converted.append({"role": role, "content": msg.content})

        return system_prompt, converted

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    @traced("anthropic_generate")
    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> GenerateResponse:
        """Generate completion using Anthropic API."""
        config = config or GenerateConfig()
        model = config.model or self.default_model

        start_time = time.perf_counter()
        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_prompt:
                kwargs["system"] = system_prompt
            if config.stop:
                kwargs["stop_sequences"] = config.stop
            if config.tools:
                kwargs["tools"] = self._convert_tools(config.tools)

            response = await self.client.messages.create(**kwargs)

        except Exception as e:
            self._handle_error(e, model)
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        record_llm_metrics(
            provider=self.provider_name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )

        return GenerateResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    @traced("anthropic_generate_stream")
    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        config = config or GenerateConfig()
        model = config.model or self.default_model

        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "stream": True,
            }

            if system_prompt:
                kwargs["system"] = system_prompt
            if config.stop:
                kwargs["stop_sequences"] = config.stop

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_error(e, model)
            raise

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return converted

    def _handle_error(self, error: Exception, model: str) -> None:
        """Convert Anthropic errors to our custom exceptions."""
        import anthropic

        error_message = str(error)

        if isinstance(error, anthropic.RateLimitError):
            raise RateLimitError(self.provider_name)
        elif isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError(self.provider_name)
        elif isinstance(error, anthropic.NotFoundError):
            raise ModelNotFoundError(self.provider_name, model)
        elif "context_length" in error_message.lower():
            raise ContextLengthExceededError(
                provider=self.provider_name,
                model=model,
                max_tokens=0,
                requested_tokens=0,
            )

        logger.error(
            "Anthropic API error",
            error=error_message,
            model=model,
        )
