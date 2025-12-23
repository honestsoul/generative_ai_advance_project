"""
AWS Bedrock LLM client implementation.

This is an optional provider for AWS Bedrock models.
Requires: pip install boto3
"""

import json
import time
from typing import Any, AsyncIterator

from genai_project.core.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
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


class BedrockClient(BaseLLMClient):
    """
    AWS Bedrock API client for text generation.

    Supports Claude models on Bedrock.
    """

    provider_name = "bedrock"

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region: str | None = None,
        default_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ) -> None:
        """
        Initialize Bedrock client.

        Args:
            aws_access_key_id: AWS access key (defaults to settings)
            aws_secret_access_key: AWS secret key (defaults to settings)
            region: AWS region (defaults to settings)
            default_model: Default Bedrock model ID
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock. Install with: pip install boto3"
            )

        self.region = region or settings.aws_region
        self.default_model = default_model

        # Get credentials
        access_key = aws_access_key_id or settings.aws_access_key_id
        secret_key = aws_secret_access_key or (
            settings.aws_secret_access_key.get_secret_value()
            if settings.aws_secret_access_key
            else None
        )

        # Create client - uses default credential chain if keys not provided
        client_kwargs: dict[str, Any] = {"region_name": self.region}
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

        try:
            self.client = boto3.client("bedrock-runtime", **client_kwargs)
        except Exception as e:
            raise AuthenticationError(self.provider_name) from e

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Bedrock/Claude format."""
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                converted.append({"role": role, "content": msg.content})

        return system_prompt, converted

    @traced("bedrock_generate")
    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> GenerateResponse:
        """Generate completion using Bedrock API."""
        import asyncio

        config = config or GenerateConfig()
        model = config.model or self.default_model

        start_time = time.perf_counter()
        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            body: dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": converted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_prompt:
                body["system"] = system_prompt
            if config.stop:
                body["stop_sequences"] = config.stop

            # Run synchronous boto3 call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                ),
            )

            response_body = json.loads(response["body"].read())

        except Exception as e:
            self._handle_error(e, model)
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract content
        content = ""
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = Usage(
            input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
            output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
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
            model=model,
            usage=usage,
            finish_reason=response_body.get("stop_reason"),
            raw_response=response_body,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion using Bedrock API."""
        import asyncio

        config = config or GenerateConfig()
        model = config.model or self.default_model

        system_prompt, converted_messages = self._convert_messages(messages)

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": converted_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        if system_prompt:
            body["system"] = system_prompt

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model_with_response_stream(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                ),
            )

            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

        except Exception as e:
            self._handle_error(e, model)
            raise

    def _handle_error(self, error: Exception, model: str) -> None:
        """Convert Bedrock errors to our custom exceptions."""
        error_message = str(error)
        error_type = type(error).__name__

        if "AccessDeniedException" in error_type or "credentials" in error_message.lower():
            raise AuthenticationError(self.provider_name)
        elif "ResourceNotFoundException" in error_type:
            raise ModelNotFoundError(self.provider_name, model)
        elif "ThrottlingException" in error_type:
            raise RateLimitError(self.provider_name)

        logger.error(
            "Bedrock API error",
            error=error_message,
            error_type=error_type,
            model=model,
        )
        raise ProviderError(
            message=f"Bedrock error: {error_message}",
            provider=self.provider_name,
        )
