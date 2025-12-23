"""
Base class for LLM providers.

Defines the interface that all LLM provider implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class GenerateConfig:
    """Configuration for text generation."""

    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class GenerateResponse:
    """Response from text generation."""

    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw_response: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    provider_name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> GenerateResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Returns:
            GenerateResponse with the generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion for the given messages.

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Yields:
            String chunks of the generated content
        """
        pass

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
    ) -> str:
        """
        Simple text generation from a prompt string.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            config: Generation configuration

        Returns:
            Generated text content
        """
        messages = []
        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(Message(role=MessageRole.USER, content=prompt))

        response = await self.generate(messages, config)
        return response.content
