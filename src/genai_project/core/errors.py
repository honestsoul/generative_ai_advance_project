"""
Custom exceptions for the GenAI project.

Provides a hierarchy of exceptions for different error types
with structured error information.
"""

from typing import Any


class GenAIError(Exception):
    """Base exception for all GenAI project errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(GenAIError):
    """Base exception for LLM/embedding provider errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.provider = provider
        self.details["provider"] = provider


class RateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
    ) -> None:
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after}s"
        super().__init__(
            message,
            provider=provider,
            code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when provider authentication fails."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Authentication failed for {provider}. Check your API key.",
            provider=provider,
            code="AUTHENTICATION_FAILED",
        )


class ModelNotFoundError(ProviderError):
    """Raised when requested model is not available."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            f"Model '{model}' not found for provider {provider}",
            provider=provider,
            code="MODEL_NOT_FOUND",
            details={"model": model},
        )


class ContextLengthExceededError(ProviderError):
    """Raised when input exceeds model's context length."""

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int,
        requested_tokens: int,
    ) -> None:
        super().__init__(
            f"Context length exceeded for {model}. Max: {max_tokens}, Requested: {requested_tokens}",
            provider=provider,
            code="CONTEXT_LENGTH_EXCEEDED",
            details={
                "model": model,
                "max_tokens": max_tokens,
                "requested_tokens": requested_tokens,
            },
        )


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(GenAIError):
    """Base exception for storage-related errors."""

    pass


class CacheError(StorageError):
    """Raised when cache operations fail."""

    pass


class BlobStorageError(StorageError):
    """Raised when blob storage operations fail."""

    pass


# =============================================================================
# Prompt Errors
# =============================================================================


class PromptError(GenAIError):
    """Base exception for prompt-related errors."""

    pass


class PromptNotFoundError(PromptError):
    """Raised when a prompt template is not found."""

    def __init__(self, prompt_name: str) -> None:
        super().__init__(
            f"Prompt template '{prompt_name}' not found",
            code="PROMPT_NOT_FOUND",
            details={"prompt_name": prompt_name},
        )


class PromptRenderError(PromptError):
    """Raised when prompt rendering fails."""

    def __init__(self, prompt_name: str, reason: str) -> None:
        super().__init__(
            f"Failed to render prompt '{prompt_name}': {reason}",
            code="PROMPT_RENDER_ERROR",
            details={"prompt_name": prompt_name, "reason": reason},
        )


# =============================================================================
# Workflow Errors
# =============================================================================


class WorkflowError(GenAIError):
    """Base exception for workflow-related errors."""

    pass


class ToolExecutionError(WorkflowError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(
            f"Tool '{tool_name}' execution failed: {reason}",
            code="TOOL_EXECUTION_ERROR",
            details={"tool_name": tool_name, "reason": reason},
        )


class ChainExecutionError(WorkflowError):
    """Raised when a chain execution fails."""

    def __init__(self, chain_name: str, step: str, reason: str) -> None:
        super().__init__(
            f"Chain '{chain_name}' failed at step '{step}': {reason}",
            code="CHAIN_EXECUTION_ERROR",
            details={"chain_name": chain_name, "step": step, "reason": reason},
        )
