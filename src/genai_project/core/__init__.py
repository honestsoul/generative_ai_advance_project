"""Core module - configuration, logging, and utilities."""

from genai_project.core.settings import settings, get_settings
from genai_project.core.logging import get_logger, setup_logging
from genai_project.core.errors import (
    GenAIError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ContextLengthExceededError,
    StorageError,
    CacheError,
    BlobStorageError,
    PromptError,
    PromptNotFoundError,
    PromptRenderError,
    WorkflowError,
    ToolExecutionError,
    ChainExecutionError,
)

__all__ = [
    # Settings
    "settings",
    "get_settings",
    # Logging
    "get_logger",
    "setup_logging",
    # Errors
    "GenAIError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ContextLengthExceededError",
    "StorageError",
    "CacheError",
    "BlobStorageError",
    "PromptError",
    "PromptNotFoundError",
    "PromptRenderError",
    "WorkflowError",
    "ToolExecutionError",
    "ChainExecutionError",
]
