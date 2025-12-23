"""
GenAI Project - A production-ready Generative AI template.

This package provides a structured foundation for building
LLM-powered applications with best practices baked in.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from genai_project.core.settings import settings
from genai_project.core.logging import get_logger

__all__ = ["settings", "get_logger", "__version__"]
