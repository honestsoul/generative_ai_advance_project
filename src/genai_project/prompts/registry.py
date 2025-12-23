"""
Prompt template registry and management.

Provides versioned prompt loading from templates, system prompts,
and few-shot examples using Jinja2.
"""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from genai_project.core.errors import PromptNotFoundError, PromptRenderError
from genai_project.core.logging import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """
    Registry for managing prompt templates.

    Supports:
    - Jinja2 templates with variables
    - System prompts
    - Few-shot examples
    - Template versioning
    """

    def __init__(
        self,
        templates_dir: str | Path | None = None,
        system_dir: str | Path | None = None,
        fewshot_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize prompt registry.

        Args:
            templates_dir: Directory for Jinja2 templates
            system_dir: Directory for system prompts
            fewshot_dir: Directory for few-shot examples
        """
        base_dir = Path(__file__).parent

        self.templates_dir = Path(templates_dir) if templates_dir else base_dir / "templates"
        self.system_dir = Path(system_dir) if system_dir else base_dir / "system"
        self.fewshot_dir = Path(fewshot_dir) if fewshot_dir else base_dir / "fewshot"

        for directory in [self.templates_dir, self.system_dir, self.fewshot_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        self._system_cache: dict[str, str] = {}
        self._fewshot_cache: dict[str, list[dict[str, str]]] = {}

        logger.info(
            "Prompt registry initialized",
            templates_dir=str(self.templates_dir),
            system_dir=str(self.system_dir),
            fewshot_dir=str(self.fewshot_dir),
        )

    def render_template(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> str:
        """
        Render a Jinja2 template with variables.

        Args:
            name: Template name (without extension)
            variables: Template variables
            version: Optional version suffix

        Returns:
            Rendered template string
        """
        template_name = f"{name}_{version}.jinja2" if version else f"{name}.jinja2"

        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**(variables or {}))

            logger.debug(
                "Rendered template",
                template=template_name,
                variables=list((variables or {}).keys()),
            )

            return rendered

        except TemplateNotFound:
            raise PromptNotFoundError(template_name)
        except Exception as e:
            raise PromptRenderError(template_name, str(e))

    def get_system_prompt(self, name: str, version: str | None = None) -> str:
        """Load a system prompt from file."""
        filename = f"{name}_{version}.txt" if version else f"{name}.txt"
        cache_key = filename

        if cache_key in self._system_cache:
            return self._system_cache[cache_key]

        filepath = self.system_dir / filename

        if not filepath.exists():
            raise PromptNotFoundError(f"system/{filename}")

        content = filepath.read_text().strip()
        self._system_cache[cache_key] = content

        logger.debug("Loaded system prompt", name=name, version=version)

        return content

    def get_fewshot_examples(
        self,
        name: str,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Load few-shot examples from JSONL or JSON file."""
        if name in self._fewshot_cache:
            examples = self._fewshot_cache[name]
            return examples[:limit] if limit else examples

        jsonl_path = self.fewshot_dir / f"{name}.jsonl"
        json_path = self.fewshot_dir / f"{name}.json"

        examples: list[dict[str, str]] = []

        if jsonl_path.exists():
            with open(jsonl_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
        elif json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                examples = data if isinstance(data, list) else [data]
        else:
            raise PromptNotFoundError(f"fewshot/{name}")

        self._fewshot_cache[name] = examples

        logger.debug("Loaded few-shot examples", name=name, count=len(examples))

        return examples[:limit] if limit else examples

    def format_fewshot(
        self,
        examples: list[dict[str, str]],
        input_prefix: str = "Input: ",
        output_prefix: str = "Output: ",
        separator: str = "\n\n",
    ) -> str:
        """Format few-shot examples into a string."""
        formatted = []
        for ex in examples:
            formatted.append(
                f"{input_prefix}{ex.get('input', '')}\n"
                f"{output_prefix}{ex.get('output', '')}"
            )
        return separator.join(formatted)

    def build_prompt(
        self,
        template: str,
        variables: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        fewshot: str | None = None,
        fewshot_limit: int | None = None,
    ) -> dict[str, str]:
        """Build a complete prompt with system prompt and few-shot examples."""
        result: dict[str, str] = {}

        if system_prompt:
            result["system"] = self.get_system_prompt(system_prompt)

        user_parts = []

        if fewshot:
            examples = self.get_fewshot_examples(fewshot, fewshot_limit)
            user_parts.append(self.format_fewshot(examples))

        user_parts.append(self.render_template(template, variables))

        result["user"] = "\n\n".join(user_parts)

        return result

    def clear_cache(self) -> None:
        """Clear all cached prompts."""
        self._system_cache.clear()
        self._fewshot_cache.clear()
        logger.info("Prompt cache cleared")


_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Get the global prompt registry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
