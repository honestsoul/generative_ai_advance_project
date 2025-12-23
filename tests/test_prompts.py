"""Tests for prompt management."""

import pytest
from genai_project.prompts import PromptRegistry


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_render_template(self, tmp_path):
        """Test rendering a Jinja2 template."""
        # Create test template
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "test.jinja2").write_text("Hello, {{ name }}!")

        registry = PromptRegistry(templates_dir=templates_dir)
        result = registry.render_template("test", {"name": "World"})

        assert result == "Hello, World!"

    def test_get_system_prompt(self, tmp_path):
        """Test loading system prompt."""
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        (system_dir / "assistant.txt").write_text("You are a helpful assistant.")

        registry = PromptRegistry(system_dir=system_dir)
        result = registry.get_system_prompt("assistant")

        assert result == "You are a helpful assistant."

    def test_get_fewshot_examples(self, tmp_path):
        """Test loading few-shot examples."""
        fewshot_dir = tmp_path / "fewshot"
        fewshot_dir.mkdir()
        (fewshot_dir / "qa.jsonl").write_text(
            '{"input": "Q1", "output": "A1"}\n{"input": "Q2", "output": "A2"}'
        )

        registry = PromptRegistry(fewshot_dir=fewshot_dir)
        examples = registry.get_fewshot_examples("qa")

        assert len(examples) == 2
        assert examples[0]["input"] == "Q1"
        assert examples[1]["output"] == "A2"

    def test_fewshot_limit(self, tmp_path):
        """Test limiting few-shot examples."""
        fewshot_dir = tmp_path / "fewshot"
        fewshot_dir.mkdir()
        (fewshot_dir / "qa.jsonl").write_text(
            '{"input": "Q1", "output": "A1"}\n'
            '{"input": "Q2", "output": "A2"}\n'
            '{"input": "Q3", "output": "A3"}'
        )

        registry = PromptRegistry(fewshot_dir=fewshot_dir)
        examples = registry.get_fewshot_examples("qa", limit=2)

        assert len(examples) == 2
