"""
Tool definitions for LLM function calling.

Provides a framework for defining and executing tools
that can be used by LLMs.
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from genai_project.core.errors import ToolExecutionError
from genai_project.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class Tool:
    """Definition of a tool for LLM function calling."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        if self.handler is None:
            raise ToolExecutionError(self.name, "No handler defined")

        try:
            logger.debug("Executing tool", tool=self.name, args=list(kwargs.keys()))

            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)

            logger.debug("Tool executed successfully", tool=self.name)
            return result

        except Exception as e:
            logger.error("Tool execution failed", tool=self.name, error=str(e))
            raise ToolExecutionError(self.name, str(e))


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info("Registered tool", tool=tool.name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any] | str) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if tool is None:
            raise ToolExecutionError(name, "Tool not found")

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return await tool.execute(**arguments)


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator for creating tools from functions.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Example:
        @tool(name="search", description="Search the web")
        def search(query: str, limit: int = 10) -> list[str]:
            '''Search for something.'''
            ...
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        # Extract parameters from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Map Python types to JSON Schema types
            python_type = hints.get(param_name, str)
            json_type = _python_type_to_json(python_type)

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=json_type,
                    description=f"Parameter: {param_name}",
                    required=param.default == inspect.Parameter.empty,
                    default=None if param.default == inspect.Parameter.empty else param.default,
                )
            )

        return Tool(
            name=tool_name,
            description=tool_description.strip(),
            parameters=parameters,
            handler=func,
        )

    return decorator


def _python_type_to_json(python_type: type) -> str:
    """Convert Python type to JSON Schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Handle Optional and other typing constructs
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

    return type_map.get(python_type, "string")


# Global registry
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


# =============================================================================
# Example Tools
# =============================================================================


@tool(description="Get the current date and time")
def get_current_time() -> str:
    """Returns the current date and time."""
    from datetime import datetime

    return datetime.now().isoformat()


@tool(description="Perform a calculation")
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        # Only allow safe operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Register example tools
_registry.register(get_current_time)
_registry.register(calculate)
