"""Tests for workflows and tools."""

import pytest
from genai_project.workflows import Tool, ToolParameter, ToolRegistry, tool


class TestTools:
    """Tests for tool definitions."""

    def test_tool_schema(self):
        """Test tool OpenAI schema generation."""
        t = Tool(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=10,
                ),
            ],
        )

        schema = t.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution."""

        def add(a: int, b: int) -> int:
            return a + b

        t = Tool(
            name="add",
            description="Add numbers",
            parameters=[
                ToolParameter(name="a", type="integer", description="First number"),
                ToolParameter(name="b", type="integer", description="Second number"),
            ],
            handler=add,
        )

        result = await t.execute(a=2, b=3)
        assert result == 5

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(description="Multiply two numbers")
        def multiply(x: int, y: int) -> int:
            """Multiply x and y."""
            return x * y

        assert multiply.name == "multiply"
        assert multiply.description == "Multiply two numbers"
        assert len(multiply.parameters) == 2


class TestToolRegistry:
    """Tests for tool registry."""

    def test_register_and_get(self):
        """Test registering and retrieving tools."""
        registry = ToolRegistry()

        t = Tool(name="test", description="Test tool")
        registry.register(t)

        retrieved = registry.get("test")
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()

        registry.register(Tool(name="a", description="A"))
        registry.register(Tool(name="b", description="B"))

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_to_openai_tools(self):
        """Test converting to OpenAI format."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="search",
                description="Search",
                parameters=[
                    ToolParameter(name="q", type="string", description="Query")
                ],
            )
        )

        tools = registry.to_openai_tools()

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"
