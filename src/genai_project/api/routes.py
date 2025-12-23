"""
API routes for the GenAI project.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genai_project.core.errors import GenAIError
from genai_project.core.logging import get_logger
from genai_project.providers.llm import (
    Message,
    MessageRole,
    GenerateConfig,
    OpenAIClient,
    AnthropicClient,
)

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="User prompt")
    system_prompt: str | None = Field(None, description="System prompt")
    provider: str = Field("openai", description="LLM provider (openai, anthropic)")
    model: str | None = Field(None, description="Model to use")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(4096, ge=1, le=100000)


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    provider: str


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str


class ChatRequest(BaseModel):
    """Request model for chat completion."""

    messages: list[ChatMessage]
    provider: str = Field("openai")
    model: str | None = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(4096, ge=1, le=100000)


# =============================================================================
# Routes
# =============================================================================


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from a prompt."""
    try:
        # Select provider
        if request.provider == "anthropic":
            client = AnthropicClient()
        else:
            client = OpenAIClient()

        # Generate response
        response = await client.generate_text(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            config=GenerateConfig(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
        )

        # For simple generate_text, we need to call generate to get usage
        messages = []
        if request.system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=request.system_prompt))
        messages.append(Message(role=MessageRole.USER, content=request.prompt))

        full_response = await client.generate(
            messages=messages,
            config=GenerateConfig(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
        )

        return GenerateResponse(
            content=full_response.content,
            model=full_response.model,
            input_tokens=full_response.usage.input_tokens,
            output_tokens=full_response.usage.output_tokens,
            provider=request.provider,
        )

    except GenAIError as e:
        logger.error("Generation error", error=str(e), code=e.code)
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest) -> GenerateResponse:
    """Chat completion with message history."""
    try:
        # Select provider
        if request.provider == "anthropic":
            client = AnthropicClient()
        else:
            client = OpenAIClient()

        # Convert messages
        messages = [
            Message(
                role=MessageRole(msg.role),
                content=msg.content,
            )
            for msg in request.messages
        ]

        # Generate response
        response = await client.generate(
            messages=messages,
            config=GenerateConfig(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
        )

        return GenerateResponse(
            content=response.content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider=request.provider,
        )

    except GenAIError as e:
        logger.error("Chat error", error=str(e), code=e.code)
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models() -> dict[str, list[str]]:
    """List available models by provider."""
    return {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
    }
