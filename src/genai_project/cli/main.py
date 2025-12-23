"""
CLI entrypoints for the GenAI project.

Run with: genai --help
"""

import argparse
import asyncio
import sys
from typing import NoReturn

from genai_project.core.logging import get_logger
from genai_project.core.settings import settings
from genai_project.providers.llm import OpenAIClient, AnthropicClient, GenerateConfig

logger = get_logger(__name__)


def main() -> NoReturn:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="GenAI Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", help="The prompt to generate from")
    gen_parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider",
    )
    gen_parser.add_argument("--model", "-m", help="Model to use")
    gen_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Temperature (0-2)",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate",
    )
    gen_parser.add_argument(
        "--system",
        "-s",
        help="System prompt",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider",
    )
    chat_parser.add_argument("--model", "-m", help="Model to use")

    # Info command
    subparsers.add_parser("info", help="Show configuration info")

    # API command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument("--host", default=settings.api_host)
    api_parser.add_argument("--port", "-p", type=int, default=settings.api_port)

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(cmd_generate(args))
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "info":
        cmd_info()
    elif args.command == "api":
        cmd_api(args)
    else:
        parser.print_help()
        sys.exit(0)

    sys.exit(0)


async def cmd_generate(args: argparse.Namespace) -> None:
    """Handle generate command."""
    try:
        if args.provider == "anthropic":
            client = AnthropicClient()
        else:
            client = OpenAIClient()

        config = GenerateConfig(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        response = await client.generate_text(
            prompt=args.prompt,
            system_prompt=args.system,
            config=config,
        )

        print(response)

    except Exception as e:
        logger.error("Generation failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_chat(args: argparse.Namespace) -> None:
    """Handle interactive chat command."""
    from genai_project.providers.llm import Message, MessageRole

    try:
        if args.provider == "anthropic":
            client = AnthropicClient()
        else:
            client = OpenAIClient()

        messages: list[Message] = []
        model = args.model or (
            settings.anthropic_default_model
            if args.provider == "anthropic"
            else settings.openai_default_model
        )

        print(f"Chat with {args.provider} ({model})")
        print("Type 'quit' or 'exit' to end the conversation")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append(Message(role=MessageRole.USER, content=user_input))

            response = await client.generate(
                messages=messages,
                config=GenerateConfig(model=args.model),
            )

            messages.append(
                Message(role=MessageRole.ASSISTANT, content=response.content)
            )

            print(f"\nAssistant: {response.content}")

    except Exception as e:
        logger.error("Chat failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info() -> None:
    """Show configuration information."""
    print("GenAI Project Configuration")
    print("=" * 40)
    print(f"Environment: {settings.environment}")
    print(f"Log Level: {settings.log_level}")
    print(f"Debug: {settings.debug}")
    print()
    print("Providers:")
    print(f"  OpenAI: {'configured' if settings.openai_api_key else 'not configured'}")
    print(f"  Anthropic: {'configured' if settings.anthropic_api_key else 'not configured'}")
    print()
    print("Storage:")
    print(f"  Redis: {settings.redis_url or 'not configured'}")
    print(f"  S3: {settings.aws_s3_bucket or 'not configured'}")


def cmd_api(args: argparse.Namespace) -> None:
    """Run the API server."""
    try:
        import uvicorn

        print(f"Starting API server at http://{args.host}:{args.port}")
        uvicorn.run(
            "genai_project.api.main:app",
            host=args.host,
            port=args.port,
            reload=settings.is_development,
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
