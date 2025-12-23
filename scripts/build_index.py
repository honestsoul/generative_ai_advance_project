#!/usr/bin/env python3
"""
Build vector index from documents.

Usage:
    python scripts/build_index.py --input ./docs --output ./index
"""

import argparse
import asyncio
from pathlib import Path

from genai_project.core.logging import get_logger
from genai_project.providers.embeddings import OpenAIEmbeddingClient
from genai_project.retrieval import IndexBuilder, get_vectorstore

logger = get_logger(__name__)


async def main(args: argparse.Namespace) -> None:
    """Build index from documents."""
    input_dir = Path(args.input)

    if not input_dir.exists():
        logger.error("Input directory not found", path=str(input_dir))
        return

    # Initialize components
    embedding_client = OpenAIEmbeddingClient()
    vector_store = get_vectorstore("memory")  # Use pgvector for production

    builder = IndexBuilder(
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=args.batch_size,
    )

    # Build index
    logger.info("Building index", input=str(input_dir), pattern=args.pattern)

    stats = await builder.index_directory(
        directory=input_dir,
        glob_pattern=args.pattern,
    )

    logger.info(
        "Index built successfully",
        documents=stats.documents_processed,
        chunks=stats.chunks_created,
        embeddings=stats.embeddings_generated,
        errors=stats.errors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector index from documents")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing documents",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="**/*.txt",
        help="Glob pattern for files (default: **/*.txt)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
