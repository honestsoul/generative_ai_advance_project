"""
Index building pipeline for RAG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genai_project.core.logging import get_logger
from genai_project.providers.embeddings import BaseEmbeddingClient
from genai_project.retrieval.chunking import BaseChunker, Chunk, get_chunker
from genai_project.retrieval.vectorstore import BaseVectorStore

logger = get_logger(__name__)


@dataclass
class Document:
    """A document to index."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class IndexStats:
    """Statistics from an indexing run."""

    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    errors: int


class IndexBuilder:
    """Build vector index from documents."""

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        vector_store: BaseVectorStore,
        chunker: BaseChunker | None = None,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize index builder.

        Args:
            embedding_client: Client for generating embeddings
            vector_store: Vector store for storing embeddings
            chunker: Chunking strategy (default: recursive)
            batch_size: Number of chunks to embed at once
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.chunker = chunker or get_chunker("recursive")
        self.batch_size = batch_size

    async def index_documents(
        self,
        documents: list[Document],
    ) -> IndexStats:
        """
        Index a list of documents.

        Args:
            documents: Documents to index

        Returns:
            IndexStats with indexing results
        """
        stats = IndexStats(
            documents_processed=0,
            chunks_created=0,
            embeddings_generated=0,
            errors=0,
        )

        all_chunks: list[Chunk] = []
        chunk_doc_ids: list[str] = []

        # Chunk all documents
        for doc in documents:
            try:
                doc_metadata = doc.metadata or {}
                doc_metadata["document_id"] = doc.id

                for chunk in self.chunker.chunk(doc.content, doc_metadata):
                    all_chunks.append(chunk)
                    chunk_doc_ids.append(doc.id)

                stats.documents_processed += 1

            except Exception as e:
                logger.error("Failed to chunk document", doc_id=doc.id, error=str(e))
                stats.errors += 1

        stats.chunks_created = len(all_chunks)
        logger.info("Chunked documents", chunks=len(all_chunks))

        # Generate embeddings in batches
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i : i + self.batch_size]
            texts = [chunk.content for chunk in batch]

            try:
                response = await self.embedding_client.embed(texts)
                embeddings = response.embeddings

                # Store in vector store
                ids = [f"{chunk_doc_ids[i + j]}_{batch[j].index}" for j in range(len(batch))]
                metadatas = [chunk.metadata for chunk in batch]

                await self.vector_store.add(
                    texts=texts,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas,
                )

                stats.embeddings_generated += len(embeddings)

            except Exception as e:
                logger.error("Failed to embed batch", batch_start=i, error=str(e))
                stats.errors += 1

        logger.info(
            "Indexing complete",
            documents=stats.documents_processed,
            chunks=stats.chunks_created,
            embeddings=stats.embeddings_generated,
            errors=stats.errors,
        )

        return stats

    async def index_file(
        self,
        filepath: str | Path,
        doc_id: str | None = None,
    ) -> IndexStats:
        """Index a single file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text()
        doc_id = doc_id or filepath.stem

        doc = Document(
            id=doc_id,
            content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
            },
        )

        return await self.index_documents([doc])

    async def index_directory(
        self,
        directory: str | Path,
        glob_pattern: str = "**/*.txt",
    ) -> IndexStats:
        """Index all files in a directory."""
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents = []
        for filepath in directory.glob(glob_pattern):
            if filepath.is_file():
                try:
                    content = filepath.read_text()
                    documents.append(
                        Document(
                            id=str(filepath.relative_to(directory)),
                            content=content,
                            metadata={
                                "source": str(filepath),
                                "filename": filepath.name,
                            },
                        )
                    )
                except Exception as e:
                    logger.error("Failed to read file", path=str(filepath), error=str(e))

        logger.info("Found files to index", count=len(documents))

        return await self.index_documents(documents)
