"""Tests for retrieval pipeline."""

import pytest
from genai_project.retrieval import (
    FixedSizeChunker,
    RecursiveChunker,
    InMemoryVectorStore,
)


class TestChunking:
    """Tests for chunking strategies."""

    def test_fixed_size_chunker(self):
        """Test fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        text = "Hello World! This is a test."

        chunks = list(chunker.chunk(text))

        assert len(chunks) > 0
        assert all(len(c.content) <= 10 for c in chunks)

    def test_recursive_chunker(self):
        """Test recursive chunking."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = list(chunker.chunk(text))

        assert len(chunks) > 0

    def test_chunk_metadata(self):
        """Test that metadata is preserved."""
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Test content"
        metadata = {"source": "test.txt"}

        chunks = list(chunker.chunk(text, metadata))

        assert chunks[0].metadata == metadata


class TestVectorStore:
    """Tests for vector stores."""

    @pytest.mark.asyncio
    async def test_in_memory_add_and_search(self):
        """Test adding and searching in memory store."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["Hello world", "Goodbye world"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            ids=["1", "2"],
        )

        results = await store.search([1.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].id == "1"

    @pytest.mark.asyncio
    async def test_in_memory_delete(self):
        """Test deleting from memory store."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["Test"],
            embeddings=[[1.0, 0.0]],
            ids=["1"],
        )

        await store.delete(["1"])
        results = await store.search([1.0, 0.0])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_filter_search(self):
        """Test filtered search."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["Doc A", "Doc B"],
            embeddings=[[1.0, 0.0], [0.9, 0.1]],
            ids=["a", "b"],
            metadatas=[{"type": "article"}, {"type": "book"}],
        )

        results = await store.search(
            [1.0, 0.0],
            top_k=10,
            filter={"type": "book"},
        )

        assert len(results) == 1
        assert results[0].id == "b"
