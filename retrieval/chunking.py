"""
Document chunking strategies for RAG pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from genai_project.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict | None = None


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            yield Chunk(
                content=text[start:end],
                index=index,
                start_char=start,
                end_char=end,
                metadata=metadata,
            )

            start += self.chunk_size - self.chunk_overlap
            index += 1


class SentenceChunker(BaseChunker):
    """Sentence-based chunking."""

    def __init__(
        self,
        max_sentences: int = 5,
        min_chunk_size: int = 100,
    ) -> None:
        self.max_sentences = max_sentences
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text by sentences."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_start = 0
        char_pos = 0
        index = 0

        for sentence in sentences:
            current_chunk.append(sentence)

            if len(current_chunk) >= self.max_sentences:
                content = " ".join(current_chunk)
                if len(content) >= self.min_chunk_size:
                    yield Chunk(
                        content=content,
                        index=index,
                        start_char=current_start,
                        end_char=char_pos + len(sentence),
                        metadata=metadata,
                    )
                    index += 1
                    current_start = char_pos + len(sentence) + 1
                    current_chunk = []

            char_pos += len(sentence) + 1

        # Yield remaining
        if current_chunk:
            content = " ".join(current_chunk)
            yield Chunk(
                content=content,
                index=index,
                start_char=current_start,
                end_char=len(text),
                metadata=metadata,
            )


class RecursiveChunker(BaseChunker):
    """Recursive text splitting by separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Recursively split text."""
        chunks = self._split_text(text, self.separators)

        for index, (content, start, end) in enumerate(chunks):
            yield Chunk(
                content=content,
                index=index,
                start_char=start,
                end_char=end,
                metadata=metadata,
            )

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[tuple[str, int, int]]:
        """Recursively split text by separators."""
        if not separators:
            return [(text, 0, len(text))]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            return self._merge_splits(list(text), separator)

        splits = text.split(separator)

        if len(splits) == 1:
            # Separator not found, try next
            return self._split_text(text, remaining_separators)

        # Merge small splits
        return self._merge_splits(splits, separator)

    def _merge_splits(
        self,
        splits: list[str],
        separator: str,
    ) -> list[tuple[str, int, int]]:
        """Merge splits to respect chunk size."""
        chunks = []
        current_chunk = []
        current_length = 0
        char_pos = 0

        for split in splits:
            split_length = len(split) + len(separator)

            if current_length + split_length > self.chunk_size and current_chunk:
                content = separator.join(current_chunk)
                start = char_pos - len(content)
                chunks.append((content, start, char_pos))
                current_chunk = current_chunk[-1:] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(split)
            current_length += split_length
            char_pos += split_length

        if current_chunk:
            content = separator.join(current_chunk)
            chunks.append((content, char_pos - len(content), char_pos))

        return chunks


def get_chunker(
    strategy: str = "recursive",
    **kwargs,
) -> BaseChunker:
    """Get a chunker by strategy name."""
    chunkers = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunkers[strategy](**kwargs)
