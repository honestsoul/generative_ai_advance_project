"""Retrieval pipeline for RAG applications."""

from genai_project.retrieval.chunking import (
    Chunk,
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    get_chunker,
)
from genai_project.retrieval.vectorstore import (
    SearchResult,
    BaseVectorStore,
    InMemoryVectorStore,
    get_vectorstore,
)
from genai_project.retrieval.index import (
    Document,
    IndexStats,
    IndexBuilder,
)

__all__ = [
    # Chunking
    "Chunk",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "get_chunker",
    # Vector Store
    "SearchResult",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "get_vectorstore",
    # Index
    "Document",
    "IndexStats",
    "IndexBuilder",
]

# Optional imports
try:
    from genai_project.retrieval.vectorstore import PGVectorStore

    __all__.append("PGVectorStore")
except ImportError:
    pass

try:
    from genai_project.retrieval.rerank import (
        RerankResult,
        BaseReranker,
        get_reranker,
    )

    __all__.extend(["RerankResult", "BaseReranker", "get_reranker"])
except ImportError:
    pass
