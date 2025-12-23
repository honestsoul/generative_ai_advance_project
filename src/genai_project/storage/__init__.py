"""Storage utilities for caching and blob storage."""

from genai_project.storage.cache import (
    BaseCache,
    SQLiteCache,
    get_cache,
)
from genai_project.storage.blob import (
    BaseBlobStorage,
    LocalBlobStorage,
    get_blob_storage,
)

__all__ = [
    # Cache
    "BaseCache",
    "SQLiteCache",
    "get_cache",
    # Blob
    "BaseBlobStorage",
    "LocalBlobStorage",
    "get_blob_storage",
]

# Optional imports
try:
    from genai_project.storage.cache import RedisCache

    __all__.append("RedisCache")
except ImportError:
    pass

try:
    from genai_project.storage.blob import S3BlobStorage

    __all__.append("S3BlobStorage")
except ImportError:
    pass
