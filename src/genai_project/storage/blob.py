"""
Blob storage utilities for S3 and GCS.

Provides a unified interface for object storage operations.
"""

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from genai_project.core.errors import BlobStorageError
from genai_project.core.logging import get_logger
from genai_project.core.settings import settings

logger = get_logger(__name__)


class BaseBlobStorage(ABC):
    """Abstract base class for blob storage implementations."""

    @abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload data to blob storage.

        Args:
            key: Object key/path
            data: Data to upload
            content_type: MIME type

        Returns:
            URL or URI of uploaded object
        """
        pass

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """
        Download data from blob storage.

        Args:
            key: Object key/path

        Returns:
            Downloaded bytes
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete object from blob storage."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass

    @abstractmethod
    async def list_objects(self, prefix: str = "") -> list[str]:
        """List objects with given prefix."""
        pass


class S3BlobStorage(BaseBlobStorage):
    """AWS S3 blob storage implementation."""

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
    ) -> None:
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            region: AWS region
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required. Install with: pip install boto3")

        self.bucket = bucket or settings.aws_s3_bucket
        self.region = region or settings.aws_region

        if not self.bucket:
            raise BlobStorageError("S3 bucket not configured")

        self.client = boto3.client("s3", region_name=self.region)

    async def upload(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload to S3."""
        import asyncio

        try:
            if isinstance(data, bytes):
                data = BytesIO(data)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upload_fileobj(
                    data,
                    self.bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                ),
            )

            return f"s3://{self.bucket}/{key}"

        except Exception as e:
            logger.error("S3 upload error", key=key, error=str(e))
            raise BlobStorageError(f"Failed to upload to S3: {e}")

    async def download(self, key: str) -> bytes:
        """Download from S3."""
        import asyncio

        try:
            buffer = BytesIO()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.download_fileobj(self.bucket, key, buffer),
            )
            return buffer.getvalue()

        except Exception as e:
            logger.error("S3 download error", key=key, error=str(e))
            raise BlobStorageError(f"Failed to download from S3: {e}")

    async def delete(self, key: str) -> None:
        """Delete from S3."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_object(Bucket=self.bucket, Key=key),
            )
        except Exception as e:
            logger.error("S3 delete error", key=key, error=str(e))
            raise BlobStorageError(f"Failed to delete from S3: {e}")

    async def exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.head_object(Bucket=self.bucket, Key=key),
            )
            return True
        except Exception:
            return False

    async def list_objects(self, prefix: str = "") -> list[str]:
        """List objects in S3."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix),
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            logger.error("S3 list error", prefix=prefix, error=str(e))
            raise BlobStorageError(f"Failed to list S3 objects: {e}")


class LocalBlobStorage(BaseBlobStorage):
    """Local filesystem blob storage for development."""

    def __init__(self, base_path: str = ".storage") -> None:
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get full path for a key."""
        return self.base_path / key

    async def upload(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload to local filesystem."""
        try:
            path = self._get_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                path.write_bytes(data.read())

            return f"file://{path.absolute()}"

        except Exception as e:
            logger.error("Local upload error", key=key, error=str(e))
            raise BlobStorageError(f"Failed to upload locally: {e}")

    async def download(self, key: str) -> bytes:
        """Download from local filesystem."""
        try:
            path = self._get_path(key)
            return path.read_bytes()
        except Exception as e:
            logger.error("Local download error", key=key, error=str(e))
            raise BlobStorageError(f"Failed to download locally: {e}")

    async def delete(self, key: str) -> None:
        """Delete from local filesystem."""
        try:
            path = self._get_path(key)
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.error("Local delete error", key=key, error=str(e))

    async def exists(self, key: str) -> bool:
        """Check if file exists locally."""
        return self._get_path(key).exists()

    async def list_objects(self, prefix: str = "") -> list[str]:
        """List files in local storage."""
        try:
            prefix_path = self.base_path / prefix if prefix else self.base_path
            if not prefix_path.exists():
                return []

            return [
                str(p.relative_to(self.base_path))
                for p in prefix_path.rglob("*")
                if p.is_file()
            ]
        except Exception as e:
            logger.error("Local list error", prefix=prefix, error=str(e))
            return []


def get_blob_storage() -> BaseBlobStorage:
    """
    Get the appropriate blob storage instance based on configuration.

    Returns S3 if configured, otherwise local storage.
    """
    if settings.aws_s3_bucket:
        return S3BlobStorage()
    return LocalBlobStorage()
