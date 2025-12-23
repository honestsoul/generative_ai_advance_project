"""
Application settings using Pydantic Settings.

Supports loading from environment variables and .env files.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug: bool = Field(default=False)

    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None)
    openai_org_id: str | None = Field(default=None)
    openai_default_model: str = Field(default="gpt-4o")

    # Anthropic
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_default_model: str = Field(default="claude-3-5-sonnet-20241022")

    # AWS Bedrock (Optional)
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_region: str = Field(default="us-east-1")

    # Cohere (Optional)
    cohere_api_key: SecretStr | None = Field(default=None)

    # Redis (Optional)
    redis_url: str | None = Field(default=None)

    # S3/GCS (Optional)
    aws_s3_bucket: str | None = Field(default=None)
    gcs_bucket: str | None = Field(default=None)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)

    # Telemetry
    otel_exporter_otlp_endpoint: str | None = Field(default=None)
    otel_service_name: str = Field(default="genai-project")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
