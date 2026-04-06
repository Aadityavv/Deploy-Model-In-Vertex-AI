"""Centralized configuration for Vertex AI deployment (Pydantic Settings)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# LogRecord attributes we do not merge into JSON (everything else is treated as structured extra)
_LOG_RECORD_STANDARD = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "taskName",
        "asctime",
    }
)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """
    JSON-style structured logging to stdout (works well with Cloud Logging agents).
    Pass structured fields via logger.info("msg", extra={"endpoint_id": "123"}).
    """
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger("vertex_deploy")

    handler = logging.StreamHandler(sys.stdout)

    class StructuredFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "severity": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            for key, value in record.__dict__.items():
                if key not in _LOG_RECORD_STANDARD:
                    payload[key] = value
            return json.dumps(payload, default=str)

    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    return logging.getLogger("vertex_deploy")


class Settings(BaseSettings):
    """
    Deploy an existing Vertex AI Model Registry entry to an Endpoint.

    Environment variables use UPPER_SNAKE_CASE matching each field name
    (pydantic-settings default), e.g. gcp_project_id -> GCP_PROJECT_ID.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- GCP core ---
    gcp_project_id: str
    gcp_region: str = "us-central1"

    # --- Optional staging (only needed if your workflow calls APIs that require it) ---
    gcs_bucket_name: Optional[str] = None
    """If set, default staging URI becomes gs://{name}/vertex-staging when STAGING_BUCKET_URI is unset."""

    staging_bucket_uri: Optional[str] = None
    """Full gs:// URI for Vertex staging. Overrides gcs_bucket_name-based default."""

    # --- Model Registry (existing model — no upload in this tool) ---
    model_display_name: str = ""
    """
    Display name of the model in Vertex Model Registry (exact match).
    Used when registry_model_resource_name is not set.
    """

    registry_model_resource_name: Optional[str] = None
    """
    Optional. Full resource name (projects/.../locations/.../models/ID), numeric model ID,
    or name with version: .../models/ID@version_or_alias. When set, display name lookup is skipped.
    """

    registry_model_version: Optional[str] = None
    """Optional version ID or alias (used with registry_model_resource_name if it has no @version)."""

    # --- Endpoint ---
    endpoint_display_name: str = "custom-llm-endpoint"
    deployed_model_display_name: str = "custom-llm-deployed"
    endpoint_id: Optional[str] = None
    """If set, deploy targets this endpoint; inference uses this ID."""

    # --- Hardware (defaults: CPU-only — no GPU quota required) ---
    use_gpu: bool = False
    """Set true only if you have GPU quota and a GPU-capable serving image."""
    machine_type: str = "n1-standard-8"
    accelerator_type: Optional[str] = None
    accelerator_count: int = Field(default=0, ge=0)

    min_replica_count: int = Field(default=1, ge=1)
    max_replica_count: int = Field(default=1, ge=1)

    # --- Deployment wait ---
    deploy_timeout_seconds: int = Field(
        default=3600,
        ge=1800,
        description="Minimum 1800s (30m) recommended for large container images / GCS artifact pulls.",
    )
    deploy_poll_interval_seconds: float = Field(default=30.0, ge=5.0)

    log_level: str = "INFO"

    @model_validator(mode="after")
    def require_model_identity(self) -> Settings:
        has_resource = bool(self.registry_model_resource_name and self.registry_model_resource_name.strip())
        has_display = bool(self.model_display_name and self.model_display_name.strip())
        if not has_resource and not has_display:
            raise ValueError(
                "Set MODEL_DISPLAY_NAME (registry display name) and/or REGISTRY_MODEL_RESOURCE_NAME "
                "(model ID or full resource name)."
            )
        return self

    @property
    def staging_uri(self) -> Optional[str]:
        if self.staging_bucket_uri:
            return self.staging_bucket_uri.rstrip("/")
        if self.gcs_bucket_name:
            return f"gs://{self.gcs_bucket_name}/vertex-staging"
        return None


def get_settings() -> Settings:
    return Settings()
