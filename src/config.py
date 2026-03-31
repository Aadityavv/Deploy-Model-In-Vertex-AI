"""Centralized configuration for Vertex AI deployment (Pydantic Settings)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from pydantic import Field, field_validator
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
    Environment variables use the same name as fields in UPPER_SNAKE_CASE
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
    gcs_bucket_name: str
    """Bucket name only (no gs:// prefix)."""

    gcs_model_artifact_prefix: str = "models/my-llm/1"
    """Path inside the bucket to model artifacts (folder prefix)."""

    staging_bucket_uri: Optional[str] = None
    """
    Optional gs:// URI for staging. Defaults to gs://{gcs_bucket_name}/vertex-staging
    if unset.
    """

    # --- Model registry ---
    model_display_name: str = "custom-llm"
    model_description: str = "Custom LLM uploaded from GCS"

    # --- Serving container (must match your image: vLLM / PyTorch / TGI) ---
    serving_container_image_uri: str
    serving_container_predict_route: str = "/predict"
    serving_container_health_route: str = "/health"
    serving_container_ports: list[int] = Field(default_factory=lambda: [8080])

    # --- Endpoint ---
    endpoint_display_name: str = "custom-llm-endpoint"
    deployed_model_display_name: str = "custom-llm-deployed"
    endpoint_id: Optional[str] = None
    """If set, deploy can target an existing endpoint; inference uses this ID."""

    # --- Hardware ---
    use_gpu: bool = True
    machine_type: str = "n1-standard-4"
    accelerator_type: Optional[str] = "NVIDIA_TESLA_T4"
    """Vertex accelerator enum, e.g. NVIDIA_TESLA_T4, NVIDIA_L4. Ignored if use_gpu is False."""
    accelerator_count: int = Field(default=1, ge=0)

    min_replica_count: int = Field(default=1, ge=1)
    max_replica_count: int = Field(default=1, ge=1)

    # --- Deployment wait ---
    deploy_timeout_seconds: int = Field(default=3600, ge=60)
    deploy_poll_interval_seconds: float = Field(default=30.0, ge=5.0)

    log_level: str = "INFO"

    @field_validator("serving_container_ports", mode="before")
    @classmethod
    def parse_ports(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",") if p.strip()]
        return v

    @property
    def artifact_uri(self) -> str:
        p = self.gcs_model_artifact_prefix.strip("/")
        return f"gs://{self.gcs_bucket_name}/{p}"

    @property
    def staging_uri(self) -> str:
        if self.staging_bucket_uri:
            return self.staging_bucket_uri.rstrip("/")
        return f"gs://{self.gcs_bucket_name}/vertex-staging"


def get_settings() -> Settings:
    return Settings()
