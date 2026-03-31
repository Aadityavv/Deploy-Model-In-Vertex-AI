"""
Upload a custom model from GCS to Vertex AI Model Registry and deploy to an Endpoint.

Run from repo root:  python -m src.deploy
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# Allow `python src/deploy.py` from repo root
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.api_core import exceptions as gcp_exceptions
from google.api_core import operation as gcp_operation
from google.cloud import aiplatform

from src.config import configure_logging, get_settings


def _find_model_by_display_name(display_name: str) -> Optional[aiplatform.Model]:
    for m in aiplatform.Model.list():
        if m.display_name == display_name:
            return m
    return None


def _find_endpoint_by_display_name(display_name: str) -> Optional[aiplatform.Endpoint]:
    for e in aiplatform.Endpoint.list():
        if e.display_name == display_name:
            return e
    return None


def _wait_for_deployment(
    logger,
    deploy_operation: Optional[gcp_operation.Operation],
    endpoint: aiplatform.Endpoint,
    timeout_seconds: int,
    poll_interval: float,
) -> None:
    """
    Wait for LRO completion when available; then poll until deployed models appear.
    Uses one overall wall-clock budget (timeout_seconds) for both phases.
    """
    overall_deadline = time.monotonic() + timeout_seconds

    if deploy_operation is not None:
        remaining = overall_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"No time left for deployment LRO (endpoint {endpoint.resource_name})."
            )
        try:
            deploy_operation.result(timeout=remaining)
        except gcp_exceptions.GoogleAPICallError as e:
            logger.error(
                "Deployment operation failed",
                extra={"error": str(e), "code": getattr(e, "code", None)},
            )
            raise
        logger.info(
            "Deployment LRO finished",
            extra={"endpoint_name": endpoint.resource_name},
        )

    deadline = overall_deadline
    while time.monotonic() < deadline:
        endpoint.reload()
        deployed = list(endpoint.list_models())
        if deployed:
            logger.info(
                "Endpoint has deployed models",
                extra={
                    "endpoint_resource_name": endpoint.resource_name,
                    "deployed_count": len(deployed),
                },
            )
            return
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Deployment did not become ready within {timeout_seconds}s "
        f"(endpoint {endpoint.resource_name})."
    )


def main() -> int:
    settings = get_settings()
    log = configure_logging(settings.log_level)

    try:
        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.gcp_region,
            staging_bucket=settings.staging_uri,
        )
    except gcp_exceptions.Unauthenticated as e:
        log.error(
            "Authentication failed — run gcloud auth application-default login",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.Forbidden as e:
        log.error(
            "Permission denied — check IAM roles (e.g. Vertex AI User, Storage)",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Failed to initialize Vertex AI client", extra={"error": str(e)})
        return 1

    log.info(
        "Vertex AI initialized",
        extra={
            "project": settings.gcp_project_id,
            "region": settings.gcp_region,
            "artifact_uri": settings.artifact_uri,
        },
    )

    # --- Model: reuse if same display name exists ---
    model: Optional[aiplatform.Model] = None
    try:
        existing = _find_model_by_display_name(settings.model_display_name)
        if existing is not None:
            log.info(
                "Model already in registry; skipping upload",
                extra={"model_resource_name": existing.resource_name},
            )
            model = existing
        else:
            log.info("Uploading model from GCS to Model Registry")
            model = aiplatform.Model.upload(
                display_name=settings.model_display_name,
                description=settings.model_description,
                artifact_uri=settings.artifact_uri,
                serving_container_image_uri=settings.serving_container_image_uri,
                serving_container_predict_route=settings.serving_container_predict_route,
                serving_container_health_route=settings.serving_container_health_route,
                serving_container_ports=settings.serving_container_ports,
                sync=True,
            )
            log.info(
                "Model upload complete",
                extra={"model_resource_name": model.resource_name},
            )
    except gcp_exceptions.ResourceExhausted as e:
        log.error("Quota exceeded", extra={"error": str(e)})
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Model upload or registry check failed", extra={"error": str(e)})
        return 1

    assert model is not None

    # --- Endpoint: get by display name, optional endpoint_id override ---
    endpoint: Optional[aiplatform.Endpoint] = None
    try:
        if settings.endpoint_id:
            endpoint = aiplatform.Endpoint(settings.endpoint_id)
            endpoint.reload()
            log.info(
                "Using existing endpoint from ENDPOINT_ID",
                extra={"endpoint_id": settings.endpoint_id, "resource_name": endpoint.resource_name},
            )
        else:
            existing_ep = _find_endpoint_by_display_name(settings.endpoint_display_name)
            if existing_ep is not None:
                endpoint = existing_ep
                log.info(
                    "Reusing endpoint with matching display name",
                    extra={"resource_name": endpoint.resource_name},
                )
            else:
                log.info("Creating new endpoint", extra={"display_name": settings.endpoint_display_name})
                endpoint = aiplatform.Endpoint.create(
                    display_name=settings.endpoint_display_name,
                    sync=True,
                )
                log.info("Endpoint created", extra={"resource_name": endpoint.resource_name})
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Endpoint create/lookup failed", extra={"error": str(e)})
        return 1

    assert endpoint is not None

    deploy_kwargs: dict = {
        "model": model,
        "deployed_model_display_name": settings.deployed_model_display_name,
        "machine_type": settings.machine_type,
        "min_replica_count": settings.min_replica_count,
        "max_replica_count": settings.max_replica_count,
        "sync": False,
    }
    if settings.use_gpu and settings.accelerator_type and settings.accelerator_count > 0:
        deploy_kwargs["accelerator_type"] = settings.accelerator_type
        deploy_kwargs["accelerator_count"] = settings.accelerator_count
        log.info(
            "Deploying with GPU",
            extra={
                "machine_type": settings.machine_type,
                "accelerator_type": settings.accelerator_type,
                "accelerator_count": settings.accelerator_count,
            },
        )
    else:
        log.info(
            "Deploying CPU-only",
            extra={"machine_type": settings.machine_type},
        )

    try:
        deploy_operation = endpoint.deploy(**deploy_kwargs)
    except gcp_exceptions.ResourceExhausted as e:
        log.error("Quota exceeded during deploy", extra={"error": str(e)})
        return 1
    except gcp_exceptions.FailedPrecondition as e:
        log.error(
            "Deploy precondition failed (endpoint state, machine quota, or image)",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Deploy request failed", extra={"error": str(e)})
        return 1

    try:
        _wait_for_deployment(
            log,
            deploy_operation,
            endpoint,
            settings.deploy_timeout_seconds,
            settings.deploy_poll_interval_seconds,
        )
    except TimeoutError as e:
        log.error(str(e))
        return 1

    endpoint.reload()
    endpoint_numeric_id = endpoint.resource_name.rsplit("/", 1)[-1]
    log.info(
        "Deploy finished",
        extra={
            "endpoint_id": endpoint_numeric_id,
            "endpoint_resource_name": endpoint.resource_name,
        },
    )
    print(
        f"\nSet ENDPOINT_ID={endpoint_numeric_id} for inference (or add to .env).\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
