"""
Deploy an existing Vertex AI Model Registry model to an Endpoint.

Assumes the model is already registered (artifacts and container spec are on the Model resource).
Does not upload or create registry entries.

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


def _refresh_endpoint(endpoint: aiplatform.Endpoint) -> None:
    """
    Fetch latest endpoint state from the API.
    google-cloud-aiplatform Endpoint has no public reload(); sync uses _sync_gca_resource().
    """
    if hasattr(endpoint, "reload"):
        endpoint.reload()
    else:
        endpoint._sync_gca_resource()


def _model_sort_key(m: aiplatform.Model) -> tuple:
    """Prefer newest version when multiple registry entries share a display name."""
    try:
        t = m.version_create_time
        return (int(t.seconds), int(t.nanos))
    except Exception:
        return (0, 0)


def _find_models_by_display_name(display_name: str) -> list[aiplatform.Model]:
    return [m for m in aiplatform.Model.list() if m.display_name == display_name]


def _find_endpoint_by_display_name(display_name: str) -> Optional[aiplatform.Endpoint]:
    for e in aiplatform.Endpoint.list():
        if e.display_name == display_name:
            return e
    return None


def _resolve_registered_model(
    log,
    settings,
) -> aiplatform.Model:
    if settings.registry_model_resource_name and settings.registry_model_resource_name.strip():
        ref = settings.registry_model_resource_name.strip()
        log.info(
            "Loading registered model by resource name or ID",
            extra={"registry_model_ref": ref},
        )
        return aiplatform.Model(
            ref,
            version=settings.registry_model_version,
        )

    display = settings.model_display_name.strip()
    matches = _find_models_by_display_name(display)
    if not matches:
        raise LookupError(
            f"No Model found in registry with display_name={display!r} in project "
            f"{settings.gcp_project_id} region {settings.gcp_region}. "
            "Use REGISTRY_MODEL_RESOURCE_NAME if the display name differs or you need a specific resource."
        )

    if len(matches) > 1:
        matches.sort(key=_model_sort_key, reverse=True)
        log.warning(
            "Multiple models share this display name; deploying the newest version by version_create_time",
            extra={"display_name": display, "match_count": len(matches)},
        )

    chosen = matches[0]
    log.info(
        "Resolved model from registry by display name",
        extra={
            "display_name": display,
            "model_resource_name": chosen.resource_name,
        },
    )
    return chosen


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
        _refresh_endpoint(endpoint)
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
    try:
        settings = get_settings()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    log = configure_logging(settings.log_level)

    init_kwargs = {
        "project": settings.gcp_project_id,
        "location": settings.gcp_region,
    }
    if settings.staging_uri:
        init_kwargs["staging_bucket"] = settings.staging_uri

    try:
        aiplatform.init(**init_kwargs)
    except gcp_exceptions.Unauthenticated as e:
        log.error(
            "Authentication failed — run gcloud auth application-default login",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.Forbidden as e:
        log.error(
            "Permission denied — check IAM roles (e.g. Vertex AI User)",
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
            "staging_bucket": settings.staging_uri,
        },
    )

    try:
        model = _resolve_registered_model(log, settings)
    except LookupError as e:
        log.error(str(e))
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Failed to load model from registry", extra={"error": str(e)})
        return 1

    # --- Endpoint: get by display name, optional endpoint_id override ---
    endpoint: Optional[aiplatform.Endpoint] = None
    try:
        if settings.endpoint_id:
            endpoint = aiplatform.Endpoint(settings.endpoint_id)
            _refresh_endpoint(endpoint)
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
        # Route 100% traffic to the newly deployed revision ("0" = placeholder for this deploy request).
        "traffic_split": {"0": 100},
        # vLLM must match Model Registry container ports (8080). Not an Endpoint.deploy() argument —
        # popped before deploy(); confirm the registered Model spec lists this port.
        "serving_container_ports": [8080],
    }
    log.info(
        "Deploying CPU-only (no accelerators)",
        extra={"machine_type": settings.machine_type},
    )

    serving_ports = deploy_kwargs.pop("serving_container_ports", None)
    log.info(
        "Container ports (verify Model Registry model matches vLLM / Vertex expectations)",
        extra={"serving_container_ports": serving_ports},
    )

    try:
        deploy_operation = endpoint.deploy(**deploy_kwargs)
    except gcp_exceptions.ResourceExhausted as e:
        log.error("Quota exceeded during deploy", extra={"error": str(e)})
        return 1
    except gcp_exceptions.FailedPrecondition as e:
        log.error(
            "Deploy precondition failed (endpoint state, machine quota, or model cannot be deployed)",
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

    _refresh_endpoint(endpoint)
    endpoint_numeric_id = endpoint.resource_name.rsplit("/", 1)[-1]
    log.info(
        "Deploy finished",
        extra={
            "endpoint_id": endpoint_numeric_id,
            "endpoint_resource_name": endpoint.resource_name,
            "model_resource_name": model.resource_name,
        },
    )
    print(
        f"\nSet ENDPOINT_ID={endpoint_numeric_id} for inference (or add to .env).\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
