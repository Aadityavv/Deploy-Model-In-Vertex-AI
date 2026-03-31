"""
Call a deployed Vertex AI Endpoint with a sample payload (online prediction).

Run from repo root:  python -m src.inference

Requires ENDPOINT_ID in the environment (or .env). Shape of `instances` must match
your serving container contract (vLLM / TGI / custom often differ).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.api_core import exceptions as gcp_exceptions
from google.cloud import aiplatform

from src.config import configure_logging, get_settings


def default_sample_instances() -> list[dict]:
    """
    Generic placeholder; replace keys to match your model server's `/predict` schema.
    Many PyTorch samples use a single dict with pre-tokenized features or raw text.
    """
    return [
        {
            "prompt": "Explain structured logging in one sentence.",
            "max_tokens": 64,
            "temperature": 0.7,
        }
    ]


def main() -> int:
    settings = get_settings()
    log = configure_logging(settings.log_level)

    if not settings.endpoint_id:
        log.error(
            "ENDPOINT_ID is required for inference",
            extra={"hint": "Run deploy.py and set ENDPOINT_ID in .env"},
        )
        return 1

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
            "Permission denied — check Vertex AI User / predict IAM",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Failed to initialize Vertex AI client", extra={"error": str(e)})
        return 1

    endpoint = aiplatform.Endpoint(settings.endpoint_id)
    instances = default_sample_instances()
    parameters: dict = {}

    log.info(
        "Sending predict request",
        extra={
            "endpoint_id": settings.endpoint_id,
            "instance_count": len(instances),
        },
    )

    try:
        response = endpoint.predict(instances=instances, parameters=parameters)
    except gcp_exceptions.ResourceExhausted as e:
        log.error("Quota or rate limit exceeded", extra={"error": str(e)})
        return 1
    except gcp_exceptions.InvalidArgument as e:
        log.error(
            "Invalid payload for this endpoint — adjust instances to match the container",
            extra={"error": str(e)},
        )
        return 1
    except gcp_exceptions.GoogleAPICallError as e:
        log.error("Predict call failed", extra={"error": str(e)})
        return 1

    out = {
        "predictions": response.predictions,
        "deployed_model_id": response.deployed_model_id,
    }
    print(json.dumps(out, indent=2, default=str))
    log.info("Predict succeeded", extra={"deployed_model_id": response.deployed_model_id})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
