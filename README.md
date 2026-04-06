# Vertex AI — deploy a registered model

This project **only deploys models that already exist in Vertex AI Model Registry** (you or another process already uploaded artifacts and registered the model). You provide your **GCP project**, **region**, and **how to find that model** (display name or resource ID). The code creates or reuses an **Endpoint** and deploys the model with your chosen machine/GPU settings—no Console UI required.

---

## Prerequisites

- **Python 3.10+**
- **Google Cloud SDK** (`gcloud`) on your `PATH`
- A **GCP project** with **Vertex AI API** enabled
- **IAM**: principal can read the model, manage endpoints, and deploy (e.g. Vertex AI User / appropriate custom role)
- **Model already registered** in Model Registry for the same project and region you configure (container spec and artifacts are on that Model resource)

---

## 1. Open the project

Use the folder that contains `requirements.txt` and `src/` as the **project root**. All commands below run from there.

---

## 2. Virtual environment and dependencies

### Bash (Linux, macOS, Git Bash, WSL)

```bash
bash setup_env.sh
source .venv/bin/activate
```

### Windows PowerShell

```powershell
cd "path\to\GCP Vertex AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

---

## 3. Configure `.env`

Create **`.env`** in the project root.

### Required

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Your GCP project ID. |

### Identify the registered model (at least one required)

| Variable | Description |
|----------|-------------|
| `MODEL_DISPLAY_NAME` | Exact **display name** of the model in Model Registry (used if `REGISTRY_MODEL_RESOURCE_NAME` is not set). |
| `REGISTRY_MODEL_RESOURCE_NAME` | Optional. **Numeric model ID**, full resource name (`projects/.../locations/.../models/123`), or include a version: `.../models/123@your-alias`. Skips listing by display name. |
| `REGISTRY_MODEL_VERSION` | Optional. Version ID or alias when using resource name **without** `@version` in the string. |

If several models share the same display name, deploy uses the **newest** by `version_create_time` and logs a warning.

### Region and endpoint

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_REGION` | `us-central1` | Must match where the **model** is registered. |
| `ENDPOINT_DISPLAY_NAME` | `custom-llm-endpoint` | Used to find or create an endpoint when `ENDPOINT_ID` is unset. |
| `DEPLOYED_MODEL_DISPLAY_NAME` | `custom-llm-deployed` | Name of this deployment on the endpoint. |
| `ENDPOINT_ID` | *(empty)* | If set, deploy attaches to this endpoint instead of searching by `ENDPOINT_DISPLAY_NAME`. After first deploy, set this for inference. |

### Hardware

| Variable | Default | Notes |
|----------|---------|--------|
| `USE_GPU` | `true` | `false` for CPU-only. |
| `MACHINE_TYPE` | `n1-standard-4` | Compatible with accelerators / workload. |
| `ACCELERATOR_TYPE` | `NVIDIA_TESLA_T4` | Ignored when `USE_GPU=false`. |
| `ACCELERATOR_COUNT` | `1` | |
| `MIN_REPLICA_COUNT` / `MAX_REPLICA_COUNT` | `1` | |

### Timeouts and logging

| Variable | Default |
|----------|---------|
| `DEPLOY_TIMEOUT_SECONDS` | `3600` |
| `DEPLOY_POLL_INTERVAL_SECONDS` | `30` |
| `LOG_LEVEL` | `INFO` |

### Optional staging bucket

Some Vertex workflows expect a **staging** GCS URI in `aiplatform.init`. If you omit both of the following, init runs **without** `staging_bucket` (fine for deploy-only in many cases).

| Variable | Description |
|----------|-------------|
| `STAGING_BUCKET_URI` | Full `gs://...` staging prefix. |
| `GCS_BUCKET_NAME` | If set (and `STAGING_BUCKET_URI` unset), staging defaults to `gs://{GCS_BUCKET_NAME}/vertex-staging`. |

### Example `.env` (by display name)

```env
GCP_PROJECT_ID=my-project
GCP_REGION=us-central1
MODEL_DISPLAY_NAME=my-registered-llm

USE_GPU=true
MACHINE_TYPE=n1-standard-4
ACCELERATOR_TYPE=NVIDIA_TESLA_T4
ACCELERATOR_COUNT=1
```

### Example `.env` (by model resource ID)

```env
GCP_PROJECT_ID=my-project
GCP_REGION=us-central1
REGISTRY_MODEL_RESOURCE_NAME=1234567890123456789
```

---

## 4. Deploy

```bash
python -m src.deploy
```

What happens:

1. Initializes the Vertex AI SDK for `GCP_PROJECT_ID` and `GCP_REGION`.
2. Loads the model from Model Registry (**by resource name/ID, or by listing and matching `MODEL_DISPLAY_NAME`**). **No upload.**
3. Uses `ENDPOINT_ID` if set; otherwise finds or creates an endpoint named `ENDPOINT_DISPLAY_NAME`.
4. Calls `endpoint.deploy(...)` with your machine/accelerator settings and waits (LRO + polling) until deployed models appear or timeout.

On success, stderr reminds you to set **`ENDPOINT_ID`** for prediction:

```env
ENDPOINT_ID=1234567890123456789
```

Logs are JSON lines on stdout.

---

## 5. Test prediction

After `ENDPOINT_ID` is in `.env`:

```bash
python -m src.inference
```

Adjust `default_sample_instances()` in `src/inference.py` so the payload matches your model’s `/predict` contract.

---

## 6. Command reference

| Step | Command |
|------|---------|
| Setup (Bash) | `bash setup_env.sh` |
| Activate (Bash) | `source .venv/bin/activate` |
| Activate (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Deploy | `python -m src.deploy` |
| Predict | `python -m src.inference` |

---

## Troubleshooting

- **`Configuration error: Set MODEL_DISPLAY_NAME...`** — Provide either `MODEL_DISPLAY_NAME` or `REGISTRY_MODEL_RESOURCE_NAME` in `.env`.
- **`No Model found in registry with display_name=...`** — Name must match exactly, or use `REGISTRY_MODEL_RESOURCE_NAME`.
- **`Unauthenticated`** — Run `gcloud auth application-default login`.
- **`Permission denied`** — Vertex AI / endpoint permissions on the project.
- **`Quota exceeded`** — Quota for `MACHINE_TYPE` / accelerators in that region.
- **`FailedPrecondition` on deploy** — Model may not support online deployment, or endpoint/state conflicts; see error message in Cloud Logging.
- **`InvalidArgument` on predict** — Payload does not match the serving container.

---

## Further reading

- [Deploy a model to an endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment) — Vertex AI deployment overview.
