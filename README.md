# Vertex AI custom LLM deployment

Python tooling to register a model from Google Cloud Storage on **Vertex AI Model Registry**, deploy it to an **Endpoint**, and run a test prediction via the SDK—no Console UI required.

---

## Prerequisites

- **Python 3.10+** (3.12 works with the pinned dependencies).
- **Google Cloud SDK** (`gcloud`) installed and on your `PATH`.
- A **GCP project** with the **Vertex AI API** enabled and billing active.
- **IAM**: your principal needs roles such as Vertex AI User and access to the GCS bucket (e.g. Storage Object Viewer on artifacts, and ability to write to the staging prefix if needed).
- **Model weights** already uploaded under a prefix in a GCS bucket (e.g. `gs://your-bucket/models/my-llm/v1/...`).
- A **serving container image** that matches how you run inference (Vertex prebuilt PyTorch images, or your own vLLM/TGI image). Routes and ports in `.env` must match that image.

---

## 1. Clone or open the project

Use this folder as the project root (the directory that contains `requirements.txt` and `src/`).

All commands below assume the **current working directory is the project root**.

---

## 2. Create the virtual environment and install dependencies

### Option A — Bash (Linux, macOS, Git Bash, WSL)

```bash
bash setup_env.sh
```

This creates `.venv`, installs `requirements.txt`, runs `gcloud auth application-default login`, and optionally sets the default project if `GCP_PROJECT_ID` is exported before running the script.

After it finishes, activate the venv:

```bash
source .venv/bin/activate
```

### Option B — Windows PowerShell

If you do not use Bash, run the equivalent steps manually:

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

## 3. Configure environment variables

Create a file named **`.env`** in the project root (same folder as `requirements.txt`). The app loads it automatically via `pydantic-settings`.

### Required

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Your GCP project ID. |
| `GCS_BUCKET_NAME` | Bucket name only (no `gs://`). |
| `SERVING_CONTAINER_IMAGE_URI` | Full image URI for prediction (must match your region and serving stack). |

### Strongly recommended

| Variable | Description |
|----------|-------------|
| `GCP_REGION` | Vertex region (default: `us-central1`). |
| `GCS_MODEL_ARTIFACT_PREFIX` | Path inside the bucket to the model folder (default: `models/my-llm/1`). Resolved as `gs://{GCS_BUCKET_NAME}/{prefix}`. |

### Serving container (must match your image)

| Variable | Default |
|----------|---------|
| `SERVING_CONTAINER_PREDICT_ROUTE` | `/predict` |
| `SERVING_CONTAINER_HEALTH_ROUTE` | `/health` |
| `SERVING_CONTAINER_PORTS` | `8080` (comma-separated for multiple) |

### Endpoint naming

| Variable | Default |
|----------|---------|
| `MODEL_DISPLAY_NAME` | `custom-llm` |
| `ENDPOINT_DISPLAY_NAME` | `custom-llm-endpoint` |
| `DEPLOYED_MODEL_DISPLAY_NAME` | `custom-llm-deployed` |
| `ENDPOINT_ID` | Empty until after first deploy; set for inference and to deploy onto an existing endpoint. |

### Hardware

| Variable | Default | Notes |
|----------|---------|--------|
| `USE_GPU` | `true` | Set to `false` for CPU-only. |
| `MACHINE_TYPE` | `n1-standard-4` | Must be compatible with chosen accelerators. |
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

| Variable | Description |
|----------|-------------|
| `STAGING_BUCKET_URI` | If unset, defaults to `gs://{GCS_BUCKET_NAME}/vertex-staging`. |

### Example `.env`

```env
GCP_PROJECT_ID=my-project
GCP_REGION=us-central1
GCS_BUCKET_NAME=my-model-bucket
GCS_MODEL_ARTIFACT_PREFIX=models/my-llm/v1
SERVING_CONTAINER_IMAGE_URI=us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-2:latest

USE_GPU=true
MACHINE_TYPE=n1-standard-4
ACCELERATOR_TYPE=NVIDIA_TESLA_T4
ACCELERATOR_COUNT=1
```

For CPU-only deployment, set `USE_GPU=false` and pick a CPU-appropriate `MACHINE_TYPE` and CPU serving image.

---

## 4. Deploy the model

With the venv activated and `.env` in place, from the project root:

```bash
python -m src.deploy
```

What this does in order:

1. Initializes the Vertex AI client for your project and region.
2. Looks for an existing model with `MODEL_DISPLAY_NAME`; if found, skips upload.
3. Otherwise uploads from `gs://{GCS_BUCKET_NAME}/{GCS_MODEL_ARTIFACT_PREFIX}` using your serving container settings.
4. Reuses an endpoint named `ENDPOINT_DISPLAY_NAME`, uses `ENDPOINT_ID` if set, or creates a new endpoint.
5. Deploys the model and waits until deployed models appear or the timeout is reached.

On success, the script prints a line to **stderr** telling you to set **`ENDPOINT_ID`**—add that value to `.env`:

```env
ENDPOINT_ID=1234567890123456789
```

Logs are JSON lines on stdout (severity, message, and any `extra` fields).

---

## 5. Run a test prediction

After `ENDPOINT_ID` is set in `.env`:

```bash
python -m src.inference
```

The sample payload is defined in `src/inference.py` (`default_sample_instances`). **You must change it** to match your container’s expected JSON schema (vLLM/TGI/PyTorch samples differ).

---

## 6. Quick command reference

| Step | Command |
|------|---------|
| Install + ADC (Bash) | `bash setup_env.sh` |
| Activate venv (Bash) | `source .venv/bin/activate` |
| Activate venv (Windows CMD) | `.venv\Scripts\activate.bat` |
| Activate venv (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Deploy | `python -m src.deploy` |
| Predict | `python -m src.inference` |

---

## Troubleshooting

- **`Unauthenticated` / ADC errors**  
  Run `gcloud auth application-default login` again. The Python SDK uses Application Default Credentials, not only `gcloud auth login`.

- **`Permission denied` / IAM**  
  Confirm the active account has Vertex AI and GCS access on the project and bucket.

- **`Quota exceeded`**  
  Request quota for the chosen `MACHINE_TYPE` and `ACCELERATOR_TYPE` in that region, or switch to smaller machine/accelerator counts.

- **`InvalidArgument` on predict**  
  Your `instances` payload does not match the serving container. Align with the image’s HTTP API (routes and body shape).

- **Windows: `pip install` fails with “file is being used”**  
  Close editors or processes locking `.venv`, then run `pip install -r requirements.txt` again.

- **Path with spaces**  
  Quote paths in shells: `cd "C:\Users\...\GCP Vertex AI"`.

---

## Further reading

- [Vertex AI prebuilt prediction containers](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers) — choose an image URI that matches your framework and GPU/CPU target.
