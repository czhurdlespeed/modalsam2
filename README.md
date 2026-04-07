# modalsam2

Modal Labs deployment layer for fine-tuning [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2) on custom video object segmentation datasets. Exposes training and job cancellation as serverless GPU-backed FastAPI endpoints.

This repo is part of the [SAM2 Fine-tuning](https://github.com/czhurdlespeed/modalsam2) project — a full-stack app where users configure training parameters in a web UI and jobs run on Modal's serverless GPUs.

## How it works

```
Frontend (Next.js)
    │
    ▼  POST /train (UserSelections JSON)
┌─────────────────────────────┐
│  FastAPI endpoint (lightweight)  │
│  - Validates input               │
│  - Adds job to queue             │
│  - Selects GPU based on config   │
└──────────┬──────────────────┘
           │  modal.Cls.with_options(gpu=...)
           ▼
┌─────────────────────────────┐
│  SAM2Training (GPU container)    │
│  - Builds Hydra config           │
│  - Runs single_node_runner()     │
│  - Streams logs via SSE          │
│  - Zips & uploads checkpoint     │
│    to Cloudflare R2              │
└─────────────────────────────┘
```

Training logs are streamed back to the frontend in real-time via Server-Sent Events. Checkpoints are uploaded to Cloudflare R2 on completion and cleaned from the Modal volume.

## Setup

**Prerequisites**: Python 3.10+, [uv](https://docs.astral.sh/uv/), [Modal CLI](https://modal.com/docs/guide)

```bash
# Install dependencies
uv sync

# Authenticate with Modal (first time)
modal token set

# Upload training datasets to Modal volume
python src/createvolume.py
```

### Required Modal Secrets

Configure these in the [Modal dashboard](https://modal.com/secrets):

- **`r2_secret`**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CF_R2_ACCOUNTID`, `CF_R2_BUCKET_NAME`
- **`logfire`**: `LOGFIRE_TOKEN`

## Deployment

```bash
# Deploy to Modal (production)
modal deploy src/main.py

# Run with hot-reload (development)
modal serve src/main.py
```

## Endpoints

### `POST /train`

Starts a training job. Requires Modal proxy auth headers.

**Request body** (`UserSelections`):
```json
{
  "userjob": { "user_id": "alice", "job_id": 1 },
  "fullfinetune": false,
  "lora_rank": 4,
  "base_model": "base_plus",
  "dataset": "irPOLYMER",
  "num_epochs": 10
}
```

| Field | Options |
|-------|---------|
| `base_model` | `tiny`, `small`, `base_plus`, `large` |
| `lora_rank` | `2`, `4`, `8`, `16`, `32` (null if fullfinetune) |
| `dataset` | `MAZAK`, `irPOLYMER`, `visPOLYMER`, `TIG` |
| `num_epochs` | 1–100 |

Returns an SSE stream of training log lines.

### `POST /cancel_job?user_plus_job_id={user}_{job}`

Cancels a running job, terminates the GPU container, and cleans up the checkpoint volume.

## GPU Selection

GPU type is automatically selected based on model size and training mode:

| Mode | Model Size | GPU |
|------|-----------|-----|
| Full finetune | Any | A100-80GB |
| LoRA | tiny, small | L40S |
| LoRA | base_plus, large | A100-80GB |

## Project Structure

```
src/
  main.py            # Modal app, FastAPI endpoints, SAM2Training class
  config.py          # UserSelections, ModelYamlConfig, Hydra config builder
  cloud.py           # CloudBucket — async R2 upload via aioboto3
  containerimages.py # Modal container image definitions
  createvolume.py    # One-time script to upload datasets to Modal volume

sam2/                # Fork of Meta's SAM 2 (branch: Modal)
  training/
    train.py         # Training entry point (single/multi-node)
    trainer.py       # Main training loop
    model/sam2.py    # SAM2Train model with LoRA support
    dataset/         # VOS dataset loaders and transforms
    loss_fns.py      # MultiStepMultiMasksAndIous loss
  sam2/
    configs/sam2.1_training/  # Hydra YAML configs ({DATASET}_{FT|LoRA}_{size}.yaml)
    modeling/                 # SAM 2 model architecture
  checkpoints/               # Pre-trained model weights
```

## Testing Locally

Test endpoints against your dev Modal app using curl:

```bash
# Submit a training job
curl --request POST \
  --header "Modal-Key: <your-key>" \
  --header "Modal-Secret: <your-secret>" \
  --header "Content-Type: application/json" \
  --data '{"userjob":{"user_id":"alice","job_id":1},"fullfinetune":false,"lora_rank":4,"base_model":"base_plus","dataset":"irPOLYMER","num_epochs":2}' \
  --no-buffer \
  https://<your-workspace>--sam2modalwebapp-train-dev.modal.run

# Cancel a running job
curl --request POST \
  --header "Modal-Key: <your-key>" \
  --header "Modal-Secret: <your-secret>" \
  "https://<your-workspace>--sam2modalwebapp-cancel-job-dev.modal.run?user_plus_job_id=alice_1"
```
