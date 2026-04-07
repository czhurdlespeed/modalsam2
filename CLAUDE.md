# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Modal Labs training backend for SAM2 fine-tuning. It wraps Meta's SAM 2 repository (in `sam2/`) with a Modal serverless deployment layer (in `src/`) that exposes training and cancellation as FastAPI endpoints.

## Common Commands

```bash
uv sync                          # Install dependencies (uses uv, not pip)
modal deploy src/main.py         # Deploy to Modal Labs
modal serve src/main.py          # Run locally with Modal dev server

# Test endpoints manually
bash curltrain.sh                # Submit a training job
bash curlcancel.sh               # Cancel a running job

# SAM2 training (local, outside Modal)
cd sam2 && pip install -e ".[dev]"
cd sam2/checkpoints && sh download_ckpts.sh
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 0 --num-gpus 1
```

## Architecture

### Two-layer structure

- **`src/`** — Modal deployment layer. Defines the Modal app, container images, volumes, secrets, and FastAPI endpoints.
- **`sam2/`** — Fork of Meta's SAM 2 (branch `Modal` from `ORNLxUTK/sam2`). Contains the actual training code, model definitions, dataset loaders, and Hydra configs.

### Request flow

1. Frontend POSTs `UserSelections` to the `train` endpoint (`src/main.py`)
2. The lightweight FastAPI function adds the job to `job_queue` (Modal Dict), then calls `SAM2Training.launch_training` on a GPU-equipped class
3. `SAM2Training` is a `modal.Cls` — GPU type is dynamically selected via `with_options(gpu=...)` based on model size and finetune mode (see `config.py:gpu_type`)
4. Training streams log lines back as SSE via `yield from single_node_runner()`
5. After training: checkpoint is zipped, uploaded to R2 via `CloudBucket`, volume cleaned up, job removed from queue

### GPU selection logic (`src/config.py`)

- Full finetune → always A100-80GB
- LoRA + tiny/small → L40S
- LoRA + base_plus/large → A100-80GB

### Config resolution (`src/config.py`)

`UserSelections` (Pydantic model) maps user choices to:
- A Hydra YAML template: `sam2/sam2/configs/sam2.1_training/{DATASET}_{FT|LoRA}_{size}.yaml`
- Volume paths for dataset images/annotations: `/data/SAM2images/{dataset}/`
- Checkpoint path: `sam2/checkpoints/sam2.1_hiera_{size}.pt`

The `create_cfg()` function loads the YAML via OmegaConf and overrides paths, epochs, LoRA rank, etc. from user selections.

### Modal infrastructure

- **Volumes**: `sam2_input_data` (training datasets, read-only), `sam2_checkpoints` (training output, ephemeral)
- **Secrets**: `r2_secret` (Cloudflare R2 creds), `logfire` (observability token)
- **Images**: `SAM2_BASE_IMAGE` (PyTorch+CUDA+SAM2, heavy), `FASTAPI_LIGHTWEIGHT_IMAGE` (Python 3.13, minimal)
- **Dict**: `job-queue` — tracks running jobs by `{user_id}_{job_id}` for cancellation support

### Hydra training configs

Located in `sam2/sam2/configs/sam2.1_training/`. Named `{DATASET}_{FT|LoRA}_{size}.yaml`. These define the full training pipeline: data transforms, model architecture, optimizer, loss functions, and distributed settings. Key `scratch` params: `resolution`, `batch_size`, `num_epochs`, `base_lr`.
