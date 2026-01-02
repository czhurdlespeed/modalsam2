import asyncio
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import aioboto3
import modal
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


class CloudBucket:
    def __init__(self, bucket_name: str, max_concurrent: int = 10):
        """
        Initialize CloudBucket with async S3/R2 client.

        Args:
            bucket_name: Name of the R2 bucket
            max_concurrent: Maximum number of concurrent uploads/downloads
        """
        if not bucket_name:
            raise ValueError("Bucket name is required")
        self.bucket_name = bucket_name
        self.max_concurrent = max_concurrent
        self.endpoint_url = (
            f"https://{os.getenv('CF_R2_ACCOUNTID')}.r2.cloudflarestorage.com"
        )
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.config = Config(signature_version="s3v4")
        self.session = aioboto3.Session()

    async def _upload_file(self, s3_client, file_path: Path, s3_key: str):
        """Upload a single file asynchronously"""
        try:
            print(f"Uploading {file_path} → {s3_key}")
            await s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
        except Exception as e:
            raise RuntimeError(f"Failed to upload {file_path} to {s3_key}: {e}")

    async def upload_directory(
        self, local_directory: Path, s3_prefix: str = "", max_concurrent: int = None
    ):
        """
        Upload a local directory to S3/R2 asynchronously, preserving directory structure.

        Args:
            local_directory: Path to local directory
            s3_prefix: Optional prefix in bucket
            max_concurrent: Override default max concurrent uploads
        """
        local_directory = Path(local_directory)
        if not local_directory.exists():
            raise ValueError(f"Local directory does not exist: {local_directory}")

        # Get all files to upload
        files = [f for f in local_directory.rglob("*") if f.is_file()]
        if not files:
            print("No files to upload")
            return

        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_upload(s3_client, file_path: Path, s3_key: str):
            async with semaphore:
                await self._upload_file(s3_client, file_path, s3_key)

        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="auto",
            config=self.config,
        ) as s3:
            # Create upload tasks
            tasks = []
            for file_path in files:
                relative_path = file_path.relative_to(local_directory)
                s3_key = str(Path(s3_prefix) / relative_path).replace("\\", "/")
                tasks.append(bounded_upload(s3, file_path, s3_key))

            # Execute all uploads concurrently
            await asyncio.gather(*tasks)
            print(f"✅ All {len(files)} uploads complete!")

    async def _download_file(self, s3_client, s3_key: str, local_path: Path):
        """Download a single file asynchronously"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await s3_client.download_file(self.bucket_name, s3_key, str(local_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download {s3_key} to {local_path}: {e}")

    async def download_directory(
        self, s3_prefix: str, local_directory: Path, max_concurrent: int = None
    ):
        """
        Download all objects with the given prefix from S3/R2 to a local directory.

        Args:
            s3_prefix: S3 prefix to filter objects
            local_directory: Local directory to save zip file
            max_concurrent: Override default max concurrent downloads
        """
        local_directory = Path(local_directory)
        local_directory.mkdir(parents=True, exist_ok=True)

        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_download(s3_client, s3_key: str, local_path: Path):
            async with semaphore:
                await self._download_file(s3_client, s3_key, local_path)

        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="auto",
            config=self.config,
        ) as s3:
            # List all objects with pagination
            paginator = s3.get_paginator("list_objects_v2")
            download_tasks = []

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Collect all download tasks
                async for page in paginator.paginate(
                    Bucket=self.bucket_name, Prefix=s3_prefix
                ):
                    if "Contents" not in page:
                        continue
                    for obj in page["Contents"]:
                        s3_key = obj["Key"]
                        relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                        if not relative_path:
                            continue

                        local_path = temp_path / relative_path
                        download_tasks.append(bounded_download(s3, s3_key, local_path))

                # Execute all downloads concurrently
                if download_tasks:
                    await asyncio.gather(*download_tasks)
                    print(f"✅ All {len(download_tasks)} downloads complete!")

                # Create zip file
                zip_path = local_directory / "checkpoint.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
                    for file_path in temp_path.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_path)
                            zip_ref.write(file_path, arcname)

                print(f"Created zip file: {zip_path}")


sam2_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "SAM2_BUILD_ALLOW_ERRORS": "0",
            "CUDA_HOME": "/usr/local/cuda",
            "UV_SYSTEM_PYTHON": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0 7.2 7.5 8.0 8.6 8.9 9.0",
            "PYTHONPATH": "/sam2modalwebapp/.venv/lib/python3.11/site-packages:$PYTHONPATH",
        },
    )
    .apt_install(
        "ffmpeg",
        "libavutil-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "pkg-config",
        "build-essential",
        "g++-11",
        "gcc-11",
        "libffi-dev",
        "curl",
        "ca-certificates",
        "wget",
        "git",
    )
    .workdir("/sam2modalwebapp")
    .run_commands("git clone -b Modal https://github.com/ORNLxUTK/sam2.git")
    .workdir("/sam2modalwebapp/sam2")
    .run_commands("pip install -e '.[dev]'")
    .workdir("/sam2modalwebapp/sam2/checkpoints")
    .run_commands("sh download_ckpts.sh")
    .workdir("/sam2modalwebapp")
    .add_local_dir("sam2/", remote_path="/sam2modalwebapp/sam2/")
)


app = modal.App(name="sam2modalwebapp")

r2_secret = modal.Secret.from_name(
    "r2_secret",
    required_keys=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CF_R2_ACCOUNTID",
        "CF_R2_BUCKET_NAME",
    ],
)

data_volume = modal.Volume.from_name("sam2_input_data")
checkpoint_volume = modal.Volume.from_name(
    "sam2_checkpoints", create_if_missing=True, version=1
)


@app.function(
    image=sam2_image,
    gpu="L40s",  # Single GPU - adjust as needed (A10G, A100, T4, etc.)
    volumes={
        "/data": data_volume,
        "/trainingresults": checkpoint_volume,
    },
    secrets=[r2_secret],
    timeout=7200,  # 2 hours - adjust as needed
)
async def train_sam2(
    config_path: str = "sam2.1_training/MAZAK_LoRA4_tiny",
    experiment_name: str = "sam2_training_run1",
    gpu: str = "L40s",
):
    """
    Train SAM2 on Modal with single GPU.

    Args:
        config_path: Config name (without .yaml extension) relative to sam2/sam2/configs/
                    e.g., "sam2.1_training/MAZAK_LoRA4_tiny"
        experiment_name: Name for the experiment (used for log/checkpoint dirs)
    """
    import logging
    import os

    from hydra import compose, initialize_config_module
    from iopath.common.file_io import g_pathmgr
    from omegaconf import OmegaConf
    from training.train import single_node_runner
    from training.utils.train_utils import makedir, register_omegaconf_resolvers

    initialize_config_module("sam2/sam2", version_base="1.2")
    register_omegaconf_resolvers()

    logging.basicConfig(level=logging.INFO)
    cfg = compose(config_name=config_path)
    experiment_dir = f"/trainingresults/{experiment_name}"
    makedir(experiment_dir)
    cfg.launcher.experiment_log_dir = experiment_dir
    cfg.launcher.num_nodes = 1
    num_gpus = 1 if ":" not in gpu else int(gpu.split(":")[1])
    cfg.launcher.gpus_per_node = num_gpus
    cfg.submitit.use_cluster = False

    cfg.dataset.img_folder = "/data/SAM2images/MAZAK/JPEGImages/train"
    cfg.dataset.gt_folder = "/data/SAM2images/MAZAK/Annotations/train"
    cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = (
        "/sam2modalwebapp/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    )
    cfg.scratch.num_epochs = 1

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    main_port = 29500
    single_node_runner(cfg, main_port)

    checkpoint_volume.commit()
    logging.info("Checkpoints committed to volume")

    logging.info(f"Copying {experiment_dir} to R2 bucket")
    cloudbucket = CloudBucket(bucket_name=os.getenv("CF_R2_BUCKET_NAME"))
    await cloudbucket.upload_directory(Path(experiment_dir), f"{experiment_name}/")

    logging.info(
        f"Successfully copied {experiment_dir} to {os.getenv('CF_R2_BUCKET_NAME')}"
    )
    shutil.rmtree(f"/trainingresults/{experiment_name}")
    logging.info(f"Successfully removed {experiment_dir}")


@app.local_entrypoint()
def main(
    config: str = "configs/sam2.1_training/MAZAK_LoRA4_tiny",
    experiment_name: str = "sam2_training_run1",
):
    """
    Local entrypoint to launch training.

    Usage:
        modal run main.py --config "configs/sam2.1_training/MAZAK_LoRA4_tiny" --experiment-name "my_run"
    """
    train_sam2.remote(config, experiment_name)
