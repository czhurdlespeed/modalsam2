import os
import shutil
import zipfile
from pathlib import Path

import modal
from dotenv import load_dotenv

from .baseimage import BASE_IMAGE
from .cloud import CloudBucket

load_dotenv()


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
    image=BASE_IMAGE,
    gpu="L40s",
    volumes={
        "/data": data_volume,
        "/trainingresults": checkpoint_volume,
    },
    secrets=[r2_secret],
    timeout=7200,
)
async def train(
    config_path: str = "sam2.1_training/MAZAK_LoRA4_tiny",
    experiment_name: str = "sam2_training_run1",
    gpu: str = "L40s",
):
    """
    Args:
        config_path: Config name (without .yaml extension) relative to sam2/sam2/configs/
                    e.g., "sam2.1_training/MAZAK_LoRA4_tiny"
        experiment_name: Name for the experiment (used for log/checkpoint dirs)
    """
    import logging

    from hydra import compose, initialize_config_module
    from iopath.common.file_io import g_pathmgr
    from omegaconf import OmegaConf
    from training.train import single_node_runner
    from training.utils.train_utils import makedir, register_omegaconf_resolvers

    initialize_config_module("sam2.sam2", version_base="1.2")
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

    cfg.dataset.img_folder = "/data/SAM2images/irPOLYMER/JPEGImages/train"
    cfg.dataset.gt_folder = "/data/SAM2images/irPOLYMER/Annotations/train"
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

    with zipfile.ZipFile(Path(f"/trainingresults/{experiment_name}.zip"), "w") as zip:
        for file in Path(experiment_dir).rglob("*"):
            if file.is_file():
                zip.write(file, file.relative_to(experiment_dir))
                logging.info(f"Added {file} to {zip.filename}")

    shutil.rmtree(f"/trainingresults/{experiment_name}")
    logging.info(f"Successfully removed {experiment_dir}")
    cloudbucket = CloudBucket(bucket_name=os.getenv("CF_R2_BUCKET_NAME"))
    logging.info(f"Uploading {experiment_name}.zip to R2 bucket")
    try:
        await cloudbucket.upload_file(
            Path(f"/trainingresults/{experiment_name}.zip"), f"{experiment_name}.zip"
        )
    except Exception as e:
        logging.error(f"Error uploading {experiment_name}.zip to R2 bucket: {e}")
        raise e
    finally:
        Path(f"/trainingresults/{experiment_name}.zip").unlink()
        logging.info(f"Successfully removed {experiment_name}.zip from Modal Volume")

    logging.info(
        f"Successfully copied {experiment_dir} to {os.getenv('CF_R2_BUCKET_NAME')}"
    )


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
    train.remote(config, experiment_name)
