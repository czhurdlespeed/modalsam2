import os
import shutil
import uuid
import zipfile
from pathlib import Path

import modal
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from omegaconf import OmegaConf

from .baseimage import BASE_IMAGE
from .cloud import CloudBucket
from .config import ModelYamlConfig, UserJob, UserSelections, create_cfg

load_dotenv()

auth_scheme = HTTPBearer()

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

token_secret = modal.Secret.from_name("token", required_keys=["AUTH_TOKEN"])

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
    secrets=[r2_secret, token_secret],
    timeout=7200,
)
@modal.fastapi_endpoint(method="POST")
async def train(
    userselections: UserSelections,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """
    Args:
        userselections: User selections for the training
        token: Authorization token
    """
    if token.credentials != os.getenv("AUTH_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    cfg = create_cfg(
        ModelYamlConfig(
            userselections=userselections,
            userjob=UserJob(user_id=uuid.uuid4().hex, job_id=1),
            use_cluster=False,
            dataset="irPOLYMER",
            base_model="tiny",
            num_epochs=1,
        )
    )

    import logging

    from hydra import initialize_config_module
    from iopath.common.file_io import g_pathmgr
    from training.train import single_node_runner
    from training.utils.train_utils import makedir, register_omegaconf_resolvers

    initialize_config_module("sam2.sam2", version_base="1.2")
    register_omegaconf_resolvers()

    logging.basicConfig(level=logging.INFO)
    makedir(cfg.launcher.experiment_log_dir)

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

    zip_path = Path(
        f"{Path(cfg.launcher.experiment_log_dir).parent / Path(cfg.launcher.experiment_log_dir).name}/checkpoint.zip"
    )

    with zipfile.ZipFile(zip_path, "w") as zip:
        for file in Path(cfg.launcher.experiment_log_dir).rglob("*"):
            if file.is_file():
                zip.write(file, file.relative_to(cfg.launcher.experiment_log_dir))
                logging.info(f"Added {file} to {zip.filename}")

    cloudbucket = CloudBucket(bucket_name=os.getenv("CF_R2_BUCKET_NAME"))
    logging.info(f"Uploading {zip_path} to R2 bucket")
    try:
        await cloudbucket.upload_file(
            str(zip_path), str(zip_path.relative_to("/trainingresults"))
        )
    except Exception as e:
        logging.error(f"Error uploading {zip_path} to R2 bucket: {e}")
        raise e

    logging.info(f"Successfully copied {zip_path} to {os.getenv('CF_R2_BUCKET_NAME')}")
    shutil.rmtree(Path(cfg.launcher.experiment_log_dir).parent)
    logging.info(
        f"Successfully cleaned user {Path(cfg.launcher.experiment_log_dir).parent.name} job {Path(cfg.launcher.experiment_log_dir).name} from Modal Volume"
    )


# @app.local_entrypoint()
# def main(
#     config: str = "configs/sam2.1_training/MAZAK_LoRA4_tiny",
#     experiment_name: str = "sam2_training_run1",
# ):
#     """
#     Local entrypoint to launch training.

#     Usage:
#         modal run main.py --config "configs/sam2.1_training/MAZAK_LoRA4_tiny" --experiment-name "my_run"
#     """
#     train.remote(config, experiment_name)
