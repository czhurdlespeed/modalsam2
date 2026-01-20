import asyncio
import os
import shutil
import zipfile
from pathlib import Path

import modal
from dotenv import load_dotenv
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from modal import Dict

from .config import ModelYamlConfig, UserSelections
from .containerimages import FASTAPI_LIGHTWEIGHT_IMAGE, SAM2_BASE_IMAGE

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

logfire_secret = modal.Secret.from_name(name="logfire", required_keys=["LOGFIRE_TOKEN"])


data_volume = modal.Volume.from_name("sam2_input_data")
checkpoint_volume = modal.Volume.from_name(
    "sam2_checkpoints", create_if_missing=True, version=1
)

job_queue = Dict.from_name("job-queue", create_if_missing=True)


@app.function(
    image=SAM2_BASE_IMAGE,
    gpu="L40s",
    volumes={"/data": data_volume, "/trainingresults": checkpoint_volume},
    secrets=[r2_secret],
    timeout=7200,
)
def launch_training(userselections: UserSelections):
    """Synchronous function to run training in background."""
    import logging

    from hydra import initialize_config_module
    from iopath.common.file_io import g_pathmgr
    from omegaconf import OmegaConf
    from training.train import single_node_runner
    from training.utils.train_utils import makedir, register_omegaconf_resolvers

    from .cloud import CloudBucket
    from .config import create_cfg

    cfg = create_cfg(
        ModelYamlConfig(
            userselections=userselections,
        )
    )

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

    yield from single_node_runner(cfg, main_port)
    checkpoint_volume.commit()

    zip_path = Path(
        f"{Path(cfg.launcher.experiment_log_dir).parent / Path(cfg.launcher.experiment_log_dir).name}/checkpoint.zip"
    )

    with zipfile.ZipFile(zip_path, "w") as zip:
        for file in Path(cfg.launcher.experiment_log_dir).rglob("*"):
            if file.is_file() and file.name != "checkpoint.zip":
                zip.write(file, file.relative_to(cfg.launcher.experiment_log_dir))
                logging.info(f"Added {file} to {zip.filename}")

    # Upload to cloud bucket - create new event loop for async operation in executor
    cloudbucket = CloudBucket(bucket_name=os.getenv("CF_R2_BUCKET_NAME"))
    logging.info(f"Uploading {zip_path} to R2 bucket")
    try:
        # Create new event loop for async operations in executor thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(
                cloudbucket.upload_file(
                    str(zip_path), str(zip_path.relative_to("/trainingresults"))
                )
            )
        finally:
            new_loop.close()
    except Exception as e:
        logging.error(f"Error uploading {zip_path} to R2 bucket: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading {zip_path} to R2 bucket: {str(e)}",
        )

    logging.info(f"Successfully copied {zip_path} to {os.getenv('CF_R2_BUCKET_NAME')}")
    # Remove all training results from Modal Volume. Keeping all user's jobs in the CloudFlare R2 bucket
    shutil.rmtree(Path(cfg.launcher.experiment_log_dir).parent)
    logging.info(
        f"Successfully cleaned user {Path(cfg.launcher.experiment_log_dir).parent.name} job {Path(cfg.launcher.experiment_log_dir).name} from Modal Volume"
    )

    try:
        user_plus_job_id = (
            f"{userselections.userjob.user_id}_{userselections.userjob.job_id}"
        )
        if user_plus_job_id in job_queue:
            job_queue.pop(user_plus_job_id)
            logging.info(
                f"Job {user_plus_job_id} removed from queue after training completion"
            )
    except Exception as e:
        logging.error(
            f"Error removing job {user_plus_job_id} from queue after training completion: {str(e)}"
        )


@app.function(
    image=FASTAPI_LIGHTWEIGHT_IMAGE,
    secrets=[logfire_secret],
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
async def train(
    userselections: UserSelections,
):
    """
    Args:
        userselections: User selections for the training
    """
    import logfire

    logfire.configure(service_name=app.name)
    with logfire.span(
        f"Creating training job for user {userselections.userjob.user_id} job {userselections.userjob.job_id}"
    ):
        user_plus_job_id = (
            f"{userselections.userjob.user_id}_{userselections.userjob.job_id}"
        )
        with logfire.span("Adding job to queue"):
            try:
                job_queue[user_plus_job_id] = modal.current_function_call_id()
                logfire.info(f"Job {user_plus_job_id} added to queue")
            except Exception as e:
                logfire.error(f"Error adding job {user_plus_job_id} to queue: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error adding job {user_plus_job_id} to queue: {str(e)}",
                )

        with logfire.span("Launching training...🚀"):
            return StreamingResponse(
                launch_training.remote_gen(userselections=userselections),
                media_type="text/event-stream",
            )


@app.function(
    image=FASTAPI_LIGHTWEIGHT_IMAGE, volumes={"/trainingresults": checkpoint_volume}
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
async def cancel_job(user_plus_job_id: str):
    if user_plus_job_id not in job_queue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User's job not found",
        )

    try:
        modal.FunctionCall.from_id(job_queue[user_plus_job_id]).cancel(
            terminate_containers=True
        )
        job_queue.pop(user_plus_job_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling job: {str(e)}",
        )
    user_dir, job_id = user_plus_job_id.split("_")
    checkpoint_volume.reload()
    if Path(f"/trainingresults/{user_dir}/{job_id}").exists():
        shutil.rmtree(Path(f"/trainingresults/{user_dir}/{job_id}"))
        print(f"Successfully cleaned user {user_dir} job {job_id} from Modal Volume")
    else:
        print(f"User {user_dir} job {job_id} not found in Modal Volume")
    return {"message": f"User {user_dir} job {job_id} cancelled successfully ✅\n"}
