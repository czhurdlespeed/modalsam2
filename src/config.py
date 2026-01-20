from typing import Annotated, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field


class UserJob(BaseModel):
    user_id: str
    job_id: int


class UserSelections(BaseModel):
    userjob: UserJob
    fullfinetune: bool
    lora_rank: Literal[2, 4, 8, 16, 32] | None
    base_model: Literal["tiny", "small", "base_plus", "large"]
    dataset: Literal["irPOLYMER", "visPOLYMER", "TIG", "MAZAK"]
    num_epochs: Annotated[int, Field(ge=1, le=100)]


class ModelYamlConfig(BaseModel):
    userselections: UserSelections
    use_cluster: bool = False
    num_nodes: int = 1
    gpus_per_node: int = 1

    @computed_field
    @property
    def experiment_dir(self) -> str:
        """Auto-generated from userjob."""
        return f"/trainingresults/{self.userselections.userjob.user_id}/{self.userselections.userjob.job_id}"

    @computed_field
    @property
    def num_gpus(self) -> int:
        """Auto-computed from gpus_per_node * num_nodes."""
        return self.gpus_per_node * self.num_nodes

    @computed_field
    @property
    def img_folder(self) -> str:
        """Auto-generated from dataset."""
        return f"/data/SAM2images/{self.userselections.dataset}/JPEGImages/train"

    @computed_field
    @property
    def gt_folder(self) -> str:
        """Auto-generated from dataset."""
        return f"/data/SAM2images/{self.userselections.dataset}/Annotations/train"

    @computed_field
    @property
    def checkpoint_path(self) -> str:
        """Auto-generated from base_model."""
        return f"/sam2modalwebapp/sam2/checkpoints/sam2.1_hiera_{self.userselections.base_model}.pt"

    @computed_field
    @property
    def baseconfig_path(self) -> str:
        if self.userselections.fullfinetune:
            return f"/sam2modalwebapp/sam2/sam2/configs/sam2.1_training/{{DATASET}}_FT_{self.userselections.base_model}.yaml"
        elif self.userselections.lora_rank is not None:
            return f"/sam2modalwebapp/sam2/sam2/configs/sam2.1_training/{{DATASET}}_LoRA_{self.userselections.base_model}.yaml"
        else:
            raise ValueError("Either fullfinetune or lora_rank must be set")

    @computed_field
    @property
    def gpu_type(self) -> str:
        match self.userselections.base_model:
            case "tiny" | "small":
                if self.num_gpus > 1:
                    return f"L40S:{self.num_gpus}"
                return "L40S"
            case "base_plus" | "large":
                if self.num_gpus > 1:
                    return f"A100-80GB:{self.num_gpus}"
                return "A100-80GB"


def create_cfg(cfg: ModelYamlConfig) -> OmegaConf:
    with open(cfg.baseconfig_path, "r") as f:
        base_cfg = OmegaConf.load(f)

    base_cfg.launcher.experiment_log_dir = cfg.experiment_dir
    base_cfg.launcher.num_nodes = cfg.num_nodes
    base_cfg.launcher.gpus_per_node = cfg.gpus_per_node
    base_cfg.submitit.use_cluster = cfg.use_cluster
    base_cfg.dataset.img_folder = cfg.img_folder
    base_cfg.dataset.gt_folder = cfg.gt_folder
    base_cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = (
        cfg.checkpoint_path
    )
    base_cfg.scratch.num_epochs = cfg.userselections.num_epochs
    if not cfg.userselections.fullfinetune:
        base_cfg.trainer.LoRA.r = cfg.userselections.lora_rank
        base_cfg.trainer.LoRA.adapter_name = (
            f"SAM2_LoRA{cfg.userselections.lora_rank}_{cfg.userselections.base_model}"
        )
    base_cfg["userjob"] = "_".join(
        [
            str(cfg.userselections.userjob.user_id),
            str(cfg.userselections.userjob.job_id),
        ]
    )

    return base_cfg
