from typing import Annotated, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field


class UserJob(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    job_id: int = Field(..., ge=1)

    def model_post_init(self, __context) -> None:
        """Validate user_id to prevent path traversal and injection attacks."""
        # Prevent path traversal and injection attacks
        if ".." in self.user_id or "/" in self.user_id or "\\" in self.user_id:
            raise ValueError("user_id contains invalid characters")
        # Only allow alphanumeric, hyphens, and underscores
        if not all(c.isalnum() or c in "-_" for c in self.user_id):
            raise ValueError(
                "user_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        if not self.user_id or len(self.user_id.strip()) == 0:
            raise ValueError("user_id cannot be empty")


class UserSelections(BaseModel):
    userjob: UserJob
    fullfinetune: bool
    lora_rank: Literal[2, 4, 8, 16, 32] | None
    base_model: Literal["tiny", "small", "base_plus", "large"]
    dataset: Literal["irPOLYMER", "visPOLYMER", "TIG", "MAZAK"]
    num_epochs: Annotated[int, Field(ge=1, le=100)]

    @computed_field
    @property
    def gpu_type(self) -> str:
        if self.fullfinetune:
            return "A100-80GB"
        # LoRA cases
        match self.base_model:
            case "tiny" | "small":
                return "L40S"
            case "base_plus" | "large":
                return "A100-80GB"


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
