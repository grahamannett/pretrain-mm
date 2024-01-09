from dataclasses import dataclass, field, asdict

import torch
import wandb
from simple_parsing import Serializable

from pretrain_mm import logger


"""
Note: this is called ModelInfo and not ModelConfig so that it doesn't conflict with
the ModelConfig class from transformers.  It might make sense to rename this to
ModelInfo or something
"""


class DumpMixin:
    """
    mixin for dataclasses that cant dump to yaml for simple_parsing without refactor into partials
    """

    def dumps_yaml(self) -> str:
        out_str = ""
        for key, val in asdict(self).items():
            out_str += f"{key}: {val}\n"
        return out_str


@dataclass
class ModelInitInfo(DumpMixin):
    model_name: str
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)

    model_extra_info: dict = field(default=None)

    ModelCls: callable = None
    ProcessorCls: callable = None


@dataclass
class BaseConfig(Serializable):
    pass


@dataclass
class BaseTrainConfig(BaseConfig):
    model_config: ModelInitInfo = None
    device: str = "auto"

    output_dir: str = None
    num_iters: int = None
    epochs: int = 1
    grad_accum_steps: int = 1
    gradient_clipping: float = None
    save_every: str = None


@dataclass
class BaseWandBConfig(BaseConfig):
    group: str = None
    project: str = "pretrain-mm"
    job_type: str = "finetune"
    mode: str = "disabled"


def setup_wandb(wandb_config: BaseWandBConfig, config: BaseConfig = None) -> None:
    wandb.init(
        config=config,
        project=wandb_config.project,
        group=wandb_config.group,
        job_type=wandb_config.job_type,
        mode=wandb_config.mode,
    )


def check_train_config(train_config: BaseConfig) -> None:
    logger.info(f"Running Train. Config:\n{train_config.dumps_yaml()}")
    logger.info(f"Model Config:\n{train_config.model_config.dumps_yaml()}")

    if train_config.output_dir is None:
        output_dir_warn = "`train_config.output_dir` is None"
        output_dir_warn += "\nthis will not save model and if you are doing real train you should exit now"
        logger.warn(output_dir_warn)
