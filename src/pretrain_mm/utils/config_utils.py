from dataclasses import dataclass, field, asdict

import torch
import wandb
from simple_parsing import Serializable

torch_dtype_float16 = {"torch_dtype": torch.float16}


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
    model_config: ModelInitInfo = None


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

    wandb.run.save()
