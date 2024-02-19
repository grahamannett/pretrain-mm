from dataclasses import asdict, dataclass, field

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


@dataclass
class LocalDataConfig(BaseConfig):
    """this is the config for storing local data as sometimes wandb can be a pos,
    probably using tinydb

    Args:
        enabled (bool, optional): [description]. Defaults to False.
        path (str, optional): [description]. Defaults to "./output/local_data.json".
    """

    enabled: bool = False
    path: str = "./output/local_data.json"
