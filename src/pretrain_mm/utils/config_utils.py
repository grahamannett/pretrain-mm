import functools
from dataclasses import asdict, dataclass, field

from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode, Serializable

import wandb
from pretrain_mm import logger

"""
Note: this is called ModelInfo and not ModelConfig so that it doesn't conflict with
the ModelConfig class from transformers.  It might make sense to rename this to
ModelInfo or something
"""


@dataclass
class BaseConfig(Serializable):
    pass


class FromConfig:
    """helper class so subclassing in experiment runs can use like"""

    def __class_getitem__(cls, *args, **kwargs):
        return field(default_factory=functools.partial(*args))

    @staticmethod
    def setup_parser(
        argument_generation_mode: ArgumentGenerationMode = ArgumentGenerationMode.FLAT,
        nested_mode: NestedMode = NestedMode.WITHOUT_ROOT,
        add_dest_to_option_strings: bool = True,
    ):
        return ArgumentParser(
            argument_generation_mode=argument_generation_mode,
            nested_mode=nested_mode,
            add_dest_to_option_strings=add_dest_to_option_strings,
        )

    Base = BaseConfig


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
    job_type: str = "testing"
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


from typing import dataclass_transform


@dataclass_transform(order_default=True)
def config_from(cls=None, bases=None):

    if bases:
        cls = type(cls.__name__, (bases, cls), {})

    cls.__getitem__ = lambda self, item: self.__dict__[item]
    cls.__setitem__ = lambda self, key, value: setattr(self, key, value)
    cls.get = lambda self, item, default=None: self.__dict__.get(item, default)

    cls.asdict = lambda self: asdict(self)

    cls.items = lambda self: self.asdict()
    cls.keys = lambda self: self.asdict().keys()
    cls.values = lambda self: self.asdict().values()
    cls.__iter__ = lambda self: iter(self.asdict())

    cls.__repr__ = lambda self: f"{cls.__name__}({self.asdict()})"

    return dataclass(cls)


"""_summary_
# use like

@config_as(BaseWandBConfig)
class Config:
    cmd: str
"""
