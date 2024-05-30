import functools
from dataclasses import asdict, dataclass, field
from typing import dataclass_transform

import tyro


"""
Note: this is called ModelInfo and not ModelConfig so that it doesn't conflict with
the ModelConfig class from transformers.  It might make sense to rename this to
ModelInfo or something
"""


@dataclass
class BaseConfig:
    @staticmethod
    def use(inst, **kwargs):
        return field(default_factory=inst, **kwargs)

    def dumps_yaml(self) -> str:
        return tyro.extras.to_yaml(self)


class FromConfig:
    """helper class so subclassing in experiment runs can use like"""

    def __class_getitem__(cls, key):
        if not callable(key):
            return field(default_factory=lambda: key)

        # not clear if should be using functools.partial
        return field(default_factory=functools.partial(key))

    @classmethod
    def make(cls, **kwargs):
        return field(**kwargs)

    Base = BaseConfig


class DumpMixin:
    """
    mixin for dataclasses that cant dump to yaml for simple_parsing without refactor due to classes/fields
    """

    def dumps_yaml(self) -> str:
        out_str = ""
        for key, val in asdict(self).items():
            out_str += f"{key}: {val}\n"
        return out_str


@dataclass
class ModelInitInfo(DumpMixin):
    model_name: str

    model_extra_info: dict = field(default=None)
    model_kwargs: dict = field(default_factory=dict)

    ModelConfigCls: callable = field(default=None, repr=False)
    ModelCls: callable = field(default=None, repr=False)
    ProcessorCls: callable = field(default=None, repr=False)

    ModelConstants: callable = field(default=None, repr=False)
    ModelConstantsCls: callable = field(default=None, repr=False)

    tokenizer_kwargs: dict = field(default_factory=dict)

    get_model_config_kwargs: callable = field(default=None)

    modify_model_config_callback: callable = None


@dataclass
class BaseTrainConfig(BaseConfig):
    device: str = "auto"
    epochs: int = 1
    grad_accum_steps: int = 1
    gradient_clipping: float = None
    output_dir: str = None
    num_iters: int = None
    save_every: str = None

    # for making the model have only 1 decoder block, i.e. local dev
    model_chop: bool | int | None = False
    model_modify_config: bool = False


@dataclass
class WandBConfig(BaseConfig):
    group: str = None
    project: str = "pretrain-mm"
    job_type: str = "testing"
    mode: str = "disabled"

    tags: list[str] | tuple[str, ...] | None = None
    # unlikely that you want to use these but...
    name: str | None = None


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
