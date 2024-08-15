import functools
from dataclasses import asdict, dataclass, field
from typing import Callable, dataclass_transform

import torch
import tyro
import yaml
from datasets import load_dataset

from pretrain_mm import constants


"""
Note: this is called ModelInfo and not ModelConfig so that it doesn't conflict with
the ModelConfig class from transformers.  It might make sense to rename this to
ModelInfo or something
"""


@functools.cache
def config_load(filepath: str, load_func: Callable | str = yaml.full_load):
    """
    Load configuration data from a file.

    Parameters:
    - filepath (str): The path to the configuration file.
    - load_func (Callable | str): The function or method used to load the configuration file. If a string is provided,
        it should be the name of a function from the `yaml` module.

    Returns:
    - data: The loaded configuration data.

    Note:
    - The `load_func` parameter can be either a function or a string. If it is a string, it should be the name of a
        function from the `yaml` module.
    """
    if isinstance(load_func, str):
        load_func = getattr(yaml, load_func)

    with open(filepath, "r") as f:
        data = load_func(f)
    return data


"""

below are the classes that are used to define the config for the experiments

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

    def __class_getitem__(cls, key, *args, **kwargs):
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


class CLIMixin:
    @classmethod
    def cli(cls, **kwargs):
        return tyro.cli(cls, **kwargs)


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
class BaseTrainConfig(BaseConfig, CLIMixin):
    device: str = "auto"
    model_dtype: str = None  # "float16"
    epochs: int = 1
    grad_accum_steps: int = 1
    gradient_clipping: float = None
    output_dir: str = None
    num_iters: int = None
    save_every: str = None

    # for making the model have only 1 decoder block, i.e. local dev
    # model_path: ExperimentConfigModelInfo = None
    model_chop: bool | int | None = False
    model_modify_config: bool = False

    ignore_index: int = constants.IGNORE_INDEX

    @property
    def model_init_kwargs(self):
        return {
            "device_map": self.device,
            "torch_dtype": getattr(torch, self.model_dtype) if self.model_dtype else None,
        }


@dataclass
class DatasetConfigFile:
    """
    Represents a configuration file for a dataset.

    Attributes:
        enabled (bool): Indicates if the dataset is enabled.
        meta (dict): Metadata for the dataset.
        init (dict): Initialization parameters for the dataset.
        stages (dict): Stages for the dataset.
        transforms (tuple[str, ...]): Tuple of transform function names.
        config_filepath (str): Filepath of the dataset configuration file.

    Methods:
        use(name: str = None, as_field: bool = True, config_filepath: str = None) -> Union[DatasetConfigFile, Field]:
            Returns a new instance of DatasetConfigFile with the specified configuration.

        load(available_transforms: dict = {}, **kwargs) -> Any:
            Loads the dataset using the specified transforms and additional keyword arguments.

    """

    enabled: bool = False
    meta: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    transforms: tuple[str, ...] = ()

    config_filepath: str = constants.datasets_config_filepath

    @classmethod
    def use(cls, name: str = None, as_field: bool = True, config_filepath: str = None):
        """
        Load and use a configuration for the specified class.

        Args:
            cls: The class to instantiate with the loaded configuration.
            name (str, optional): The name of the configuration to use. If provided, only the specified configuration
                will be loaded. Defaults to None.
            as_field (bool, optional): Whether to return the class instance as a field. If True, the instance will be
                returned as a default factory field. Defaults to True.
            config_filepath (str, optional): The filepath of the configuration file. If not provided, the default
                filepath of the class will be used. Defaults to None.

        Returns:
            cls: The instantiated class instance with the loaded configuration.
        """
        config_filepath = config_filepath or cls.config_filepath
        config = config_load(filepath=config_filepath)

        if name:
            config = config[name]

        if as_field:
            return field(default_factory=lambda: cls(**config))
        return cls(**config)

    def load(self, available_transforms: dict = {}, **kwargs):
        dataset = load_dataset(**self.init, **kwargs)
        for func_str in self.transforms:
            func = available_transforms[func_str]
            dataset.set_transform(func)

        self.dataset = dataset
        return dataset


@dataclass
class BaseDatasets(BaseConfig):
    """
    Base configuration for datasets.

    Methods:
        get(enabled_only: bool): Get the list of dataset configurations.
        load_transforms(transforms: dict): Load available transformations.
    """

    def get(self, enabled_only: bool = True):
        """
        Get the list of dataset configurations.

        Args:
            enabled_only (bool, optional): Whether to return only enabled datasets.

        Returns:
            list: List of dataset configurations.
        """
        ds = [v for v in self.__dict__.values() if isinstance(v, DatasetConfigFile)]
        if enabled_only:
            ds = [v for v in ds if v.enabled]

        return ds


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
