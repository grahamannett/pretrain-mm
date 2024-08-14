from collections import UserDict
from dataclasses import asdict, dataclass, field
from typing import dataclass_transform

import yaml


def _load_dataset_config(filepath: str, load_func: str | callable = yaml.full_load):
    if isinstance(load_func, str):
        load_func = getattr(yaml, load_func)

    with open(filepath, "r") as f:
        return load_func(f)


@dataclass
class ExtraDatasetArgs:
    datasets: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, filepath: str, load_func: str | callable = yaml.full_load):
        data = _load_dataset_config(filepath, load_func=load_func)
        return cls(**data)


class SampleDict(UserDict):
    # TODO: benchmark if using SampleDict is slower than create_sample_type_like_dict and create_sample_type_dict
    # this on is the one you can use like a dict without needing to stub the methods out
    def __post_init__(self):
        super().__init__(self.__dict__)


# but not clear how to do that, seems fine to just do
@dataclass_transform(order_default=True)
def create_sample_type(cls=None, *, subclasses: list[type] = None):
    def wrap(cls):
        # dynamically add subclasses
        if subclasses:
            for subclass in subclasses:
                cls = type(cls.__name__, (cls, subclass), dict(cls.__dict__))

        return dataclass(cls)

    if cls is None:
        # decorator is used with arguments, return wrapper
        return wrap

    # apply directly
    return wrap(cls)


@dataclass_transform(order_default=True)
def create_sample_type_like_dict(cls):
    # want to be able to just use like
    # class TaskSample(SampleBase):
    #     text: str
    # and then sample = TaskSample(input_str) where sample is both a dict and a dataclass
    # since that makes it more consistant/useable across various interfaces.
    # ideally would just subclass but that doesn't work with dataclasses easily
    cls.__getitem__ = lambda self, item: self.__dict__[item]
    cls.__setitem__ = lambda self, key, value: setattr(self, key, value)
    cls.get = lambda self, item, default=None: self.__dict__.get(item, default)
    cls.pop = lambda self, item, default=None: self.__dict__.pop(item, default)

    cls.asdict = lambda self: asdict(self)

    cls.items = lambda self: self.asdict()
    cls.keys = lambda self: self.asdict().keys()
    cls.values = lambda self: self.asdict().values()
    cls.__iter__ = lambda self: iter(self.asdict())

    return dataclass(cls)


@dataclass_transform(order_default=True)
def create_sample_type_dict(cls):
    cls = type(cls.__name__, (cls, SampleDict), dict(cls.__dict__))
    return dataclass(cls)
