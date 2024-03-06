from dataclasses import asdict, dataclass, is_dataclass
from typing import dataclass_transform

import torch


# want to be able to just use like
# class TaskSample(SampleBase):
#     text: str
# and then sample = TaskSample(input_str) where sample is both a dict and a dataclass
# since that makes it more consistant/useable across various interfaces.
# ideally would just subclass but that doesn't work with dataclasses or im not sure how to do it
# but not clear how to do that, seems fine to just do
@dataclass_transform(order_default=True)
def create_sample_type(cls, /, *args, **kwargs):
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


@create_sample_type
class SampleBase:
    pass


@create_sample_type
class PreProcessedSample:
    text: str = None
    images: torch.Tensor = None


@create_sample_type
class TrainSample:
    image_patches: torch.Tensor = None
    image_patches_indices: torch.Tensor = None

    input_ids: str | torch.Tensor = None
    label: str | torch.Tensor = None
    attention_mask: torch.Tensor = None
