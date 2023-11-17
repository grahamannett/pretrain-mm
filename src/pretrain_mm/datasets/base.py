from dataclasses import dataclass, asdict, is_dataclass

import torch

IGNORE_INDEX = -100


@dataclass
class Sample:
    def __getitem__(self, item: str):
        return self.__dict__[item]

    def asdict(self):
        if is_dataclass(self):
            return asdict(self)
        raise NotImplementedError(f"asdict not implemented for {self.__class__.__name__}")


@dataclass
class PreProcessedSample(Sample):
    text: str = None
    images: torch.Tensor = None

    def __iter__(self):
        breakpoint()
        return iter(self.__dict__.items())

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()


@dataclass
class TrainSample(Sample):
    image_patches: torch.Tensor = None
    image_patches_indices: torch.Tensor = None

    input_ids: str | torch.Tensor = None
    label: str | torch.Tensor = None
    attention_mask: torch.Tensor = None
