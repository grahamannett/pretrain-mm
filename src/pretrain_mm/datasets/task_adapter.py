from collections import UserDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from pretrain_mm.datasets.base import Sample


@dataclass
class RawSample(Sample):
    images: torch.Tensor = None
    text: str = None

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


@dataclass
class Task:
    def __call__(self, sample: Sample) -> dict:
        raise NotImplementedError


class TitleWebsiteTask(Task):
    def __call__(self, sample: Sample) -> dict:
        """base clm task"""
        base_instruction = f"Title the following webpage:\n{sample.desc}"
        text = f"{base_instruction}\nTitle: {sample.title}"

        return RawSample(text=text, images=sample.image)


class WebsiteTasks:
    """trying to think of pretraining task for a website given a screenshot and the Website"""

    TitleWebsiteTask = TitleWebsiteTask


class TaskAdapter(Dataset):
    def __init__(self, dataset: Dataset, task_func: Callable) -> None:
        self.dataset = dataset
        self.task_func = task_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            raise NotImplementedError("TODO: implement slicing")

        return self.convert(idx)

    def __repr__(self) -> str:
        return f"TaskAdapter(\n\tdataset:={self.dataset.__name__},\n\ttask:={self.task_func.__name__}\n)"

    def convert(self, idx: int):
        sample = self.task_func(self.dataset[idx])
        return sample


class TaskAdapterProcessor(TaskAdapter):
    def __init__(
        self,
        dataset: Dataset,
        task_func: Callable = None,
        processor: ProcessorMixin | PreTrainedTokenizer = None,
    ) -> None:
        super().__init__(dataset, task_func)
        self.processor = processor

    def convert(self, idx: int):
        sample = self.dataset[idx]
        if self.task_func:
            sample = self.task_func(sample)

        data = self.processor(**sample)
        sample = TrainSample(**data)
        return sample
