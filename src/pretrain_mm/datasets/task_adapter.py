from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from pretrain_mm.datasets.base import Sample, PreProcessedSample


def default_pre_processor(sample: Sample) -> dict:
    return {"text": sample.text, "images": [sample.image]}


class Task:
    def __call__(self, sample: Sample) -> dict:
        raise NotImplementedError


class TitleWebsiteTask(Task):
    def __call__(self, sample: Sample) -> dict:
        """base clm task"""
        base_instruction = f"Title the following webpage:\n{sample.desc}"
        text = f"{base_instruction}\nTitle: {sample.title}"

        return PreProcessedSample(text=text, images=sample.image)


class WebsiteTasks:
    """trying to think of pretraining task for a website given a screenshot and the Website"""

    TitleWebsiteTask = TitleWebsiteTask


class TaskAdapterBase(Dataset):
    """
    use this if you want to subclass TaskAdapterBase and implement your own to_task method
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        sample = self.to_task(sample)
        return sample

    def to_task(self, sample: Any):
        raise NotImplementedError


class TaskAdapter(TaskAdapterBase):
    """TaskAdapter takes a dataset and a callable task function and converts the sample to the task.
    The task

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, dataset: Dataset, task_func: Callable) -> None:
        self.dataset = dataset
        self.task_func = task_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            raise NotImplementedError("TODO: implement slicing")

        return self.to_task(idx)

    def __repr__(self) -> str:
        return f"TaskAdapter(\n\tdataset:={self.dataset.__name__},\n\ttask:={self.task_func.__name__}\n)"

    def to_task(self, idx: int):
        sample = self.task_func(self.dataset[idx])
        return sample


class TaskAdapterProcessor(TaskAdapter):
    def __init__(
        self,
        dataset: Dataset,
        task_func: Callable = None,
        processor: ProcessorMixin | PreTrainedTokenizer = None,
        # pre processor
        pre_processor: Callable = None,
        post_processor: Callable = None,
    ) -> None:
        super().__init__(dataset, task_func)
        self.processor = processor
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    def to_task(self, idx: int):
        task_sample = super().to_task(idx)
        text = task_sample["text"] + task_sample["label"]
        images = task_sample["image"]

        return {
            "text": text,
            "images": images,
        }

    # def __getitem__(self, idx):
    #     sample = super().__getitem__(idx)

    #     return sample

    # def convert(self, idx: int):
    #     sample = self.dataset[idx]
    #     if self.task_func:
    #         sample = self.task_func(sample)

    #     if self.pre_processor:
    #         sample = self.pre_processor(sample)

    #     sample = self.processor(**sample)

    #     if self.post_processor:
    #         sample = self.post_processor(sample)

    #     return sample
