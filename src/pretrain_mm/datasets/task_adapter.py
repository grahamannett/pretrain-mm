from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from pretrain_mm.datasets.base import Sample


@dataclass
class TrainSample(Sample):
    image: torch.Tensor = None
    input_ids: str | torch.Tensor = None
    label: str | torch.Tensor = None
    attention_mask: torch.Tensor = None


class WebsiteTasks:
    """trying to think of pretraining task for a website given a screenshot and the Website"""

    @staticmethod
    def base_task(sample: Sample) -> dict:
        """base clm task"""
        base_instruction = f"Title the following webpage:\n{sample.desc}"
        input_ids = f"{base_instruction}\nTitle: {sample.title}"

        return TrainSample(input_ids=input_ids, image=sample.image)


class TaskAdapter(Dataset):
    def __init__(self, dataset: Dataset, task_func: callable) -> None:
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
