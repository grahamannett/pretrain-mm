from typing import Any, Callable


from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from pretrain_mm import logger
from pretrain_mm.datasets.base import Sample, PreProcessedSample


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
        # processors are for converting the sample to the format needed for tokenizer/model input generally
        preprocessor: Callable = None,
        postprocessor: Callable = None,
    ) -> None:
        super().__init__(dataset, task_func)
        self.processor = processor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def to_task(self, idx: int):
        sample = super().to_task(idx)

        if self.preprocessor:
            sample = self.preprocessor(sample)

        try:
            sample = self.processor(**sample)
        except Exception as err:
            raise SystemExit(f"Could not use processor on sample: {sample} with Error: {err}")

        if self.postprocessor:
            sample = self.postprocessor(sample)

        return sample
