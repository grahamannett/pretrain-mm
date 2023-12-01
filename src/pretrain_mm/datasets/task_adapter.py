from typing import Callable


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


class TaskAdapter(Dataset):
    def __init__(self, dataset: Dataset, transforms: dict[str, callable] | list[callable] = None, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset

        if isinstance(transforms, list):
            transforms = {idx: fn for idx, fn in enumerate(transforms)}

        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        for t_name, t_func in self.transforms.items():
            sample = self._handle_func(sample, t_func, t_name)
        return sample

    def __repr__(self) -> str:
        dataset_info = f"TaskAdapter(\n\tdataset:={self.dataset.__name__},"
        for t_name, t_func in self.transforms.items():
            dataset_info += f"\n\t{t_name}:={t_func.__name__},"
        return dataset_info + "\n)"

    def _handle_func(self, sample: dict, func: Callable, func_name: str) -> dict:
        """handle a function on a sample"""
        try:
            return func(**sample)
        except Exception as err:
            raise SystemExit(f"Issue for {func_name} on sample: {sample} with Error: {err}")

    def _sort_transforms(
        self,
        key_order: list[str | int] = ["task_func", "preprocessor", "processor", "postprocessor"],
    ) -> None:
        """sort transforms by key_order,

        in case provided as dict and maybe like
            processor:..., preprocessor:..., postprocessor:..., task_func:... etc
        rather than:
            preprocessor:..., processor:..., postprocessor:...

        """
        _new_transforms = {}
        for key in key_order:
            _new_transforms[key] = self.transforms[key]
        self.transforms = _new_transforms
