from typing import Callable

import time
from torch.utils.data import Dataset

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

        if not isinstance(transforms, (list, dict)):
            transforms = [transforms]

        if isinstance(transforms, list):
            transforms = {f"{idx}_{fn.__name__}": fn for idx, fn in enumerate(transforms)}

        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        sample = self.call_transforms(sample)
        return sample

    def __repr__(self) -> str:
        name = getattr(self.dataset, "__name__", self.dataset.__class__.__name__)
        dataset_info = f"TaskAdapter(\n\tdataset:={name},"
        for t_name, t_func in self.transforms.items():
            dataset_info += f"\n\t{t_name}:={t_func.__name__},"
        return dataset_info + "\n)"

    def call_transforms(self, sample: dict, func_kwargs: list[dict] = None) -> dict:
        """call all transforms on sample"""
        tc = [(0, time.perf_counter())]
        for fn_idx, (fn_name, fn) in enumerate(self.transforms.items()):
            fn_kwargs = func_kwargs[fn_idx] if func_kwargs else {}
            try:
                sample = fn(sample, **fn_kwargs)
                tc.append((fn_name, time.perf_counter()))
            except Exception as err:
                raise SystemExit(f"Issue for {fn_name} on sample: {sample}|{fn_kwargs} with Error: {err}")

            # sample = self._handle_func(
            #     sample,
            #     func=t_func,
            #     func_name=t_name,
            #     func_kwargs=fn_kwargs,
            # )
        tc.append((time.perf_counter(), "end"))

        tc.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Done call_transforms on sample. Time Taken: {tc[0][0] - tc[-1][0]} and tc: {tc}")
        return sample

    def _handle_func(self, sample: dict, func: Callable, func_name: str = "unknown", func_kwargs: dict = {}) -> dict:
        """handle a function on a sample"""
        try:
            return func(sample, **func_kwargs)
        except Exception as err:
            raise SystemExit(f"Issue for {func_name} on sample: {sample}|{func_kwargs} with Error: {err}")

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

    @staticmethod
    def unpack_for_transform(func):
        def _fn(sample):
            return func(**sample)

        return _fn


class FeedbackDatasetAdapter(Dataset):
    def __init__(
        self,
    ) -> None:
        super().__init__()
