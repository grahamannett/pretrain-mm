from collections import UserDict

import torch

from pretrain_mm.datasets.base import create_sample_type


@create_sample_type(subclasses=[UserDict])
class SampleBase:
    pass


@create_sample_type
class PreProcessedSample:
    text: str = None
    images: torch.Tensor = None


class Task:
    def __call__(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class TitleWebsiteTask(Task):
    def __call__(self, sample: SampleBase) -> dict:
        """base clm task"""
        base_instruction = f"Title the following webpage:\n{sample.desc}"
        text = f"{base_instruction}\nTitle: {sample.title}"

        return PreProcessedSample(text=text, images=sample.image)


class WebsiteTasks:
    """trying to think of pretraining task for a website given a screenshot and the Website"""

    TitleWebsiteTask = TitleWebsiteTask
