from torch.utils.data import Dataset

from pretrain_mm.datasets.base import PreProcessedSample, SampleBase


class Task:
    def __call__(self, sample: SampleBase) -> dict:
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
        return self.call_transforms(self.dataset[idx])

    def __repr__(self) -> str:
        name = getattr(self.dataset, "__name__", self.dataset.__class__.__name__)
        dataset_info = f"TaskAdapter(\n\tdataset:={name},"
        for t_name, t_func in self.transforms.items():
            dataset_info += f"\n\t{t_name}:={t_func.__name__},"
        return dataset_info + "\n)"

    def call_transforms(self, sample: dict, func_kwargs: list[dict] = None) -> dict:
        """call all transforms on sample"""
        for fn_idx, (fn_name, fn) in enumerate(self.transforms.items()):
            fn_kwargs = func_kwargs[fn_idx] if func_kwargs else {}
            sample = fn(sample, **fn_kwargs)

            if sample in [None, False]:
                # how to handle when likely task transform fails?
                break
            # try:
            #     sample = fn(sample, **fn_kwargs)
            # except Exception as err:
            #     raise SystemExit(f"Issue for {fn_name} on sample: {sample}|{fn_kwargs} with Error: {err}")
        return sample

    def _handle_func(self, sample: dict, func: callable, func_name: str = "unknown", func_kwargs: dict = {}) -> dict:
        """handle a function on a sample"""
        try:
            return func(sample, **func_kwargs)
        except Exception as err:
            raise SystemExit(f"Issue for {func_name} on sample: {sample}|{func_kwargs} with Error: {err}")


class FeedbackDatasetAdapter(Dataset):
    def __init__(
        self,
    ) -> None:
        super().__init__()
