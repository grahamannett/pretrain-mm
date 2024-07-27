import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler


_stage_defaults = {
    "replacement": True,
    "num_samples": 10,
}

_dataset_defaults = {
    "replacement": False,
    "num_samples": 1,
}


class WeightedStagedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        stage_thresholds: list[int],
        stage_weights: list[list[float]],
        stage_kwargs: dict[int, dict] = {},
        num_samples: int = None,
    ):
        self.num_samples = num_samples
        self.datasets = datasets
        self.stage_weights = stage_weights

        while len(stage_thresholds) < len(stage_weights):
            stage_thresholds += [1]

        self.stage_samplers = [
            WeightedRandomSampler(
                x,
                **(stage_kwargs.get(i, {}) | _stage_defaults | {"num_samples": stage_thresholds[i]}),
            )
            for i, x in enumerate(stage_weights)
        ]

        self.dataset_samplers = [
            RandomSampler(
                x,
                **(stage_kwargs.get(i, {}) | _dataset_defaults),
            )
            for i, x in enumerate(datasets)
        ]

        self.stage_samplers = map(iter, self.stage_samplers)
        self.dataset_samplers = map(iter, self.dataset_samplers)

    def reset(self):
        self.curr_stage = 0

    def _get_dataset_idx(self):
        try:
            yield next(self.stage_samplers[self.curr_stage])
        except StopIteration:
            self.curr_stage += 1

    def __iter__(self):
        n = 0
        self.curr_stage = 0

        def _should_sample():
            nonlocal n
            if self.num_samples is None:
                n += 1
                return True
            else:
                return n < self.num_samples

        while _should_sample():
            dataset_idx = self._get_dataset_idx()
            dataset = self.datasets[dataset_idx]

            sample_idx = next(self.dataset_samplers[dataset_idx])
            sample = dataset[sample_idx]
            yield sample
