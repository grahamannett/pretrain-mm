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


class WeightedStagedDataset(torch.utils.data.IterableDataset):
    """
    A dataset that samples data from multiple stages with different weights.

    Args:
        datasets (list[torch.utils.data.Dataset]): List of datasets to sample from.
        stage_iters (list[int]): List of iterations for each stage.
        stage_weights (list[list[float]]): List of weights for each stage.
        stage_kwargs (dict[int, dict], optional): Additional keyword arguments for each stage. Defaults to {}.
        num_samples (int, optional): Number of samples to generate. Defaults to None.
    """

    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        stage_iters: list[int],
        stage_weights: list[list[float]],
        stage_kwargs: dict[int, dict] = {},
        num_samples: int = None,
    ):
        self.num_samples = num_samples
        self.datasets = datasets
        self.stage_iters = stage_iters
        self.stage_weights = stage_weights
        self.stage_kwargs = stage_kwargs

        while len(stage_iters) < len(stage_weights):
            stage_iters += [1]

        # for info
        stage_kw = []
        dataset_kw = []

        stage_samplers = []
        dataset_samplers = []

        stage_samplers_iter = []
        dataset_samplers_iter = []

        # for i, stage in

        for i, ds in enumerate(datasets):
            stage_kw_for_ds = _stage_defaults | {"num_samples": stage_iters[i]} | stage_kwargs.get(i, {})
            stage_kw.append(stage_kw_for_ds)

            dataset_kw_for_ds = _dataset_defaults | stage_kwargs.get(i, {})
            dataset_kw.append(dataset_kw_for_ds)

            stage_sampler = WeightedRandomSampler(stage_weights[i], **stage_kw_for_ds)
            stage_samplers.append(stage_sampler)
            stage_samplers_iter.append(iter(stage_sampler))

            dataset_sampler = RandomSampler(ds, **dataset_kw_for_ds)
            dataset_samplers.append(dataset_sampler)
            dataset_samplers_iter.append(iter(dataset_sampler))

        # REMOVE
        # self._stage_kw = [
        #     stage_kwargs.get(i, {}) | _stage_defaults | {"num_samples": stage_iters[i]} for i in range(len(datasets))
        # ]
        # self._dataset_sampler_kw = [stage_kwargs.get(i, {}) | _dataset_defaults for i in range(len(datasets))]

        # self.stage_samplers = [WeightedRandomSampler(x, **self._stage_kwargs[i]) for i, x in enumerate(stage_weights)]
        # self.dataset_samplers = [RandomSampler(x, **self._dataset_sampler_kwargs[i]) for i, x in enumerate(datasets)]

        # self.stage_samplers = list(map(iter, self.stage_samplers))
        # self.dataset_samplers = list(map(iter, self.dataset_samplers))

    def reset(self):
        """
        Reset the current stage to the initial stage.
        """
        self.curr_stage = 0

    def _get_dataset_idx(self):
        """
        Get the index of the dataset to sample from based on the current stage.

        Yields:
            int: Index of the dataset.
        """
        try:
            stage_sampler = self.stage_samplers[self.curr_stage]
            return next(stage_sampler)

        except StopIteration:
            self.curr_stage += 1
            return self._get_dataset_idx()

    def __iter__(self):
        """
        Iterate over the dataset and yield samples.

        Yields:
            Any: Sample from the dataset.
        """
        n = 0
        self.curr_stage = 0

        def _should_sample(inc: bool = True):
            nonlocal n
            if self.num_samples is None:
                if inc:
                    n += 1
                return True
            else:
                return n < self.num_samples

        self._should_sample = _should_sample
        return self

    def __next__(self):
        while self._should_sample():
            dataset_idx = self._get_dataset_idx()
            sampler = self.dataset_samplers[dataset_idx]
            breakpoint()
            sample_idx = next(sampler)
            sample = self.datasets[dataset_idx][sample_idx]
            return sample
