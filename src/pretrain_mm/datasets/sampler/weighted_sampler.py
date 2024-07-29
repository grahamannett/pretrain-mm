from torch.utils.data import Dataset, IterableDataset, RandomSampler, WeightedRandomSampler


class StagedDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float],
        num_iters: int = None,
        include_idxs: bool = False,
    ):
        """
        Initializes a WeightedSampler object.

        Args:
            datasets (list[Dataset]): A list of datasets to sample from.
            weights (list[float]): A list of weights corresponding to each dataset.
            num_iters (int, optional): The number of iterations to sample. If not provided,
                it will be calculated as the sum of the lengths of all datasets. Defaults to None.
            include_idxs (bool, optional): Whether to include the sampled indices in the output. Defaults to False.
        """
        self.datasets = datasets
        self.weights = weights
        self.num_iters = num_iters
        self.include_idxs = include_idxs

        num_iters = num_iters or sum(len(ds) for ds in datasets)
        self.sampler = WeightedRandomSampler(weights, replacement=True, num_samples=num_iters)
        self.dataset_samplers = [RandomSampler(ds, replacement=False) for ds in datasets]

    def __iter__(self):
        self.sampler_iter = iter(self.sampler)
        self.dataset_iter = [iter(ds) for ds in self.dataset_samplers]
        return self

    def __next__(self):
        """
        Retrieves the next sample from the weighted sampler.

        Returns:
            The next sample from the weighted sampler.
        """
        sample_from_ds = next(self.sampler_iter)

        if (dataset_idx := next(self.dataset_iter[sample_from_ds], None)) is None:
            dataset_idx = self._reset_dataset_iter(sample_from_ds)

        sample = self.datasets[sample_from_ds][dataset_idx]

        if self.include_idxs:
            sample = (sample, sample_from_ds, dataset_idx)

        return sample

    def _reset_dataset_iter(self, dataset_idx: int):
        """
        Reset the dataset iterator for the specified dataset index.

        This method allows resetting the dataset sampler if one of the datasets
          is exhausted but we still have more iterations.

        Parameters:
            dataset_idx (int): The index of the dataset to reset.

        Returns:
            Any: The next item from the reset dataset iterator.

        """
        self.dataset_iter[dataset_idx] = iter(self.dataset_samplers[dataset_idx])
        return next(self.dataset_iter[dataset_idx])


class WeightedStagedDataset(IterableDataset):
    """
    A dataset that samples data from multiple stages with different weights.

    use like
    ds1, ds2, ds3 = datasets of len 10, 50, 100

    dataset = WeightedStagedDataset(
        # key from here is used in stages
        datasets={"ds1": ds1, "ds2": ds2, ...},

        # stages,
        # -- dict with keys that access the datasets above with their respective weights
        # -- iters key is the number of iterations to sample from the stage, if None,
        # will sample from all until the first dataset is exhausted
        stages=[
            {"ds1": 9, "ds2": 0.1, "iters": 10},
            {"ds2": 0.5, "ds3": 0.5, "iters": 20}
            {"ds1": 0.5, "ds2": 0.5, "ds3": 0.5, "iters": 10}
        ]
    )

    """

    def __init__(
        self,
        datasets: list[str, Dataset],
        stages: list[dict[str, float | int]],
        total_iters: int = None,
        return_info: bool = False,
    ):
        super(WeightedStagedDataset, self).__init__()

        self.datasets = datasets
        self.stages = stages
        self.ds_lens = {k: len(ds) for k, ds in datasets.items()}
        self.dataset_generators = [[k, iter(RandomSampler(ds))] for k, ds in datasets.items()]
        self.stage_generators = self._create_stage_generators()

        # the max it can be is sum of all iters, but we will stop once the first dataset is exhausted
        self.total_iters = total_iters or sum(self.ds_lens.values())
        self.return_info = return_info

    def _create_stage_generators(self):
        stage_generators = []
        for stage in self.stages:
            iters = stage.pop("iters", sum(self.ds_lens.values()))

            # weight of 0 means no sampling from that dataset for this stage
            stage_weights = [stage.get(k, 0) for k in self.datasets.keys()]
            stage_get_ds_idx = WeightedRandomSampler(stage_weights, num_samples=iters, replacement=True)
            stage_generators.append(stage_get_ds_idx)

        return stage_generators

    def __iter__(self):
        def _end(n):
            return (self.total_iters is not None) and (n >= self.total_iters)

        def _metadata():
            nonlocal curr_stage, ds_idx, ds_name
            return {"curr_stage": curr_stage, "ds_idx": ds_idx, "ds_name": ds_name}

        n = 0
        for curr_stage, stage in enumerate(self.stage_generators):
            for ds_idx in stage:
                ds_name, ds_sampler = self.dataset_generators[ds_idx]

                if ds_sample_idx := next(ds_sampler, None) is None:
                    continue
                # ds_sample_idx = next(ds_sampler, None)

                sample = self.datasets[ds_name][ds_sample_idx]
                if self.return_info:
                    sample = (sample, _metadata())

                yield sample

                n += 1
                if _end(n):
                    break


if __name__ == "__main__":
    # example usage

    class MockDS(Dataset):
        def __init__(self, name, len=5):
            self.name, self.len = name, len

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return (idx, self.name)

    ds1 = MockDS("ds1", len=3)
    ds2 = MockDS("ds2", len=100)
    ds3 = MockDS("ds3", len=100)

    ds = WeightedStagedDataset(
        datasets={
            "ds1": ds1,
            "ds2": ds2,
            "ds3": ds3,
        },
        stages=[
            {"ds1": 9, "ds2": 5, "iters": 10},
            {"ds2": 0.5, "ds3": 0.5, "iters": 20},
            {"ds1": 0.5, "ds2": 0.5, "ds3": 0.5},
        ],
        return_info=True,
    )

    dsi = iter(ds)
    for i, s in enumerate(dsi):
        print(i, s)
