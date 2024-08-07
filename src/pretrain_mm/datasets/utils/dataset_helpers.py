import random
from dataclasses import asdict, dataclass, field

from torch.utils.data import Dataset as TDataset
from torch.utils.data import IterableDataset as TIterableDataset

from pretrain_mm import constants, logger


@dataclass
class DatasetInitHelper:
    make: type
    sample: type = None

    task: callable = None
    encoder: callable = None
    tasks: type = None

    dataset_kwargs: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    # labels related
    IGNORE_INDEX: int = constants.IGNORE_INDEX

    # dataset.map related
    map_num_workers: int = 16
    map_load_from_cache_file: bool = True

    # fsdp related
    fsdp_enabled: bool = False

    local_rank: int = None
    is_local_main_process: bool = None

    disble_progress: bool = False

    def __post_init__(self):
        if (self.local_rank is not None) and (self.local_rank != 0):
            # disable progress bar if distributed and local rank is not 0
            self.disable_progress = True
        if self.is_local_main_process is False:
            # disable progress bar if
            self.disable_progress = True

    def _init_from_dev_config(self, ensure_set: list[str] = []):
        """allow for setting attributes if using from dev_config and DatasetConfig subclass not initialized itself

        Args:
            ensure_set (list[str], optional): _description_. Defaults to [].
        """
        from config.dev._dev_utils import get_dev_config

        _dataset_name = self.dataset_path.split("/")[1].lower()
        _config = get_dev_config(_dataset_name)

        # set attributes that are part of config but not train/test specific
        for key in ensure_set:
            if getattr(self, key) is None:
                setattr(self, key, _config[key])

        # if split is in config, use it to set config vals where data is loaded from
        if self.split in _config:
            for key, val in _config[self.split].items():
                setattr(self, key, val)

        # warn as we should avoid doing this if possible but helpful for scripts
        to_log = f"INIT `{self.__class__.__name__}` USING `get_dev_config` FOR `{_dataset_name}`."
        to_log = to_log + f"\n\tUSING SPLIT: `{self.split}`" if self.split in _config else to_log
        logger.warn(to_log)


class DatasetProgressMixin:
    _progress_bars = {}
    _task_ids = {}

    def _progress_start(self, amt: int, desc: str = "[cyan]Flattening dataset...") -> None:
        if not self.config.show_progress:
            return

        self.progress = logger.progress()

        self.traj_task = self.progress.add_task(desc, total=amt)

    def _progress_update(self, advance: int = 1):
        if not self.config.show_progress:
            return

        self.progress.update(self.traj_task, advance=advance)

    def _progress_end(self):
        if not hasattr(self, self.progress):
            return

        self.progress.stop()


class Dataset(TDataset):
    _use_as_iter: bool = False

    def get_with_transform(self, transform: callable, idx: int = None, return_extra: bool = False):
        if idx is None:
            idx = random.randint(0, len(self) - 1)

        sample = self.__getitem__(idx)
        transformed_sample = transform(sample)

        return (transformed_sample, sample, idx) if return_extra else transformed_sample

    def _reset_idx_iter(self, idx_field: str = "dataset_idxs", num_iters: int = None):
        # use this if using dataset but want iterable b/c with num_workers i need that but also want to shuffle the idxs each epoch
        setattr(
            self,
            idx_field,
            self._all_idxs.select(random.sample(range(len(self._all_idxs)), num_iters or len(self))),
        )

    def use_num_iters(self, num_iters: int, idx_field: str = "dataset_idxs"):
        self._use_as_iter, self._num_iters = True, num_iters

        if (ds_field := getattr(self, idx_field, None)) is None:
            raise ValueError(f"Dataset does not have field: {idx_field}. Cant use num iters")

        # setattr(self, idx_field, ds_field.select(random.sample(range(len(self)), num_iters)))
        # save for reset
        self._all_idxs = ds_field
        self._reset_idx_iter(idx_field, num_iters)

        return self


class IterableDataset(TIterableDataset):
    def __init__(self, dataset: Dataset, shuffle: bool = False, max_iters: int = None, as_dict: bool = False, **kwargs):
        super(IterableDataset, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.max_iters = max_iters
        self.as_dict = as_dict

    @classmethod
    def from_dataset(cls, dataset, shuffle: bool = False, max_iters: int = None, **kwargs):
        return cls(dataset, shuffle=shuffle, max_iters=max_iters)

    def __iter__(self):
        yield_from = self.dataset

        if self.shuffle:
            yield_from = list(range(len(self.dataset)))
            random.shuffle(yield_from)

        def _iter():
            for idx, sample in enumerate(yield_from):
                if self.as_dict:
                    sample = asdict(sample)
                yield sample

                if self.max_iters and idx >= self.max_iters:
                    break

        return _iter()
