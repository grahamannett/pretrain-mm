import random
from dataclasses import dataclass, make_dataclass
from functools import cache
from typing import Any, Iterable, Mapping

# from datasets import Dataset as HFDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BatchFeature as HFBatchFeature


_REQ_FIELDS = ["input_ids"]


def has_field(field, samples):
    return field in samples[0]


def pad_field(field, samples, batch_first=True, padding_value=0):
    return pad_sequence([i[field] for i in samples], batch_first=batch_first, padding_value=padding_value)


def pad_field_with_check(field, samples, batch_first=True, padding_value=0):
    if not has_field(field, samples):
        return None
    return pad_sequence([i[field] for i in samples], batch_first=batch_first, padding_value=padding_value)


def pad_field_maybe_cat(field, samples, batch_first=True, padding_value=0):
    return pad_sequence(
        [i[field] if isinstance(i[field], torch.Tensor) else torch.cat(i[field]) for i in samples],
        batch_first=batch_first,
        padding_value=padding_value,
    )


class BatchBase:
    _is_valid: bool = True

    @property
    def okay(self) -> bool:
        # other checks?
        if self._is_valid:
            return True
        return False

    # allow for dict like access
    def __getitem__(self, item: str):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return getattr(self, list(self.keys())[item])
        else:
            raise KeyError(f"Key: {item} not found in {self.__class__.__name__}")

    def __setitem__(self, item: str, value: Any):
        setattr(self, item, value)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


# necessary since we can't have a dataclass with a default value and then subclass it
@dataclass
class InvalidBatch(BatchBase):
    _is_valid: bool = False


@dataclass
class Batch(BatchBase):
    input_ids: torch.Tensor

    def items(self):
        return ((key, getattr(self, key)) for key in self.keys())

    def keys(self):
        return self.__dataclass_fields__.keys()

    def pin_memory(self):
        for key in self.__dataclass_fields__.keys():
            setattr(self, key, getattr(self, key).pin_memory())
        return self

    def to(self, device: str):
        for key in self.__dataclass_fields__.keys():
            setattr(self, key, getattr(self, key).to(device))
        return self


class BatchDictMixin:
    def pin_memory(self):
        for key, value in self.data.items():
            self.data[key] = value.pin_memory()
        return self


class BatchData(BatchDictMixin):
    def __init__(self, data):
        self.data = data
        self.okay = True

    def __getattr__(self, item: str):
        return self.data[item]

    def __getitem__(self, item: str):
        return self.data[item]

    def __setitem__(self, item: str, value: Any):
        self.data[item] = value

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        data_str = "\n".join([f"\t{k}: {v.shape}" for k, v in self.data.items() if isinstance(v, torch.Tensor)])
        return f"{self.__class__.__name__}(\n{data_str}\n)"

    def keys(self):
        return self.data.keys()

    def to(self, device: str):
        for key in self.data.keys():
            self.data[key] = self.data[key].to(device)
        return self


class BatchFeature(HFBatchFeature, BatchDictMixin):
    okay: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        # this is the only way i believe to do it so that if you
        # set a value on one of the dicts, it will also update the data dict
        if name in self.__dict__.get("data", {}):
            self.data[name] = value
        else:
            super().__setattr__(name, value)


_BATCH_TYPES_MADE = {}


@cache  # i think cache breaks the pickling for dataloader workers
def get_batch_dataclass(key_fields: tuple[tuple[str, type], ...]) -> type:
    """
    Dynamically creates and caches a dataclass named 'Batch' with fields specified by 'keys'.

    Args:
    - keys (tuple of str): The names of the fields for the 'Batch' dataclass.

    Returns:
    - A dynamically created 'Batch' dataclass with the specified fields.
    """
    BatchCls = make_dataclass("Batch", [(key, key_type) for key, key_type in key_fields], bases=(Batch,))
    _BATCH_TYPES_MADE[key_fields] = BatchCls
    return BatchCls


@dataclass
class DataCollator:
    """
    Note:

    if image_patches are list[list[Tensor]] like default fuyu processor, need
    ```
    patches = pad_sequence(
        [i["image_patches"] if isinstance(i["image_patches"], torch.Tensor) else torch.cat(i.image_patches)
            for i in samples
        ],
        batch_first=True,
        padding_value=self.pad_token_id,
    )
    ```

    Returns:
        _type_: _description_
    """

    pad_token_id: int = 0
    device: str = None
    squeeze: bool = True
    include_labels: bool = False
    include_extra_loss_kwargs: bool = False

    pad_seq_kwargs = {
        "batch_first": True,
        "padding_value": pad_token_id,
    }

    def _attach_extra(self, batch: Batch, samples):
        # just attach first samples extra
        if hasattr(samples[0], "extra"):
            # this wont work for default model though
            batch.extra = samples[0].extra

        return batch

    def __call__(self, samples: list[Mapping]) -> Batch:
        if not all(samples):
            # rather than resample the dataset with wrapped datacollater, just return invalid and skip in training loop
            # needs to be pickeled for dataloader workers
            return InvalidBatch()

        data_out = {}
        # need as tuple to cache for get_batch_dataclass
        key_fields = tuple((k, type(v)) for k, v in samples[0].items())

        for k, _ in key_fields:
            # looks better than if else
            pad_func = pad_field if k in _REQ_FIELDS else pad_field_with_check
            data_out[k] = pad_func(k, samples, **self.pad_seq_kwargs)

            if self.device and data_out[k] is not None:
                data_out[k] = data_out[k].to(self.device)

        if self.squeeze or (len(samples) == 1):
            for k, _ in key_fields:
                data_out[k] = data_out[k].squeeze(0)

        # batch = BatchData(data_out)
        # BatchCls = get_batch_dataclass(key_fields)
        # batch = BatchCls(**data_out)
        # batch = BatchDataC(**data_out)
        batch = BatchFeature(data=data_out)
        self._attach_extra(batch, samples)

        return batch

    def prev_call__(self, samples: list[dict[str, Any]]):
        if not all(samples):
            # rather than resample the dataset with wrapped datacollater, just return invalid and skip in training loop
            # needs to be pickeled for dataloader workers
            return InvalidBatch()
        input_ids = pad_field("input_ids", samples, **self.pad_seq_kwargs)
        attention_mask = pad_field("attention_mask", samples, **self.pad_seq_kwargs)

        # problem with this is if we haev multiple images for an input
        image_patches = pad_field_with_check("image_patches", samples, **self.pad_seq_kwargs)
        image_patches_indices = pad_field_with_check("image_patches_indices", samples, **self.pad_seq_kwargs)

        labels = pad_field_with_check("labels", samples, **self.pad_seq_kwargs) if self.include_labels else None

        if self.squeeze or (len(samples) == 1):
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

            if image_patches is not None:
                image_patches = image_patches.squeeze(0)
                image_patches_indices = image_patches_indices.squeeze(0)

            if labels is not None:
                labels = labels.squeeze(0)

        batch = Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            labels=labels,
        )

        if self.device:
            batch.to(self.device)

        self._attach_extra(batch, samples)

        return batch


def replace_invalid(samples, collate_fn: callable, dataset: torch.utils.data.Dataset):
    """

    use like
    collate_fn = partial(replace_invalid, collate_fn=collate_fn, dataset=train_dataset)

    other choices could be like
        collate_fn = DataCollate.with_replace_invalid(collate_fn)
        collate_fn = DataCollator.with_replace_invalid(
                DataCollator, processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=True)

    # note:
    # # similar to https://stackoverflow.com/a/69578320
    """

    samples_len = len(samples)
    # Filter out all the Nones (corrupted examples)
    samples = list(filter(lambda x: x not in [None, False], samples))
    filtered_samples_len = len(samples)
    # Num of corrupted examples
    diff = samples_len - filtered_samples_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        samples.extend([dataset[random.randint(0, len(dataset) - 1)] for _ in range(diff)])
        return replace_invalid(samples, dataset)
    return collate_fn(samples)


class BatchIter:
    """
    An iterator to yield batches from a DataLoader for a specified number of iterations.

    Attributes:
        data (DataLoader): The DataLoader from which to fetch batches.
        num_iters_init (int): Initial number of iterations to perform.
    """

    num_iters_init: int = None
    yield_idx: bool = True

    def __init__(self, data: DataLoader, num_iters: int = None):
        """
        Initializes the BatchIter with a DataLoader and number of iterations.

        Args:
            data (DataLoader): The DataLoader source.
            num_iters (int): The number of batches to yield.
        """
        num_iters_init = num_iters or self.num_iters_init

        if num_iters_init is None:
            raise ValueError("num_iters must be set in the constructor or class attribute")

        self.num_iters_init = num_iters_init
        self.data = data

    @classmethod
    def config(cls, **kwargs):
        """
        Configures the class attributes based on the provided keyword arguments.

        Args:
            cls: The class to configure.
            **kwargs: The keyword arguments containing the attribute names and values.

        Returns:
            None
        """
        for k, v in kwargs.items():
            if hasattr(cls, k) and (v is not None):
                setattr(cls, k, v)

    @classmethod
    def setup(cls, data: DataLoader = None, num_iters: int = None, **kwargs):
        """
        Set up the data loader for training loop.

        Args:
            cls (type): The class object.
            data (DataLoader, optional): The data loader object. Defaults to None.
            num_iters (int, optional): The number of iterations. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            DataLoader: The initialized data loader instance.
        """

        cls.config(num_iters=num_iters, **kwargs)

        if data:
            # make it so you can use this to start the iterator
            inst = cls(data=data)
            return inst

    @classmethod
    def go(cls, *args, **kwargs):
        """
        This method is responsible for initializing and setting up the dataloader.

        Parameters:
        - args: positional arguments passed to the method.
        - kwargs: keyword arguments passed to the method.

        Returns:
        - The initialized and setup dataloader.
        """
        return cls.setup(*args, **kwargs)

    def __len__(self) -> int:
        """
        Returns the total number of iterations that the iterator will run.

        Returns:
            int: Number of iterations.
        """
        return self.num_iters_init

    def reset(self, reset_data: bool = True, reset_num_iters: bool = True) -> None:
        """
        Resets the data iterator and/or the iteration counter to their initial states.

        Args:
            reset_data (bool): If True, reset the data iterator.
            reset_num_iters (bool): If True, reset the number of iterations.
        """
        if reset_data:
            self.data_iter = iter(self.data)
        if reset_num_iters:
            self.num_iters = self.num_iters_init

    def __iter__(self) -> Iterable[tuple[int, Batch]]:
        """
        Creates an iterable object returning indexed batches that meet a condition.

        Yields:
            Tuple[int, Any]: A tuple of index and batch, where batches meet the condition `batch.okay`.
        """
        self.reset()

        while self.num_iters > 0:
            for idx, batch in enumerate(self.data_iter):
                if batch.okay:
                    yield idx, batch
                    self.num_iters -= 1
                    if self.num_iters == 0:
                        return
            self.reset(reset_data=True, reset_num_iters=False)
