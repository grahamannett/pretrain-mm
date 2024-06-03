import random
from dataclasses import dataclass, make_dataclass
from functools import cache
from typing import Any, Mapping, Sequence

# from datasets import Dataset as HFDataset
import torch
from torch.nn.utils.rnn import pad_sequence


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


class BatchData:
    def __init__(self, data):
        self.data = data
        self.okay = True

    def __getitem__(self, item: str):
        return self.data[item]

    def __iter__(self):
        for key, value in self.data.items():
            yield key, value

    def __getattr__(self, item: str):
        return self.data[item]

    def to(self, device: str):
        for key, val in self.data.items():
            self.data[key] = val.to(device)
        return self

    def keys(self):
        return self.data.keys()

    def pin_memory(self):
        for key, val in self.data.items():
            self.data[key] = val.pin_memory()
        return self


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


@dataclass
class BatchPatches(Batch):
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    labels: torch.Tensor = None


_BATCH_TYPES_MADE = {}


# @cache
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

        batch = BatchData(data_out)

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
