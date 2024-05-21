import random
from dataclasses import dataclass, make_dataclass
from functools import cache
from typing import Any, Mapping

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


class BatchABC:
    is_valid: bool = True

    @property
    def okay(self) -> bool:
        # other checks?
        if self.is_valid:
            return True
        return False

    # allow for dict like access
    def __getitem__(self, idx: str):
        return getattr(self, idx)

    def __setitem__(self, idx: str, value: Any):
        setattr(self, idx, value)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


@dataclass
class Batch(BatchABC):
    input_ids: torch.Tensor

    def pin_memory(self):
        for key in self.__dataclass_fields__.keys():
            setattr(self, key, getattr(self, key).pin_memory())
        return self

    def keys(self):
        return self.base_keys


@dataclass
class BatchPatches(Batch):
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    # attach labels after
    labels: torch.Tensor = None  # field(init=False, repr=False, default=None)


BatchPatches.base_keys = set(BatchPatches.__dataclass_fields__.keys())


# necessary since we can't have a dataclass with a default value and then subclass it
@dataclass
class InvalidBatch(BatchABC):
    is_valid: str = False


@cache
def get_batch_dataclass(key_fields: list[tuple[str, type]]):
    """
    Dynamically creates and caches a dataclass named 'Batch' with fields specified by 'keys'.

    Args:
    - keys (tuple of str): The names of the fields for the 'Batch' dataclass.

    Returns:
    - A dynamically created 'Batch' dataclass with the specified fields.
    """
    return make_dataclass("Batch", [(key, key_type) for key, key_type in key_fields], bases=(Batch,))


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

        if self.include_extra_loss_kwargs:
            batch.base_keys.add("extra")

        return batch

    def __call__(self, samples: list[Mapping]) -> Batch:
        if not all(samples):
            # rather than resample the dataset with wrapped datacollater, just return invalid and skip in training loop
            # needs to be pickeled for dataloader workers
            return InvalidBatch()

        key_fields = [(k, type(v)) for k, v in samples[0].items()]
        BatchCls = get_batch_dataclass(key_fields)
        _out = {}

        for k, _ in key_fields:
            # looks better than ternary
            if k in _REQ_FIELDS:
                _out[k] = pad_field(k, samples, **self.pad_seq_kwargs)
            else:
                _out[k] = pad_field_with_check(k, samples, **self.pad_seq_kwargs)

        if self.squeeze or (len(samples) == 1):
            for k, _ in key_fields:
                _out[k] = _out[k].squeeze(0)

        # batch = get_batch_dataclass(**_out)
        batch = BatchCls(**_out)

        if self.device:
            batch.to(self.device)

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
