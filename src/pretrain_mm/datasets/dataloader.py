import random
from dataclasses import dataclass
from typing import Any

# from datasets import Dataset as HFDataset
import torch
from torch.nn.utils.rnn import pad_sequence


class BatchBase:
    is_valid: bool = True


# necessary since we can't have a dataclass with a default value and then subclass it
@dataclass
class InvalidBatch(BatchBase):
    is_valid: str = False


InvalidBatch = InvalidBatch()

# InvalidBatch = BatchBase(is_valid=False)


# class CombinedDatasetIter(torch.utils.data.IterableDataset):
#     def __init__(
#         self, datasets: list[torch.utils.data.Dataset | HFDataset], probs: list[float] = None, shuffle: bool = True
#     ):
#         super().__init__()

#         if probs:
#             assert len(datasets) == len(probs)
#             assert sum(probs) == 1
#             self.probs = probs

#         self.shuffle = shuffle
#         self.dataset_idxs = self.make_idxs(datasets)

#         self.datasets = datasets

#     def make_idxs(self, datasets=None):
#         """make idxs for each dataset"""
#         if not datasets:
#             datasets = self.datasets

#         dataset_idxs = []
#         for idx, dataset in enumerate(datasets):
#             assert isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, HFDataset)
#             dataset_idxs = list(range(len(dataset)))
#             if self.shuffle:
#                 random.shuffle(dataset_idxs)
#             dataset_idxs.append([idx, dataset_idxs])

#         return dataset_idxs

#     def __getitem__(self, index):
#         _dataset = random.choices(self.datasets, weights=self.probs, k=1)[0]

#     def __len__(self):
#         return self.length


# @dataclass
# class InvalidBatch(BatchBase):
#     is_valid = False


@dataclass
class Batch(BatchBase):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    # attach labels after
    labels: torch.Tensor = None  # field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.base_keys = ["input_ids", "attention_mask", "image_patches", "image_patches_indices"]
        if self.labels is not None:
            self.base_keys += ["labels"]

    def __getitem__(self, idx: str):
        return getattr(self, idx)

    def __setitem__(self, idx: str, value: Any):
        setattr(self, idx, value)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.image_patches = self.image_patches.pin_memory()
        self.image_patches_indices = self.image_patches_indices.pin_memory()

        if self.labels is not None:
            self.labels = self.labels.pin_memory()

        return self

    def to(self, device: str):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.image_patches = self.image_patches.to(device)
        self.image_patches_indices = self.image_patches_indices.to(device)

        if self.labels is not None:
            self.labels = self.labels.to(device)

    def keys(self):
        return self.base_keys


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

    def __call__(self, samples: list[dict[str, Any]]):
        if not all(samples):
            # rather than resample the dataset with wrapped datacollater, just return invalid and skip in training loop
            return InvalidBatch

        input_ids = pad_field("input_ids", samples, **self.pad_seq_kwargs)
        attention_mask = pad_field("attention_mask", samples, **self.pad_seq_kwargs)

        # problem with this is if we haev multiple images for an input
        image_patches = pad_field_with_check("image_patches", samples, **self.pad_seq_kwargs)
        image_patches_indices = pad_field_with_check("image_patches_indices", samples, **self.pad_seq_kwargs)

        labels = pad_field("labels", samples, **self.pad_seq_kwargs) if self.include_labels else None

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

    def _attach_extra(self, batch: Batch, samples):
        # just attach first samples extra
        if hasattr(samples[0], "extra"):
            batch.extra = samples[0].extra
            # this wont work for default model though

        if self.include_extra_loss_kwargs:
            batch.base_keys.append("extra")

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
