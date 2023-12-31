from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

labels_ = None


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    # attach labels after
    labels: torch.Tensor = None  # field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.base_keys = ["input_ids", "attention_mask", "image_patches", "image_patches_indices"]
        if self.labels != None:
            self.base_keys += ["labels"]

    def __getitem__(self, idx: str):
        return getattr(self, idx)

    def __setitem__(self, idx: str, value: Any):
        setattr(self, idx, value)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, device: str):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.image_patches = self.image_patches.to(device)
        self.image_patches_indices = self.image_patches_indices.to(device)

        if self.labels != None:
            self.labels = self.labels.to(device)

    def keys(self):
        return self.base_keys


@dataclass
class DataCollator:
    pad_token_id: int = 0
    device: str = None
    squeeze: bool = True
    include_labels: bool = False

    def __call__(self, samples: list[dict[str, Any]]):
        input_ids = pad_sequence([i.input_ids for i in samples], batch_first=True, padding_value=self.pad_token_id)

        # problem with this is if we haev multiple images for an input
        image_patches = pad_sequence(
            [torch.cat(i.image_patches) for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        image_patches_indices = pad_sequence(
            [i.image_patches_indices for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        attention_mask = pad_sequence(
            [i.attention_mask for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        if self.squeeze:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            image_patches = image_patches.squeeze(0)
            image_patches_indices = image_patches_indices.squeeze(0)

        labels = None
        if self.include_labels:
            labels = pad_sequence([i.labels for i in samples], batch_first=True, padding_value=self.pad_token_id)
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

        return batch
