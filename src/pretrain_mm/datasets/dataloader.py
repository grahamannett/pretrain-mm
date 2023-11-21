from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    # attach labels after
    labels: torch.Tensor = field(init=False, repr=False, default=None)

    def __getitem__(self, idx: str):
        return getattr(self, idx)

    def __setitem__(self, idx: str, value: Any):
        setattr(self, idx, value)


@dataclass
class DataCollator:
    pad_token_id: int = 0
    device: str = None
    _squeeze: bool = True

    def __call__(self, samples: list[dict[str, Any]]):
        input_ids = pad_sequence([i.input_ids for i in samples], batch_first=True, padding_value=self.pad_token_id)

        image_patches = pad_sequence(
            [i.image_patches[0] for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        image_patches_indices = pad_sequence(
            [i.image_patches_indices for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        attention_mask = pad_sequence(
            [i.attention_mask for i in samples], batch_first=True, padding_value=self.pad_token_id
        )

        # problem with this is if we haev multiple images for an input
        if isinstance(image_patches, list):
            image_patches = torch.cat(image_patches)

        if self._squeeze:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            image_patches = image_patches.squeeze(0)
            image_patches_indices = image_patches_indices.squeeze(0)

        if self.device:
            input_ids = input_ids.to(self.device)
            image_patches = image_patches.to(self.device)
            image_patches_indices = image_patches_indices.to(self.device)
            attention_mask = attention_mask.to(self.device)

        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
        )
