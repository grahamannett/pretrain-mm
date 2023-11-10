from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch(UserDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_patches: torch.Tensor
    image_patches_indices: torch.Tensor

    # attach labels after
    labels: torch.Tensor = field(init=False, repr=False, default=None)


@dataclass
class DataCollator:
    pad_token_id: int = 0

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
        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
        )
