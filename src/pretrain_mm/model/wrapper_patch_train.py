import random

import torch
from torch import nn


def mask_single_patch(
    image_patches: torch.Tensor,
    patch_idx: int | torch.Tensor = None,
    image_patches_indices: torch.Tensor = None,
    batch_size: int = None,
    mask_to: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch_size:
        batch_size = image_patches.shape[0]
    if not patch_idx:
        patch_removed_indices = [
            [i, random.choice((image_patches_indices[i] != -1).nonzero())] for i in range(batch_size)
        ]

    patches_removed = image_patches[patch_removed_indices].clone()
    image_patches[patch_removed_indices] = mask_to
    return image_patches, patches_removed, patch_removed_indices


class PatchTrainWrapper(nn.Module):
    """trying to think how to do this so it minimally alters model but allows unsupervised pretraining task

    Args:
        nn (_type_): _description_
    """

    def __init__(self, image_patch_size: int, model: nn.Module, input_processed: bool = False):
        super().__init__()
        # hidden_size = hidden_size or model.config.hidden_size
        self.model = model
        self.patch_head(model.hidden_size, image_patch_size)
        self.input_processed = input_processed

    def forward(self, image_patches: torch.Tensor, image_patches_indices: torch.Tensor, *args, **kwargs):
        bs, n_patches, patch_size = image_patches.shape
        if not self.input_processed:
            image_patches, patches_removed, patch_removed_indices = mask_single_patch(image_patches, batch_size=bs)

        outputs = self.model(*args, **kwargs, output_hidden_states=True)
        patch_out = self.patch_head(outputs.hidden_states[-1])

        patch_loss = nn.functional.mse_loss(patch_out, patches_removed, reduce="mean")
        outputs.loss += patch_loss
        return outputs
