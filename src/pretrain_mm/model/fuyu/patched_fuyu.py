import torch
from pretrain_mm import logger

from transformers import FuyuForCausalLM


def gather_continuous_embeddings(
    self,
    word_embeddings: torch.Tensor,
    continuous_embeddings: list[torch.Tensor],
    image_patch_input_indices: torch.Tensor,
) -> torch.Tensor:
    """This function places the continuous_embeddings into the word_embeddings at the locations
    indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
    embeddings.

    Args:
        word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
        continuous_embeddings:
            Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
        shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
        indices in image_patch_input_indices for that batch element.
        image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
    """
    if word_embeddings is not None:
        logger.warn("Need to know where this is called from if we use it!")
        raise NotImplementedError("Verify that we are using this from ")

    if not (word_embeddings.shape[0] == len(continuous_embeddings)):
        continuous_embeddings = [continuous_embeddings[0] for _ in range(word_embeddings.shape[0])]
    output_embeddings = word_embeddings.clone()

    for batch_idx in range(word_embeddings.shape[0]):
        # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
        # positions in word_embeddings that we want to replace with content from continuous_embeddings.
        dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
        # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
        # want to use to replace the values in word_embeddings.
        src_indices = image_patch_input_indices[batch_idx][dst_indices]

        if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
            src_indices = src_indices[: continuous_embeddings[batch_idx].shape[0]]
            dst_indices = dst_indices[: len(src_indices)]

        output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[src_indices]
    return output_embeddings


# FuyuForCausalLM.gather_continuous_embeddings = gather_continuous_embeddings
