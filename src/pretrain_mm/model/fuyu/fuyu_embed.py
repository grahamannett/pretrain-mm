import torch

from pretrain_mm import logger


def _check_patch(importable_model: str, patch_func: callable):
    # import importlib; importlib.import_module(importable_model) # prefer using transformers

    import transformers

    _mod = getattr(transformers, importable_model)

    if hasattr(_mod, "patch"):
        logger.info(f"{importable_model} already patched")
    else:
        _mod.patch = patch_func
        logger.info(f"{importable_model}.patch added")


def get_embeddings(model, input_ids, image_patches, image_patches_indices, **kwargs):
    # should be similar to forward of model
    # ideally would just hook into model but register_forward_hook doesnt seem to include
    # the inputs for hf models
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    patch_embeddings = [
        model.vision_embed_tokens(patch.to(model.vision_embed_tokens.weight.dtype)).squeeze(0)
        for patch in image_patches
    ]
    input_embeds = model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
    return input_embeds


class PatchedCombineEmbedding:
    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations"""

        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings


class FuyuPatches:
    """
    NOTE: this is not needed as of 4.39 or something transformers.  It is actually generic enough that it might be useful elsewhere though
    layer to combine embeddings in models that do not already allow for multimodal inputs
    """

    _is_patched = False

    def __init__(self):
        super().__init__()

    def patch(self):
        return FuyuPatches.Patch(self)

    @classmethod
    def Patch(cls, model):  # call
        # allow for multiple patches
        model = cls.patch_gather_embeddings(model)
        return model

    @classmethod
    def patch_gather_embeddings(cls, model):
        model.gather_continuous_embeddings = cls.gather_continuous_embeddings
        return model

    @staticmethod
    def gather_continuous_embeddings(
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        monkey patch for `transformers.models.fuyu.FuyuForCausalLM.gather_continuous_embeddings` because
        its broken and unreliable and hf wont merge my PR
        """
        return FuyuPatches.combine_embeddings(word_embeddings, continuous_embeddings, image_patch_input_indices)

    @staticmethod
    def combine_embeddings(
        word_embeddings: torch.Tensor,
        patch_embeddings: list[torch.Tensor] | torch.Tensor,
        image_patches_indices: torch.Tensor,
    ):
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patches_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patches_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > patch_embeddings[batch_idx].shape[0]:
                src_indices = src_indices[: patch_embeddings[batch_idx].shape[0]]
                dst_indices = dst_indices[: len(src_indices)]
                raise ValueError(f"{patch_embeddings[batch_idx].shape=} does not match ")

            word_embeddings[batch_idx][dst_indices] = patch_embeddings[batch_idx].to(word_embeddings.device)[
                src_indices
            ]
        return word_embeddings

        # output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[src_indices]

    def forward(
        self,
        word_embeddings: torch.Tensor,
        patch_embeddings: list[torch.Tensor] | torch.Tensor,
        image_patches_indices: torch.Tensor,
    ):
        """
        Combine word and patch embeddings


        modified version of `gather_continuous_embeddings` in `transformers.models.fuyu.FuyuForCausalLM`
        word_embeddings ~ [batch_size, seq_len, hidden_size] (`word_embeddings` in original)
        patch_embeddings ~ [batch_size, num_total_patches, hidden_size] (`continuous_embeddings` in original)
        image_patches_indices ~ [batch_size, seq_len] (`image_patch_input_indices` in original)
        """

        return self.combine_embeddings(word_embeddings, patch_embeddings, image_patches_indices)


_check_patch("FuyuForCausalLM", FuyuPatches.patch)
