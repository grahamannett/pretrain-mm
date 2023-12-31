import torch


class CombineEmbeddings(torch.nn.Module):
    """
    layer to combine embeddings in models that do not already allow for multimodal inputs
    """

    def __init__(self):
        super().__init__()

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
        return CombineEmbeddings.combine_embeddings(word_embeddings, continuous_embeddings, image_patch_input_indices)

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

            word_embeddings[batch_idx][dst_indices] = patch_embeddings[batch_idx].to(word_embeddings.device)[
                src_indices
            ]

            # output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[src_indices]
        return word_embeddings

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
